#!/usr/bin/env python3
"""Upload the embedded corpus into the vector service.

The BEIR document id is carried in each point's metadata, because the service
assigns its own UUIDs; evaluation maps results back to qrels through that field.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import httpx
import numpy as np


async def worker(
    client: httpx.AsyncClient, url: str, headers: dict, queue: asyncio.Queue,
    collection: str, model_name: str, failures: list,
) -> None:
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return
        doc_id, vector, text = item
        try:
            response = await client.post(
                f"{url}/v1/embeddings",
                headers=headers,
                json={
                    "embedding": vector,
                    "collection": collection,
                    "chunks": [text],
                    "metadata": {"doc_id": doc_id},
                    "embedding_model": model_name,
                },
            )
            if response.status_code != 201:
                failures.append((doc_id, response.status_code, response.text[:120]))
        except Exception as exc:  # network hiccup on a long upload
            failures.append((doc_id, "exception", str(exc)[:120]))
        finally:
            queue.task_done()


async def main_async(args) -> int:
    data = Path(args.data)
    manifest = json.loads((data / "manifest.json").read_text())
    vectors = np.load(data / "doc_vectors.npy")
    doc_ids = manifest["doc_ids"]

    corpus_path = data / manifest["dataset"] / "corpus.jsonl"
    texts = {}
    with corpus_path.open() as handle:
        for line in handle:
            if line.strip():
                d = json.loads(line)
                texts[d["_id"]] = f"{d.get('title','')} {d.get('text','')}".strip()

    token = os.environ["TAPIS_TOKEN"]
    url = os.environ["BASE_URL"].rstrip("/")
    headers = {"X-Tapis-Token": token, "Content-Type": "application/json"}

    print(f"Uploading {len(doc_ids):,} documents ({manifest['dimension']}d) "
          f"to '{args.collection}' with {args.concurrency} workers")

    queue: asyncio.Queue = asyncio.Queue(maxsize=args.concurrency * 4)
    failures: list = []
    started = time.perf_counter()

    limits = httpx.Limits(max_connections=args.concurrency + 5)
    async with httpx.AsyncClient(timeout=120.0, limits=limits) as client:
        workers = [
            asyncio.create_task(
                worker(client, url, headers, queue, args.collection,
                       manifest["embedding_model"], failures)
            )
            for _ in range(args.concurrency)
        ]
        for i, doc_id in enumerate(doc_ids):
            await queue.put((doc_id, vectors[i].tolist(), texts.get(doc_id, "")[:4000]))
            if (i + 1) % 500 == 0:
                rate = (i + 1) / (time.perf_counter() - started)
                print(f"  {i+1:,}/{len(doc_ids):,}  ({rate:.0f}/s, {len(failures)} failed)")
        for _ in workers:
            await queue.put(None)
        await asyncio.gather(*workers)

    elapsed = time.perf_counter() - started
    print(f"\nUploaded {len(doc_ids) - len(failures):,}/{len(doc_ids):,} in {elapsed/60:.1f} min")
    if failures:
        print(f"  {len(failures)} failures, first few:")
        for f in failures[:5]:
            print(f"    {f}")
    return 1 if len(failures) > len(doc_ids) * 0.01 else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="benchmark/quality/data")
    parser.add_argument("--collection", default="scifact")
    parser.add_argument("--concurrency", type=int, default=12)
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
