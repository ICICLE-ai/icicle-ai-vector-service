#!/usr/bin/env python3
"""Does collection count degrade the service?

This is the risk the per-user-collection design introduces. Qdrant recommends
against many collections ("each collection carries its own resource overhead")
and Qdrant Cloud caps a cluster at 1000 by default. At 200 users with a few
collections each, this deployment lands in the hundreds.

A single user can own many collections, so this is measurable with one token.

Method: create collections in batches. After each batch, measure

  GET  /v1/collections   — inspects every collection the caller owns
  POST /v1/retrieve      — one search in ONE collection
  POST /v1/embeddings    — one write

Each sampled several times, median reported.

The listing endpoint is expected to grow, since its cost scales with the caller's
collection count. A search is expected to stay flat, because it addresses exactly
one collection no matter how many exist. If search degrades, collection count is
imposing a global cost and the design has a ceiling below the documented 1000.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import httpx

PREFIX = "scaletest"


def median_ms(fn, n: int) -> float:
    samples = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    return round(statistics.median(samples), 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max", type=int, default=200)
    parser.add_argument("--step", type=int, default=25)
    parser.add_argument("--dim", type=int, default=128,
                        help="small on purpose; this measures collection count, not vectors")
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--out", default="benchmark/results")
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()

    url = os.environ["BASE_URL"].rstrip("/")
    headers = {"X-Tapis-Token": os.environ["TAPIS_TOKEN"], "Content-Type": "application/json"}
    vector = [0.01] * args.dim

    with httpx.Client(timeout=180.0) as client:
        if args.cleanup:
            listing = client.get(f"{url}/v1/collections", headers=headers).json()
            targets = [c["collection"] for c in listing["collections"]
                       if c["collection"].startswith(PREFIX)]
            print(f"deleting {len(targets)} {PREFIX}* collections ...")
            for name in targets:
                client.delete(f"{url}/v1/collections/{name}", headers=headers)
            print("done")
            return 0

        baseline = len(client.get(f"{url}/v1/collections", headers=headers).json()["collections"])
        print(f"starting from {baseline} existing collection(s)")
        print(f"creating up to {args.max} more, measuring every {args.step}\n")
        print(f"{'collections':>12} {'basic ms':>10} {'full ms':>10} "
              f"{'search ms':>11} {'write ms':>10}")
        print(f"{'-' * 12} {'-' * 10} {'-' * 10} {'-' * 11} {'-' * 10}")

        def do_list_basic():
            client.get(f"{url}/v1/collections?detail=basic",
                       headers=headers).raise_for_status()

        def do_list_full():
            client.get(f"{url}/v1/collections?detail=full",
                       headers=headers).raise_for_status()

        def do_search():
            client.post(f"{url}/v1/retrieve", headers=headers, json={
                "query_embedding": vector, "collection": f"{PREFIX}_0",
                "top_k": 10}).raise_for_status()

        def do_write():
            client.post(f"{url}/v1/embeddings", headers=headers, json={
                "embedding": vector, "collection": f"{PREFIX}_0",
                "chunks": ["probe"], "embedding_model": "scaletest"}).raise_for_status()

        rows = []
        created = 0
        while True:
            if created > 0:
                row = {
                    "collections": baseline + created,
                    "list_basic_ms": median_ms(do_list_basic, args.samples),
                    "list_full_ms": median_ms(do_list_full, args.samples),
                    "search_ms": median_ms(do_search, args.samples),
                    "write_ms": median_ms(do_write, args.samples),
                }
                rows.append(row)
                print(f"{row['collections']:>12} {row['list_basic_ms']:>10} "
                      f"{row['list_full_ms']:>10} {row['search_ms']:>11} "
                      f"{row['write_ms']:>10}")
            if created >= args.max:
                break
            for i in range(created, min(created + args.step, args.max)):
                r = client.post(f"{url}/v1/embeddings", headers=headers, json={
                    "embedding": vector, "collection": f"{PREFIX}_{i}",
                    "chunks": [f"probe {i}"], "embedding_model": "scaletest"})
                if r.status_code != 201:
                    print(f"  !! create {PREFIX}_{i} -> {r.status_code} {r.text[:90]}")
            created = min(created + args.step, args.max)

    out = Path(args.out) / f"collscale-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(
        {"dim": args.dim, "samples_per_point": args.samples, "rows": rows}, indent=2))
    print(f"\nWrote {out}/results.json")
    print("Clean up with: python benchmark/scaling/collection_count.py --cleanup")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
