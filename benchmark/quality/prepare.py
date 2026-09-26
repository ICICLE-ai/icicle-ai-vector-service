#!/usr/bin/env python3
"""Download BEIR SciFact and embed it locally.

Why SciFact: it is a standard BEIR retrieval benchmark, small enough to embed on
a laptop, scientific in domain (a reasonable proxy for ICICLE's users), and it
ships relevance judgements — without which "does reranking help?" is unanswerable.

Why BEIR rather than MS MARCO: `ms-marco-MiniLM-L-6-v2`, one of the rerankers
under test, was *trained* on MS MARCO. Evaluating it there would measure
memorisation. BEIR is out-of-domain for both rerankers, so the comparison is fair.

Why bge-base-en-v1.5 for embeddings: open weights (reproducible), 768 dimensions
(the common production size), and strong MTEB retrieval scores — so the
bi-encoder baseline is a real baseline, not a strawman the reranker can
trivially beat. It is held FIXED across every condition; the rerank method is
the only thing that varies.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import requests

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
# bge-* retrieval models expect this prefix on queries only, not on documents.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def download(name: str, root: Path) -> Path:
    target = root / name
    if (target / "corpus.jsonl").exists():
        print(f"  {name} already present at {target}")
        return target
    url = BEIR_URL.format(name=name)
    print(f"  downloading {url}")
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        archive.extractall(root)
    return target


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    """query_id -> {doc_id: relevance}. First line is a header."""
    qrels: dict[str, dict[str, int]] = {}
    with path.open() as handle:
        next(handle)
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            query_id, doc_id, score = parts[0], parts[1], int(parts[2])
            qrels.setdefault(query_id, {})[doc_id] = score
    return qrels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--out", default="benchmark/quality/data")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"BEIR dataset: {args.dataset}")
    root = download(args.dataset, out)

    corpus = read_jsonl(root / "corpus.jsonl")
    queries = read_jsonl(root / "queries.jsonl")
    qrels = read_qrels(root / "qrels" / "test.tsv")
    # Only queries with judgements are evaluable.
    queries = [q for q in queries if q["_id"] in qrels]

    print(f"  corpus  : {len(corpus):,} documents")
    print(f"  queries : {len(queries):,} with relevance judgements")
    print(f"  qrels   : {sum(len(v) for v in qrels.values()):,} judged pairs")

    from sentence_transformers import SentenceTransformer

    print(f"\nEmbedding model: {args.model}")
    model = SentenceTransformer(args.model, device="cpu")
    dim = model.get_sentence_embedding_dimension()
    print(f"  dimension: {dim}")

    # BEIR convention: title and text concatenated for the document representation.
    doc_texts = [f"{d.get('title', '')} {d.get('text', '')}".strip() for d in corpus]
    query_texts = [QUERY_PREFIX + q["text"] for q in queries]

    started = time.perf_counter()
    print(f"\n  encoding {len(doc_texts):,} documents ...")
    doc_vectors = model.encode(
        doc_texts, batch_size=args.batch_size, normalize_embeddings=True,
        show_progress_bar=True, convert_to_numpy=True,
    ).astype(np.float32)
    print(f"  encoding {len(query_texts):,} queries ...")
    query_vectors = model.encode(
        query_texts, batch_size=args.batch_size, normalize_embeddings=True,
        show_progress_bar=True, convert_to_numpy=True,
    ).astype(np.float32)
    elapsed = time.perf_counter() - started

    np.save(out / "doc_vectors.npy", doc_vectors)
    np.save(out / "query_vectors.npy", query_vectors)
    (out / "manifest.json").write_text(json.dumps({
        "dataset": args.dataset,
        "embedding_model": args.model,
        "dimension": int(dim),
        "normalized": True,
        "query_prefix": QUERY_PREFIX,
        "documents": len(corpus),
        "queries": len(queries),
        "judged_pairs": sum(len(v) for v in qrels.values()),
        "encode_seconds": round(elapsed, 1),
        "doc_ids": [d["_id"] for d in corpus],
        "query_ids": [q["_id"] for q in queries],
        "query_texts": [q["text"] for q in queries],
    }, indent=2))
    (out / "qrels.json").write_text(json.dumps(qrels, indent=2))

    print(f"\nEncoded in {elapsed/60:.1f} min. Wrote {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
