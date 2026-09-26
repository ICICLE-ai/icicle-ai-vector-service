#!/usr/bin/env python3
"""Measure retrieval quality and latency for every rerank method.

One query set, one corpus, one embedding model. The ONLY thing that varies
between conditions is the rerank method, so any difference in nDCG is
attributable to reranking rather than to the retriever or the data.

Metrics
-------
nDCG@10   the BEIR standard; rewards putting relevant documents near the top.
Recall@k  what fraction of the judged-relevant documents were retrieved at all.
          Recall@fetch_k is the CEILING on what any reranker can achieve — it
          can only reorder what the bi-encoder already fetched.
latency   wall-clock per request, measured client-side, single user.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path

import httpx
import numpy as np


def dcg(relevances: list[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(relevances))


def ndcg_at_k(ranked_ids: list[str], judged: dict[str, int], k: int) -> float:
    gains = [judged.get(doc_id, 0) for doc_id in ranked_ids[:k]]
    ideal = sorted(judged.values(), reverse=True)[:k]
    denominator = dcg(ideal)
    return dcg(gains) / denominator if denominator else 0.0


def recall_at_k(ranked_ids: list[str], judged: dict[str, int], k: int) -> float:
    relevant = {d for d, score in judged.items() if score > 0}
    if not relevant:
        return 0.0
    return len(relevant & set(ranked_ids[:k])) / len(relevant)


def doc_ids_of(results: list[dict]) -> list[str]:
    return [r.get("metadata", {}).get("doc_id", "") for r in results]


CONDITIONS = [
    # (label, endpoint, extra request fields)
    ("retrieve (bi-encoder baseline)", "/v1/retrieve", {}),
    ("mmr (lambda=0.7)", "/v1/rerank", {"method": "mmr", "lambda": 0.7}),
    ("cosine_rescore", "/v1/rerank", {"method": "cosine_rescore"}),
    ("cross_encoder: MiniLM-L-6", "/v1/rerank",
     {"method": "cross_encoder", "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"}),
    ("cross_encoder: bge-reranker-base", "/v1/rerank",
     {"method": "cross_encoder", "rerank_model": "BAAI/bge-reranker-base"}),
]


def run_condition(client, url, headers, label, endpoint, extra, queries, qrels,
                  collection, top_k, fetch_k, warmup) -> dict:
    ndcgs, recalls_k, recalls_fetch, latencies = [], [], [], []
    errors = 0

    for n, (query_id, vector, text) in enumerate(queries):
        body = {"query_embedding": vector, "collection": collection, "top_k": top_k}
        if endpoint == "/v1/rerank":
            body["fetch_k"] = fetch_k
            body.update(extra)
            if extra.get("method") == "cross_encoder":
                body["query_text"] = text

        started = time.perf_counter()
        try:
            response = client.post(f"{url}{endpoint}", headers=headers, json=body)
            elapsed = (time.perf_counter() - started) * 1000
        except Exception:
            errors += 1
            continue

        if response.status_code != 200:
            errors += 1
            continue

        # Discard warm-up requests: the first cross-encoder call loads the model.
        if n >= warmup:
            latencies.append(elapsed)

        ranked = doc_ids_of(response.json()["results"])
        judged = qrels.get(query_id, {})
        ndcgs.append(ndcg_at_k(ranked, judged, top_k))
        recalls_k.append(recall_at_k(ranked, judged, top_k))

    return {
        "condition": label,
        # Per-query scores are retained so a paired significance test (bootstrap
        # or t-test) can be run afterwards; means alone cannot support one.
        "per_query_ndcg": [round(v, 6) for v in ndcgs],
        "queries": len(ndcgs),
        "errors": errors,
        f"ndcg@{top_k}": round(statistics.mean(ndcgs), 4) if ndcgs else 0.0,
        f"recall@{top_k}": round(statistics.mean(recalls_k), 4) if recalls_k else 0.0,
        "latency_p50_ms": round(statistics.median(latencies), 1) if latencies else 0.0,
        "latency_p95_ms": round(
            sorted(latencies)[int(len(latencies) * 0.95)], 1) if len(latencies) > 20 else 0.0,
        "latency_mean_ms": round(statistics.mean(latencies), 1) if latencies else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="benchmark/quality/data")
    parser.add_argument("--collection", default="scifact")
    parser.add_argument("--out", default="benchmark/results")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--fetch-k", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0, help="cap queries (0 = all)")
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    data = Path(args.data)
    manifest = json.loads((data / "manifest.json").read_text())
    qrels = json.loads((data / "qrels.json").read_text())
    vectors = np.load(data / "query_vectors.npy")

    queries = list(zip(manifest["query_ids"],
                       [v.tolist() for v in vectors],
                       manifest["query_texts"]))
    if args.limit:
        queries = queries[: args.limit]

    url = os.environ["BASE_URL"].rstrip("/")
    headers = {"X-Tapis-Token": os.environ["TAPIS_TOKEN"], "Content-Type": "application/json"}

    print(f"Dataset        : {manifest['dataset']}  "
          f"({manifest['documents']:,} docs, {len(queries)} queries)")
    print(f"Embedding model: {manifest['embedding_model']} ({manifest['dimension']}d)")
    print(f"top_k={args.top_k}  fetch_k={args.fetch_k}\n")

    rows = []
    with httpx.Client(timeout=180.0) as client:
        # Recall@fetch_k: the ceiling any reranker is bounded by.
        ceiling = []
        for query_id, vector, _ in queries:
            r = client.post(f"{url}/v1/retrieve", headers=headers, json={
                "query_embedding": vector, "collection": args.collection,
                "top_k": min(args.fetch_k, 100)})
            if r.status_code == 200:
                ceiling.append(recall_at_k(doc_ids_of(r.json()["results"]),
                                           qrels.get(query_id, {}), args.fetch_k))
        recall_ceiling = round(statistics.mean(ceiling), 4) if ceiling else 0.0
        print(f"Recall@{args.fetch_k} (reranking ceiling): {recall_ceiling}\n")

        for label, endpoint, extra in CONDITIONS:
            print(f"  running: {label} ...", flush=True)
            row = run_condition(client, url, headers, label, endpoint, extra,
                                queries, qrels, args.collection,
                                args.top_k, args.fetch_k, args.warmup)
            rows.append(row)
            print(f"    nDCG@{args.top_k}={row[f'ndcg@{args.top_k}']}  "
                  f"p50={row['latency_p50_ms']}ms  errors={row['errors']}")

    out = Path(args.out) / f"quality-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({
        "dataset": manifest["dataset"],
        "embedding_model": manifest["embedding_model"],
        "dimension": manifest["dimension"],
        "documents": manifest["documents"],
        "queries_evaluated": len(queries),
        "top_k": args.top_k,
        "fetch_k": args.fetch_k,
        f"recall_at_fetch_k": recall_ceiling,
        "conditions": rows,
    }, indent=2))
    print(f"\nWrote {out}/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
