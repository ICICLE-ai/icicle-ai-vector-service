#!/usr/bin/env python3
"""Per-endpoint latency for every v1 endpoint, end to end over the network.

Every figure includes TLS, the public internet hop to the Tapis pod, FastAPI,
Qdrant and the return trip. Unauthenticated /healthz is the floor.

Destructive endpoints run against a throwaway collection created for the purpose.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import uuid
from pathlib import Path

import httpx

SCRATCH = "endpoint_probe"


def percentiles(samples: list[float]) -> dict:
    s = sorted(samples)
    n = len(s)
    def pct(p):
        return round(s[min(n - 1, int(p * n))], 1)
    return {"n": n, "p50": round(statistics.median(s), 1),
            "p95": pct(0.95), "p99": pct(0.99),
            "min": round(s[0], 1), "max": round(s[-1], 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collection", default="scifact")
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--cross-samples", type=int, default=8)
    ap.add_argument("--out", default="benchmark/results")
    args = ap.parse_args()

    url = os.environ["BASE_URL"].rstrip("/")
    h = {"X-Tapis-Token": os.environ["TAPIS_TOKEN"], "Content-Type": "application/json"}
    vec = [0.01] * args.dim
    results = []

    with httpx.Client(timeout=300.0) as c:
        stats = c.get(f"{url}/v1/collections/{args.collection}", headers=h).json()
        corpus = stats["points"]
        print(f"corpus: {args.collection}, {corpus:,} points, {stats['vector_dim']}d\n")

        # Seed a throwaway collection for the destructive endpoints.
        ids = []
        for i in range(args.samples + 10):
            r = c.post(f"{url}/v1/embeddings", headers=h, json={
                "embedding": vec, "collection": SCRATCH, "chunks": [f"probe {i}"],
                "metadata": {"i": i}, "embedding_model": "probe"})
            if r.status_code == 201:
                ids.append(r.json()["id"])
        probe_id = ids[0]

        def timed(fn, n):
            samples = []
            fn()  # warm
            for _ in range(n):
                t0 = time.perf_counter()
                fn()
                samples.append((time.perf_counter() - t0) * 1000)
            return samples

        def add(name, method, path, fn, n=None, note=""):
            s = timed(fn, n or args.samples)
            row = {"endpoint": f"{method} {path}", "note": note, **percentiles(s)}
            results.append(row)
            print(f"  {row['endpoint']:<44}{row['p50']:>9.0f}{row['p95']:>9.0f}"
                  f"{row['p99']:>9.0f}   n={row['n']}")

        print(f"  {'endpoint':<44}{'p50':>9}{'p95':>9}{'p99':>9}")
        print(f"  {'-'*44}{'-'*9}{'-'*9}{'-'*9}")

        add("GET", "/healthz", "GET", lambda: c.get(f"{url}/healthz"),
            note="unauthenticated; network floor")
        add("GET", "/v1/rerank/methods", "GET",
            lambda: c.get(f"{url}/v1/rerank/methods", headers=h), note="static")
        add("GET", "/v1/collections", "GET",
            lambda: c.get(f"{url}/v1/collections", headers=h),
            note="scales with the caller's collection count")
        add("GET", "/v1/collections/{c}", "GET",
            lambda: c.get(f"{url}/v1/collections/{args.collection}", headers=h))
        add("GET", "/v1/collections/{c}/embeddings", "GET",
            lambda: c.get(f"{url}/v1/collections/{args.collection}/embeddings?limit=50",
                          headers=h), note="page of 50")
        add("POST", "/v1/embeddings", "POST",
            lambda: c.post(f"{url}/v1/embeddings", headers=h, json={
                "embedding": vec, "collection": SCRATCH, "chunks": ["probe"],
                "embedding_model": "probe"}), note="3 Qdrant round trips")
        add("GET", "/v1/embeddings/{id}", "GET",
            lambda: c.get(f"{url}/v1/embeddings/{probe_id}?collection={SCRATCH}", headers=h))
        add("PUT", "/v1/embeddings/{id}", "PUT",
            lambda: c.put(f"{url}/v1/embeddings/{probe_id}?collection={SCRATCH}",
                          headers=h, json={"topic": f"t{uuid.uuid4().hex[:6]}"}))
        add("POST", "/v1/retrieve", "POST",
            lambda: c.post(f"{url}/v1/retrieve", headers=h, json={
                "query_embedding": vec, "collection": args.collection, "top_k": 10}),
            note=f"top_k=10 over {corpus:,} points")
        add("POST", "/v1/rerank (mmr)", "POST",
            lambda: c.post(f"{url}/v1/rerank", headers=h, json={
                "query_embedding": vec, "collection": args.collection,
                "method": "mmr", "top_k": 10, "fetch_k": 50}), note="fetch_k=50")
        add("POST", "/v1/rerank (cosine_rescore)", "POST",
            lambda: c.post(f"{url}/v1/rerank", headers=h, json={
                "query_embedding": vec, "collection": args.collection,
                "method": "cosine_rescore", "top_k": 10, "fetch_k": 50}), note="fetch_k=50")
        add("POST", "/v1/rerank (cross_encoder MiniLM)", "POST",
            lambda: c.post(f"{url}/v1/rerank", headers=h, json={
                "query_embedding": vec, "query_text": "what causes cell division",
                "collection": args.collection, "method": "cross_encoder",
                "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "top_k": 10, "fetch_k": 50}),
            n=args.cross_samples, note="fetch_k=50, 50 model passes")
        add("POST", "/v1/rerank (cross_encoder bge)", "POST",
            lambda: c.post(f"{url}/v1/rerank", headers=h, json={
                "query_embedding": vec, "query_text": "what causes cell division",
                "collection": args.collection, "method": "cross_encoder",
                "rerank_model": "BAAI/bge-reranker-base", "top_k": 10, "fetch_k": 50}),
            n=max(3, args.cross_samples // 2), note="fetch_k=50, 50 model passes")

        # Destructive: each needs a fresh target, so time them individually.
        del_samples = []
        for eid in ids[1:1 + args.samples]:
            t0 = time.perf_counter()
            c.delete(f"{url}/v1/embeddings/{eid}?collection={SCRATCH}", headers=h)
            del_samples.append((time.perf_counter() - t0) * 1000)
        row = {"endpoint": "DELETE /v1/embeddings/{id}", "note": "", **percentiles(del_samples)}
        results.append(row)
        print(f"  {row['endpoint']:<44}{row['p50']:>9.0f}{row['p95']:>9.0f}{row['p99']:>9.0f}   n={row['n']}")

        bulk = []
        for i in range(10):
            for j in range(5):
                c.post(f"{url}/v1/embeddings", headers=h, json={
                    "embedding": vec, "collection": SCRATCH, "chunks": ["bulk"],
                    "metadata": {"batch": i}, "embedding_model": "probe"})
            t0 = time.perf_counter()
            c.post(f"{url}/v1/embeddings/bulk-delete", headers=h, json={
                "collection": SCRATCH, "filter": {"conditions": {"batch": i}}})
            bulk.append((time.perf_counter() - t0) * 1000)
        row = {"endpoint": "POST /v1/embeddings/bulk-delete", "note": "5 points by metadata filter",
               **percentiles(bulk)}
        results.append(row)
        print(f"  {row['endpoint']:<44}{row['p50']:>9.0f}{row['p95']:>9.0f}{row['p99']:>9.0f}   n={row['n']}")

        drop = []
        for i in range(10):
            name = f"{SCRATCH}_drop_{i}"
            c.post(f"{url}/v1/embeddings", headers=h, json={
                "embedding": vec, "collection": name, "chunks": ["x"],
                "embedding_model": "probe"})
            t0 = time.perf_counter()
            c.delete(f"{url}/v1/collections/{name}", headers=h)
            drop.append((time.perf_counter() - t0) * 1000)
        row = {"endpoint": "DELETE /v1/collections/{c}", "note": "drops a 1-point collection",
               **percentiles(drop)}
        results.append(row)
        print(f"  {row['endpoint']:<44}{row['p50']:>9.0f}{row['p95']:>9.0f}{row['p99']:>9.0f}   n={row['n']}")

        c.delete(f"{url}/v1/collections/{SCRATCH}", headers=h)

    out = Path(args.out) / f"endpoints-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({
        "collection": args.collection, "corpus_points": corpus,
        "dimension": args.dim, "fetch_k": 50, "top_k": 10,
        "note": "client-side, single user, over the public internet to the Tapis pod",
        "endpoints": results}, indent=2))
    print(f"\nWrote {out}/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
