#!/usr/bin/env python3
"""Compare the scalar and vectorised MMR implementations on identical inputs.

The service uses the vectorised form. The scalar form is reproduced here from the
pre-1.0.0 implementation so the two can be timed against the same candidate sets.

Local CPU measurement only — no service, no network.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from math import sqrt
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.app.reranking.vector import mmr_rerank as mmr_vectorised  # noqa: E402


def cosine_scalar(a: list[float], b: list[float]) -> float:
    dot = norm_a = norm_b = 0.0
    for x, y in zip(a, b, strict=False):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (sqrt(norm_a) * sqrt(norm_b))


def mmr_scalar(candidates, query_embedding, top_k, lambda_):
    """The pre-1.0.0 implementation: Python loops over every pair."""
    if not candidates:
        return []
    selected, remaining = [], candidates[:]
    while remaining and len(selected) < top_k:
        best_idx, best_score = 0, float("-inf")
        for idx, candidate in enumerate(remaining):
            relevance = candidate.get("score", 0.0)
            if not selected:
                score = relevance
            else:
                diversity = max(
                    cosine_scalar(candidate["embedding"], chosen["embedding"])
                    for chosen in selected
                )
                score = lambda_ * relevance - (1.0 - lambda_) * diversity
            if score > best_score:
                best_score, best_idx = score, idx
        chosen = remaining.pop(best_idx)
        chosen["rerank_score"] = best_score
        selected.append(chosen)
    for item in selected:
        item.pop("embedding", None)
    return selected


def make(n: int, dim: int, seed: int = 7):
    rng = random.Random(seed)
    return [
        {"id": str(i), "score": 1.0 - i / 1000,
         "embedding": [rng.random() for _ in range(dim)],
         "collection": "c", "topic": None, "text": None,
         "chunks": ["x"], "metadata": {}}
        for i in range(n)
    ]


def time_it(fn, candidates, dim, top_k, reps) -> float:
    samples = []
    for _ in range(reps):
        work = [dict(c) for c in candidates]
        start = time.perf_counter()
        fn(work, [0.5] * dim, top_k, 0.7)
        samples.append((time.perf_counter() - start) * 1000)
    return round(statistics.median(samples), 2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="benchmark/results")
    args = ap.parse_args()

    shapes = [(50, 768), (50, 1024), (50, 4096), (200, 768), (500, 768)]
    print(f"MMR, top_k={args.top_k}, median of {args.reps} runs, identical inputs\n")
    print(f"{'fetch_k':>8}{'dim':>7}{'scalar':>12}{'vectorised':>13}{'speed-up':>11}")
    print(f"{'-'*8}{'-'*7}{'-'*12}{'-'*13}{'-'*11}")

    rows = []
    for n, dim in shapes:
        cands = make(n, dim)
        scalar = time_it(mmr_scalar, cands, dim, args.top_k, args.reps)
        vector = time_it(mmr_vectorised, cands, dim, args.top_k, args.reps)
        rows.append({"fetch_k": n, "dim": dim, "scalar_ms": scalar,
                     "vectorised_ms": vector, "speedup": round(scalar / vector, 1)})
        print(f"{n:>8}{dim:>7}{scalar:>10.2f}ms{vector:>11.2f}ms{scalar/vector:>10.1f}x")

    out = Path(args.out) / f"rerank-impl-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(
        {"top_k": args.top_k, "reps": args.reps, "numpy": np.__version__,
         "rows": rows}, indent=2))
    print(f"\nWrote {out}/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
