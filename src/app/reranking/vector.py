from __future__ import annotations

from math import sqrt
from typing import Any


def cosine_sim(vec_a: list[float], vec_b: list[float]) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b, strict=False):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (sqrt(norm_a) * sqrt(norm_b))


def mmr_rerank(
    candidates: list[dict[str, Any]],
    query_embedding: list[float],
    top_k: int,
    lambda_: float,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    selected: list[dict[str, Any]] = []
    remaining = candidates[:]

    while remaining and len(selected) < top_k:
        best_idx = 0
        best_score = float("-inf")
        for idx, candidate in enumerate(remaining):
            relevance = candidate.get("score", 0.0)
            if not selected:
                mmr_score = relevance
            else:
                diversity = max(
                    cosine_sim(candidate["embedding"], chosen["embedding"])
                    for chosen in selected
                )
                mmr_score = lambda_ * relevance - (1.0 - lambda_) * diversity
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        selected.append(remaining.pop(best_idx))

    for item in selected:
        item.pop("embedding", None)

    return selected
