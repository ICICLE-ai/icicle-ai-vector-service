"""Vector-space reranking.

These methods operate purely on the embeddings already stored in Qdrant — no
model runs and the raw query text is never needed. They are cheap (sub-millisecond)
but they can only ever reason about vector geometry. For true relevance
judgement over the passage text, see ``cross_encoder.py``.
"""

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
    """Maximal Marginal Relevance.

    Greedily picks the candidate maximising
    ``lambda * relevance - (1 - lambda) * max_similarity_to_already_picked``.
    Note this deliberately *trades away* some relevance to avoid returning near
    duplicates: lambda=1.0 is pure relevance (identical to the vector ranking),
    lambda=0.0 is pure diversity.
    """
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
        chosen = remaining.pop(best_idx)
        # The MMR objective value, not the raw similarity — this is what the
        # ordering is actually based on.
        chosen["rerank_score"] = best_score
        selected.append(chosen)

    for item in selected:
        item.pop("embedding", None)

    return selected


def cosine_rescore(
    candidates: list[dict[str, Any]],
    query_embedding: list[float],
    top_k: int,
) -> list[dict[str, Any]]:
    """Recompute cosine similarity against the query and re-sort.

    Qdrant already returns points ordered by cosine distance, so in the common
    case this reproduces the same ranking. It is still useful as an explicit,
    exact rescoring step: Qdrant's HNSW search is *approximate*, and its score
    reflects the distance metric the collection was created with. This computes
    the exact cosine similarity over the returned vectors, which can reorder
    near-ties that the ANN traversal got slightly wrong.
    """
    if not candidates:
        return []

    for candidate in candidates:
        embedding = candidate.get("embedding") or []
        candidate["rerank_score"] = (
            cosine_sim(embedding, query_embedding) if embedding else candidate.get("score", 0.0)
        )

    candidates.sort(key=lambda item: item["rerank_score"], reverse=True)
    top = candidates[:top_k]
    for item in top:
        item.pop("embedding", None)
    return top
