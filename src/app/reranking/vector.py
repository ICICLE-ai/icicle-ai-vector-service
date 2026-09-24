"""Vector-space reranking.

These methods operate on the embeddings returned by Qdrant — no model runs and the
raw query text is never needed. For relevance judgement over passage text, see
:mod:`app.reranking.cross_encoder`.

The similarity maths is vectorised with NumPy rather than written as Python loops.
MMR compares every candidate against every already-selected one, which is
O(top_k^2 * fetch_k * dim) — far too slow in Python to run on the event loop.
"""

from __future__ import annotations

from math import sqrt
from typing import Any

import numpy as np


def cosine_sim(vec_a: list[float], vec_b: list[float]) -> float:
    """Cosine similarity between two vectors.

    Kept as a scalar helper for callers and tests; the rerankers below work on
    whole matrices instead of calling this per pair.
    """
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


def _embedding_matrix(candidates: list[dict[str, Any]]) -> np.ndarray | None:
    """Stack candidate embeddings into an (n, dim) float32 matrix.

    Returns None when the embeddings are missing or ragged — Qdrant enforces one
    dimension per collection so that should not happen, but a caller that lost
    vectors somewhere must degrade rather than raise.
    """
    vectors = [candidate.get("embedding") or [] for candidate in candidates]
    widths = {len(v) for v in vectors}
    if len(widths) != 1 or widths == {0}:
        return None
    return np.asarray(vectors, dtype=np.float32)


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    """Row-normalise so a dot product *is* the cosine similarity.

    Normalising once up front turns every later comparison into a plain matrix
    multiply, which is the whole point of doing this in NumPy.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # A zero vector has no direction; leave it at zero rather than dividing by 0,
    # which matches cosine_sim() returning 0.0 for that case.
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)


def _finalise(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip vectors before results leave the service."""
    for item in items:
        item.pop("embedding", None)
    return items


def mmr_rerank(
    candidates: list[dict[str, Any]],
    query_embedding: list[float],
    top_k: int,
    lambda_: float,
) -> list[dict[str, Any]]:
    """Maximal Marginal Relevance.

    Greedily picks the candidate maximising
    ``lambda * relevance - (1 - lambda) * max_similarity_to_already_picked``.
    This deliberately *trades away* some relevance to avoid near-duplicates:
    lambda=1.0 is pure relevance (identical to the incoming vector ranking),
    lambda=0.0 is pure diversity.

    Each round is a single matrix-vector product against the newly selected
    candidate, with a running maximum — so the similarity of every pair is
    computed once rather than once per round.
    """
    if not candidates:
        return []

    matrix = _embedding_matrix(candidates)
    if matrix is None:
        return _mmr_fallback(candidates, top_k, lambda_)

    unit = _unit_rows(matrix)
    relevance = np.asarray(
        [c.get("score", 0.0) for c in candidates], dtype=np.float32
    )

    count = len(candidates)
    wanted = min(top_k, count)
    available = np.ones(count, dtype=bool)
    # Highest similarity from each candidate to anything already selected.
    best_sim = np.full(count, -np.inf, dtype=np.float32)

    selected: list[dict[str, Any]] = []
    for step in range(wanted):
        if step == 0:
            # Nothing selected yet, so there is no diversity term to subtract.
            objective = relevance.copy()
        else:
            objective = lambda_ * relevance - (1.0 - lambda_) * best_sim
        objective[~available] = -np.inf

        pick = int(np.argmax(objective))
        chosen = candidates[pick]
        # The MMR objective value, not the raw similarity — this is what the
        # ordering is actually based on.
        chosen["rerank_score"] = float(objective[pick])
        selected.append(chosen)

        available[pick] = False
        # One matvec updates every remaining candidate's diversity penalty.
        best_sim = np.maximum(best_sim, unit @ unit[pick])

    return _finalise(selected)


def _mmr_fallback(
    candidates: list[dict[str, Any]], top_k: int, lambda_: float
) -> list[dict[str, Any]]:
    """Degrade to relevance order when vectors are unusable.

    Without embeddings the diversity term is undefined, so the best available
    answer is the ranking Qdrant already produced.
    """
    ordered = sorted(candidates, key=lambda item: item.get("score", 0.0), reverse=True)
    for item in ordered[:top_k]:
        item["rerank_score"] = item.get("score", 0.0)
    return _finalise(ordered[:top_k])


def cosine_rescore(
    candidates: list[dict[str, Any]],
    query_embedding: list[float],
    top_k: int,
) -> list[dict[str, Any]]:
    """Recompute exact cosine similarity against the query and re-sort.

    Qdrant already returns points ordered by cosine distance, so this usually
    reproduces the same ranking. It remains useful as an exact rescoring step:
    Qdrant's HNSW search is *approximate*, and this computes the true cosine
    over the returned vectors, which can reorder near-ties the ANN traversal got
    slightly wrong.
    """
    if not candidates:
        return []

    matrix = _embedding_matrix(candidates)
    if matrix is None:
        # No usable vectors: keep each candidate's stored score.
        for candidate in candidates:
            candidate["rerank_score"] = candidate.get("score", 0.0)
        candidates.sort(key=lambda item: item["rerank_score"], reverse=True)
        return _finalise(candidates[:top_k])

    query = np.asarray(query_embedding, dtype=np.float32)
    query_norm = float(np.linalg.norm(query))
    if query_norm == 0.0 or query.shape[0] != matrix.shape[1]:
        for candidate in candidates:
            candidate["rerank_score"] = candidate.get("score", 0.0)
        candidates.sort(key=lambda item: item["rerank_score"], reverse=True)
        return _finalise(candidates[:top_k])

    scores = _unit_rows(matrix) @ (query / query_norm)
    for candidate, score in zip(candidates, scores, strict=True):
        candidate["rerank_score"] = float(score)

    candidates.sort(key=lambda item: item["rerank_score"], reverse=True)
    return _finalise(candidates[:top_k])
