"""Result reranking.

Two families, deliberately kept apart:

* :mod:`vector` — ``mmr`` and ``cosine_rescore``. Pure embedding arithmetic over
  vectors already in Qdrant. Microseconds, no model, no dependencies.
* :mod:`cross_encoder` — ``cross_encoder``. Runs a transformer over
  ``(query_text, passage_text)`` pairs. The only method that judges relevance
  rather than vector geometry, and the only one that can be unavailable.

:func:`rerank` is the single entry point; route handlers do not branch on the
method themselves.
"""

from __future__ import annotations

from typing import Any

from ..schemas.search import RerankMethod
from .cross_encoder import (
    CrossEncoderUnavailable,
    cross_encoder_rerank,
    is_available,
    preload,
    resolve_model,
)
from .vector import cosine_rescore, cosine_sim, mmr_rerank

__all__ = [
    "CrossEncoderUnavailable",
    "cosine_rescore",
    "cosine_sim",
    "cross_encoder_rerank",
    "is_available",
    "mmr_rerank",
    "preload",
    "rerank",
    "resolve_model",
]


async def rerank(
    method: RerankMethod,
    candidates: list[dict[str, Any]],
    query_embedding: list[float],
    top_k: int,
    lambda_: float = 0.7,
    query_text: str | None = None,
    model_name: str | None = None,
) -> list[dict[str, Any]]:
    """Dispatch to the requested method.

    ``query_text`` and ``model_name`` are required for ``cross_encoder`` and are
    validated upstream (by the request schema and :func:`resolve_model`), so by
    the time we get here they are guaranteed present for that method.
    """
    # Run inline for now. Vectorised this is ~1.4ms at the default fetch_k=50;
    # whether it needs a worker thread is a question for the load test.
    if method == "mmr":
        return mmr_rerank(candidates, query_embedding, top_k, lambda_)
    if method == "cosine_rescore":
        return cosine_rescore(candidates, query_embedding, top_k)
    if method == "cross_encoder":
        assert query_text is not None and model_name is not None
        return await cross_encoder_rerank(candidates, query_text, top_k, model_name)
    raise ValueError(f"Unknown rerank method: {method!r}")
