"""Cross-encoder reranking.

A bi-encoder (what Qdrant stores) compresses the query and each passage into a
vector *independently*, then compares them once with cosine similarity. A
cross-encoder instead feeds the pair ``(query_text, passage_text)`` through a
transformer together, so every query token can attend to every passage token.
That is a genuine relevance judgement rather than a geometric proxy, and it
reliably outranks cosine similarity — at the cost of one forward pass per
candidate, which is why it only ever runs over the shortlist that the vector
search already narrowed down.

The model is optional. If ``sentence-transformers`` / ``torch`` are not
installed the rest of the service is unaffected; only ``method="cross_encoder"``
fails, with a 503 explaining what to install.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from fastapi import HTTPException, status

from ..core.settings import settings

logger = logging.getLogger(__name__)

# Model name -> loaded CrossEncoder. Populated lazily; a model is downloaded and
# loaded at most once per process.
_models: dict[str, Any] = {}
# Loading is not thread-safe and takes seconds; serialise it so two concurrent
# first requests don't both pull the weights.
_load_lock = threading.Lock()


class CrossEncoderUnavailable(HTTPException):
    def __init__(self, detail: str) -> None:
        super().__init__(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)


def is_available() -> bool:
    """True when the optional reranking dependencies are importable."""
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_model(requested: str | None) -> str:
    """Validate a requested model against the allowlist, or return the default."""
    if requested is None:
        return settings.rerank_model
    requested = requested.strip()
    if requested not in settings.rerank_allowed_models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Reranker model '{requested}' is not allowed. "
                f"Configured models: {', '.join(settings.rerank_allowed_models)}."
            ),
        )
    return requested


def _load(model_name: str) -> Any:
    """Load (and cache) a CrossEncoder. Blocking — call from a worker thread."""
    cached = _models.get(model_name)
    if cached is not None:
        return cached

    with _load_lock:
        # Re-check inside the lock: another thread may have loaded it while we waited.
        cached = _models.get(model_name)
        if cached is not None:
            return cached

        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise CrossEncoderUnavailable(
                "Cross-encoder reranking is not available in this deployment: "
                "the optional 'sentence-transformers' dependency is not installed. "
                "Install the service with the [rerank] extra, or use "
                "method='mmr' / 'cosine_rescore' instead."
            ) from exc

        if settings.rerank_threads > 0:
            torch.set_num_threads(settings.rerank_threads)

        logger.info("Loading cross-encoder '%s' (CPU) ...", model_name)
        model = CrossEncoder(
            model_name, max_length=settings.rerank_max_length, device="cpu"
        )
        _models[model_name] = model
        logger.info("Cross-encoder '%s' ready", model_name)
        return model


def _candidate_text(candidate: dict[str, Any]) -> str:
    """The text a candidate is scored on.

    Prefers the ``text`` field, falling back to the joined ``chunks`` — which is
    what clients actually populate today.
    """
    text = candidate.get("text")
    if text:
        return str(text)
    chunks = candidate.get("chunks") or []
    return " ".join(str(chunk) for chunk in chunks)


def _score_sync(
    model_name: str, query_text: str, candidates: list[dict[str, Any]]
) -> list[float]:
    model = _load(model_name)
    pairs = [(query_text, _candidate_text(candidate)) for candidate in candidates]
    scores = model.predict(pairs, show_progress_bar=False)
    return [float(score) for score in scores]


async def cross_encoder_rerank(
    candidates: list[dict[str, Any]],
    query_text: str,
    top_k: int,
    model_name: str,
) -> list[dict[str, Any]]:
    """Re-order candidates by cross-encoder relevance to ``query_text``.

    Inference is CPU-bound and takes hundreds of milliseconds to seconds, so it
    runs in a worker thread — otherwise it would block the event loop and stall
    every other request on the pod for its duration.
    """
    if not candidates:
        return []

    if len(candidates) > settings.rerank_max_candidates:
        logger.warning(
            "Truncating %d candidates to rerank_max_candidates=%d before cross-encoding",
            len(candidates),
            settings.rerank_max_candidates,
        )
        # The candidates are already sorted by vector score, so the tail we drop
        # is the least similar — the part a reranker is least likely to promote.
        candidates = candidates[: settings.rerank_max_candidates]

    scores = await asyncio.to_thread(_score_sync, model_name, query_text, candidates)

    for candidate, score in zip(candidates, scores, strict=True):
        candidate["rerank_score"] = score
        candidate.pop("embedding", None)

    candidates.sort(key=lambda item: item["rerank_score"], reverse=True)
    return candidates[:top_k]


async def preload() -> None:
    """Warm the default model at startup when RERANK_PRELOAD=true."""
    if not settings.rerank_preload:
        return
    if not is_available():
        logger.warning(
            "RERANK_PRELOAD is set but sentence-transformers is not installed; skipping"
        )
        return
    await asyncio.to_thread(_load, settings.rerank_model)
