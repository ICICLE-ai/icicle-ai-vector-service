"""``/v1/retrieve`` and ``/v1/rerank`` — vector search and result reranking.

Two stages, deliberately separate:

1. **Retrieve** — Qdrant's HNSW index finds the ``top_k`` nearest vectors under
   cosine distance, with ``user_id``/topic/metadata filters applied *during*
   traversal.
2. **Rerank** — fetch a wider shortlist (``fetch_k``) and reorder it. Vector
   methods ('mmr', 'cosine_rescore') work on embeddings alone. The
   'cross_encoder' method reads the actual passage text alongside the raw query
   and is the only method that judges relevance rather than geometry.
"""

import logging

from fastapi import APIRouter, Depends
from qdrant_client import AsyncQdrantClient

from ...core import UserContext, get_current_user
from ...core.settings import settings
from ...db import get_qdrant_client
from ...db.naming import Owner
from ...db.repository import fetch_candidates, retrieve_embeddings
from ...reranking import is_available, rerank as run_rerank, resolve_model
from ...schemas import (
    RerankedItem,
    RerankMethodInfo,
    RerankMethodList,
    RerankRequest,
    RerankResponse,
    ResultItem,
    RetrieveRequest,
    RetrieveResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["search"])


@router.post("/retrieve", response_model=RetrieveResponse, summary="Vector similarity search")
async def retrieve(
    payload: RetrieveRequest,
    client: AsyncQdrantClient = Depends(get_qdrant_client),
    current_user: UserContext = Depends(get_current_user),
) -> RetrieveResponse:
    logger.info(
        "Retrieving top-%d for user '%s' (collection: %s, topic: %s, filter: %s)",
        payload.top_k,
        current_user.username,
        payload.collection,
        payload.topic,
        payload.filter.conditions if payload.filter else None,
    )
    results = await retrieve_embeddings(
        client,
        Owner.from_context(current_user),
        payload.query_embedding,
        payload.top_k,
        collection=payload.collection,
        topic=payload.topic,
        metadata_filter=payload.filter,
    )
    logger.info("Returned %d results for user '%s'", len(results), current_user.username)
    return RetrieveResponse(
        user_id=current_user.username, top_k=payload.top_k, results=results
    )


@router.get(
    "/rerank/methods",
    response_model=RerankMethodList,
    summary="List available rerank methods",
)
async def rerank_methods() -> RerankMethodList:
    """What this deployment can actually do.

    ``available`` is false for 'cross_encoder' when the optional model
    dependencies were not installed in the image.
    """
    ce_available = is_available()
    return RerankMethodList(
        methods=[
            RerankMethodInfo(
                name="mmr",
                kind="vector",
                available=True,
                requires_query_text=False,
                description="Maximal Marginal Relevance. Trades some relevance for "
                "diversity so near-duplicate passages don't fill the results. "
                "'lambda' controls the balance (1.0 = pure relevance).",
            ),
            RerankMethodInfo(
                name="cosine_rescore",
                kind="vector",
                available=True,
                requires_query_text=False,
                description="Recomputes exact cosine similarity over the fetched "
                "vectors and re-sorts. Corrects near-ties that Qdrant's "
                "approximate HNSW search ordered slightly wrong.",
            ),
            RerankMethodInfo(
                name="cross_encoder",
                kind="cross-encoder",
                available=ce_available,
                requires_query_text=True,
                description="Scores the raw query against each passage's text with a "
                "transformer that reads both together. The only method that judges "
                "true relevance rather than vector geometry; also the slowest.",
            ),
        ],
        default_model=settings.rerank_model,
        allowed_models=settings.rerank_allowed_models,
    )


@router.post("/rerank", response_model=RerankResponse, summary="Rerank search results")
async def rerank(
    payload: RerankRequest,
    client: AsyncQdrantClient = Depends(get_qdrant_client),
    current_user: UserContext = Depends(get_current_user),
) -> RerankResponse:
    # Resolve the model before touching Qdrant so a bad model name fails fast
    # rather than after a search has already run.
    model_name = (
        resolve_model(payload.rerank_model) if payload.method == "cross_encoder" else None
    )

    logger.info(
        "Reranking for user '%s' (collection: %s, topic: %s, method: %s, model: %s, "
        "fetch_k: %d, top_k: %d)",
        current_user.username,
        payload.collection,
        payload.topic,
        payload.method,
        model_name,
        payload.fetch_k,
        payload.top_k,
    )
    candidates = await fetch_candidates(
        client,
        Owner.from_context(current_user),
        payload.query_embedding,
        payload.fetch_k,
        collection=payload.collection,
        topic=payload.topic,
        metadata_filter=payload.filter,
    )

    reranked = await run_rerank(
        method=payload.method,
        candidates=candidates,
        query_embedding=payload.query_embedding,
        top_k=payload.top_k,
        lambda_=payload.lambda_,
        query_text=payload.query_text,
        model_name=model_name,
    )

    results = [RerankedItem(**item) for item in reranked]
    logger.info(
        "Reranked %d -> %d results for user '%s' via %s",
        len(candidates),
        len(results),
        current_user.username,
        payload.method,
    )
    return RerankResponse(
        user_id=current_user.username,
        method=payload.method,
        model=model_name,
        top_k=payload.top_k,
        fetch_k=payload.fetch_k,
        results=results,
    )
