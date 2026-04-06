import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import AsyncQdrantClient

from .auth import UserContext, get_current_user
from .crud import (
    create_embedding,
    delete_embedding,
    fetch_candidates,
    retrieve_embeddings,
    update_embedding,
)
from .db import get_qdrant_client, _client
from .rerank import mmr_rerank
from .schemas import (
    DeleteResponse,
    EmbeddingCreate,
    EmbeddingRecord,
    EmbeddingUpdate,
    RerankRequest,
    RerankResponse,
    RetrieveRequest,
    RetrieveResponse,
    ResultItem,

)
from .settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Connecting to Qdrant at %s ...", settings.qdrant_url)
    try:
        client = _client()
        collections = await client.get_collections()
        logger.info(
            "Qdrant is reachable (%d collections found)",
            len(collections.collections or []),
        )
    except Exception as exc:
        logger.error("Failed to connect to Qdrant at %s: %s", settings.qdrant_url, exc)
        raise SystemExit(
            f"Qdrant is not reachable at {settings.qdrant_url}. "
            "Check QDRANT_URL and QDRANT_API_KEY in your .env file."
        ) from exc
    yield


app = FastAPI(
    title="ICICLE AI Vector Service",
    version="0.5.0",
    description="Vector storage and retrieval service for the ICICLE AI tenant. "
    "Clients provide their own pre-computed embeddings. "
    "Each broad domain is a Qdrant collection (e.g. 'biology', 'chemistry'). "
    "Topics are optional sub-categories within a collection (e.g. 'human', 'plant').",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["X-Tapis-Token", "Content-Type", "Authorization"],
)


@app.get("/healthz")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/embeddings", response_model=EmbeddingRecord, status_code=201)
async def store_embedding(
    payload: EmbeddingCreate,
    client: AsyncQdrantClient = Depends(get_qdrant_client),
    current_user: UserContext = Depends(get_current_user),
) -> EmbeddingRecord:
    data = payload.model_dump(by_alias=True)
    data["user_id"] = current_user.username
    logger.info(
        "Creating embedding for user '%s' (collection: %s, topic: %s, model: %s, dims: %d)",
        current_user.username,
        payload.collection,
        payload.topic,
        payload.embedding_model,
        len(payload.embedding),
    )
    record = await create_embedding(client, data)
    logger.info("Created embedding %s for user '%s'", record["id"], current_user.username)
    return EmbeddingRecord(**record)


@app.put("/v1/embeddings/{embedding_id}", response_model=EmbeddingRecord)
async def update_user_embedding(
    embedding_id: str,
    payload: EmbeddingUpdate,
    collection: str = Query(..., description="Collection the embedding belongs to"),
    client: AsyncQdrantClient = Depends(get_qdrant_client),
    current_user: UserContext = Depends(get_current_user),
) -> EmbeddingRecord:
    updates = payload.model_dump(by_alias=True, exclude_none=True)
    logger.info(
        "Updating embedding %s for user '%s' in collection '%s' (fields: %s)",
        embedding_id,
        current_user.username,
        collection,
        list(updates.keys()),
    )
    record = await update_embedding(
        client, current_user.username, collection, embedding_id, updates
    )
    logger.info("Updated embedding %s for user '%s'", embedding_id, current_user.username)
    return EmbeddingRecord(**record)


@app.delete("/v1/embeddings/{embedding_id}", response_model=DeleteResponse)
async def delete_user_embedding(
    embedding_id: str,
    collection: str = Query(..., description="Collection the embedding belongs to"),
    client: AsyncQdrantClient = Depends(get_qdrant_client),
    current_user: UserContext = Depends(get_current_user),
) -> DeleteResponse:
    logger.info(
        "Deleting embedding %s for user '%s' from collection '%s'",
        embedding_id,
        current_user.username,
        collection,
    )
    deleted = await delete_embedding(client, current_user.username, collection, embedding_id)
    if not deleted:
        logger.warning(
            "Embedding %s not found for user '%s' in collection '%s'",
            embedding_id,
            current_user.username,
            collection,
        )
        raise HTTPException(
            status_code=404,
            detail=f"Embedding '{embedding_id}' not found in collection '{collection}'.",
        )
    logger.info("Deleted embedding %s for user '%s'", embedding_id, current_user.username)
    return DeleteResponse(id=embedding_id, user_id=current_user.username, deleted=True)


@app.post("/v1/retrieve", response_model=RetrieveResponse)
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
        client, current_user.username, payload.query_embedding, payload.top_k,
        collection=payload.collection, topic=payload.topic, metadata_filter=payload.filter,
    )
    logger.info("Returned %d results for user '%s'", len(results), current_user.username)
    return RetrieveResponse(
        user_id=current_user.username, top_k=payload.top_k, results=results
    )



@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank(
    payload: RerankRequest,
    client: AsyncQdrantClient = Depends(get_qdrant_client),
    current_user: UserContext = Depends(get_current_user),
) -> RerankResponse:
    if payload.method not in {"mmr", "cosine_rescore"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported rerank method '{payload.method}'. Use 'mmr' or 'cosine_rescore'.",
        )

    logger.info(
        "Reranking for user '%s' (collection: %s, topic: %s, method: %s, fetch_k: %d, top_k: %d)",
        current_user.username,
        payload.collection,
        payload.topic,
        payload.method,
        payload.fetch_k,
        payload.top_k,
    )
    candidates = await fetch_candidates(
        client, current_user.username, payload.query_embedding, payload.fetch_k,
        collection=payload.collection, topic=payload.topic, metadata_filter=payload.filter,
    )

    if payload.method == "mmr":
        reranked = mmr_rerank(candidates, payload.query_embedding, payload.top_k, payload.lambda_)
    else:
        reranked = candidates[: payload.top_k]
        for item in reranked:
            item.pop("embedding", None)

    results = [ResultItem(**item) for item in reranked]
    logger.info(
        "Reranked %d -> %d results for user '%s'",
        len(candidates),
        len(results),
        current_user.username,
    )
    return RerankResponse(
        user_id=current_user.username,
        method=payload.method,
        top_k=payload.top_k,
        fetch_k=payload.fetch_k,
        results=results,
    )
