"""``/v1/embeddings`` — create, read, update and delete individual embeddings.

Every handler derives ``user_id`` from the verified Tapis token; it is never
accepted from the client. The CRUD layer then scopes each Qdrant operation to
that user, so a caller cannot read or mutate a point they do not own.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from qdrant_client import AsyncQdrantClient

from ...core import UserContext, get_current_user
from ...db import get_qdrant_client
from ...db.naming import Owner
from ...db.repository import (
    bulk_delete_embeddings,
    create_embedding,
    delete_embedding,
    get_embedding,
    update_embedding,
)
from ...schemas import (
    BulkDeleteRequest,
    BulkDeleteResponse,
    DeleteResponse,
    EmbeddingCreate,
    EmbeddingRecord,
    EmbeddingUpdate,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


@router.post(
    "",
    response_model=EmbeddingRecord,
    status_code=status.HTTP_201_CREATED,
    summary="Store an embedding",
)
async def store_embedding(
    payload: EmbeddingCreate,
    client: AsyncQdrantClient = Depends(get_qdrant_client),
    current_user: UserContext = Depends(get_current_user),
) -> EmbeddingRecord:
    data = payload.model_dump(by_alias=True)
    logger.info(
        "Creating embedding for user '%s' (collection: %s, topic: %s, model: %s, dims: %d)",
        current_user.username,
        payload.collection,
        payload.topic,
        payload.embedding_model,
        len(payload.embedding),
    )
    record = await create_embedding(client, Owner.from_context(current_user), data)
    logger.info("Created embedding %s for user '%s'", record["id"], current_user.username)
    return EmbeddingRecord(**record)


@router.post(
    "/bulk-delete",
    response_model=BulkDeleteResponse,
    summary="Delete many embeddings in one collection",
)
async def bulk_delete(
    payload: BulkDeleteRequest,
    client: AsyncQdrantClient = Depends(get_qdrant_client),
    current_user: UserContext = Depends(get_current_user),
) -> BulkDeleteResponse:
    """Delete by explicit ids, by topic/metadata predicate, or everything in the collection.

    Uses POST rather than DELETE because the selector is a request body, which
    DELETE cannot carry portably. Only the caller's own points are ever removed.
    """
    logger.info(
        "Bulk delete for user '%s' (collection: %s, ids: %s, topic: %s, all: %s)",
        current_user.username,
        payload.collection,
        len(payload.ids) if payload.ids else 0,
        payload.topic,
        payload.all,
    )
    deleted, dropped = await bulk_delete_embeddings(
        client,
        Owner.from_context(current_user),
        payload.collection,
        ids=payload.ids,
        topic=payload.topic,
        metadata_filter=payload.filter,
        delete_all=payload.all,
    )
    logger.info(
        "Bulk deleted %d embeddings for user '%s' from '%s' (collection dropped: %s)",
        deleted,
        current_user.username,
        payload.collection,
        dropped,
    )
    return BulkDeleteResponse(
        user_id=current_user.username,
        collection=payload.collection,
        deleted=deleted,
        collection_dropped=dropped,
    )


@router.get(
    "/{embedding_id}", response_model=EmbeddingRecord, summary="Get one embedding"
)
async def read_embedding(
    embedding_id: str,
    collection: str = Query(..., description="Collection the embedding belongs to"),
    client: AsyncQdrantClient = Depends(get_qdrant_client),
    current_user: UserContext = Depends(get_current_user),
) -> EmbeddingRecord:
    record = await get_embedding(
        client, Owner.from_context(current_user), collection, embedding_id
    )
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Embedding '{embedding_id}' not found in collection '{collection}'.",
        )
    return EmbeddingRecord(**record)


@router.put(
    "/{embedding_id}", response_model=EmbeddingRecord, summary="Update an embedding"
)
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
        client, Owner.from_context(current_user), collection, embedding_id, updates
    )
    if record is None:
        # Either the embedding does not exist or it belongs to someone else.
        # Both answer 404: distinguishing them would leak that an id is in use.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Embedding '{embedding_id}' not found in collection '{collection}'.",
        )
    logger.info("Updated embedding %s for user '%s'", embedding_id, current_user.username)
    return EmbeddingRecord(**record)


@router.delete(
    "/{embedding_id}", response_model=DeleteResponse, summary="Delete an embedding"
)
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
    deleted = await delete_embedding(
        client, Owner.from_context(current_user), collection, embedding_id
    )
    if not deleted:
        logger.warning(
            "Embedding %s not found for user '%s' in collection '%s'",
            embedding_id,
            current_user.username,
            collection,
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Embedding '{embedding_id}' not found in collection '{collection}'.",
        )
    logger.info("Deleted embedding %s for user '%s'", embedding_id, current_user.username)
    return DeleteResponse(id=embedding_id, user_id=current_user.username, deleted=True)
