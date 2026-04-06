from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)

from .schemas import MetadataFilter
from .settings import settings

Payload = dict[str, Any]

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(value: str) -> str:
    """Lowercase, replace non-alphanumeric runs with '_', strip edges."""
    return _SLUG_RE.sub("_", value.lower()).strip("_")


def _collection_name(collection: str) -> str:
    """Slugify the user-facing collection name for use as a Qdrant collection."""
    return _slugify(collection)


async def _collection_exists(client: AsyncQdrantClient, collection_name: str) -> bool:
    collections = await client.get_collections()
    return any(col.name == collection_name for col in collections.collections or [])


async def ensure_collection(client: AsyncQdrantClient, collection: str) -> str:
    collection_name = _collection_name(collection)
    if not await _collection_exists(client, collection_name):
        await client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=settings.vector_dim, distance=Distance.COSINE),
        )
    return collection_name


def _base_payload(
    user_id: str,
    collection: str,
    topic: str | None,
    token_ids: list[int],
    text: str | None,
    chunks: list[str],
    metadata: dict[str, Any],
    embedding_model: str | None,
    created_at: str,
    updated_at: str,
) -> Payload:
    return {
        "user_id": user_id,
        "collection": collection,
        "topic": topic,
        "token_ids": token_ids,
        "text": text,
        "chunks": chunks,
        "metadata": metadata,
        "embedding_model": embedding_model,
        "created_at": created_at,
        "updated_at": updated_at,
    }


async def create_embedding(
    client: AsyncQdrantClient, payload: dict[str, Any]
) -> dict[str, Any]:
    user_id = payload["user_id"]
    collection = payload["collection"]
    topic = payload.get("topic")
    collection_name = await ensure_collection(client, collection)

    now = datetime.now(timezone.utc).isoformat()
    point_id = str(uuid.uuid4())
    point_payload = _base_payload(
        user_id=user_id,
        collection=collection,
        topic=topic,
        token_ids=payload.get("token_ids", []),
        text=payload.get("text"),
        chunks=payload.get("chunks", []),
        metadata=payload.get("metadata", {}),
        embedding_model=payload.get("embedding_model"),
        created_at=now,
        updated_at=now,
    )
    point = PointStruct(id=point_id, vector=payload["embedding"], payload=point_payload)
    await client.upsert(collection_name=collection_name, points=[point], wait=True)
    return {
        "id": point_id,
        "user_id": user_id,
        "collection": collection,
        "topic": topic,
        "created_at": now,
        "updated_at": now,
        "embedding_model": payload.get("embedding_model"),
    }


async def update_embedding(
    client: AsyncQdrantClient,
    user_id: str,
    collection: str,
    embedding_id: str,
    updates: dict[str, Any],
) -> dict[str, Any]:
    collection_name = _collection_name(collection)
    if not await _collection_exists(client, collection_name):
        raise HTTPException(status_code=404, detail="Embedding not found")

    existing = await client.retrieve(
        collection_name=collection_name, ids=[embedding_id], with_vectors=True
    )
    if not existing or (existing[0].payload or {}).get("user_id") != user_id:
        raise HTTPException(status_code=404, detail="Embedding not found")

    current = existing[0]
    current_payload: Payload = current.payload or {}
    created_at = current_payload.get("created_at") or datetime.now(timezone.utc).isoformat()

    topic = updates.get("topic", current_payload.get("topic"))
    vector = updates.get("embedding") or current.vector
    metadata = updates.get("metadata", current_payload.get("metadata", {}))
    token_ids = updates.get("token_ids", current_payload.get("token_ids", []))
    text = updates.get("text", current_payload.get("text"))
    chunks = updates.get("chunks", current_payload.get("chunks", []))
    embedding_model = updates.get("embedding_model") or current_payload.get("embedding_model")
    updated_at = datetime.now(timezone.utc).isoformat()

    new_payload = _base_payload(
        user_id=user_id,
        collection=collection,
        topic=topic,
        token_ids=token_ids,
        text=text,
        chunks=chunks,
        metadata=metadata,
        embedding_model=embedding_model,
        created_at=created_at,
        updated_at=updated_at,
    )
    point = PointStruct(id=embedding_id, vector=vector, payload=new_payload)
    await client.upsert(collection_name=collection_name, points=[point], wait=True)
    return {
        "id": embedding_id,
        "user_id": user_id,
        "collection": collection,
        "topic": topic,
        "created_at": created_at,
        "updated_at": updated_at,
        "embedding_model": embedding_model,
    }


async def delete_embedding(
    client: AsyncQdrantClient, user_id: str, collection: str, embedding_id: str
) -> bool:
    collection_name = _collection_name(collection)
    if not await _collection_exists(client, collection_name):
        return False

    existing = await client.retrieve(collection_name=collection_name, ids=[embedding_id])
    if not existing or (existing[0].payload or {}).get("user_id") != user_id:
        return False

    await client.delete(collection_name=collection_name, points_selector=[embedding_id], wait=True)
    return True


def _build_qdrant_filter(
    user_id: str,
    topic: str | None = None,
    metadata_filter: MetadataFilter | None = None,
) -> Filter:
    """Build a Qdrant filter scoped to the requesting user.

    Always filters by user_id for data isolation. Optionally narrows
    results to a specific topic within the collection.
    """
    must: list[FieldCondition] = [
        FieldCondition(key="user_id", match=MatchValue(value=user_id)),
    ]
    if topic:
        must.append(FieldCondition(key="topic", match=MatchValue(value=topic)))
    if metadata_filter and metadata_filter.conditions:
        for key, value in metadata_filter.conditions.items():
            field_path = f"metadata.{key}"
            if isinstance(value, list):
                must.append(FieldCondition(key=field_path, match=MatchAny(any=value)))
            else:
                must.append(FieldCondition(key=field_path, match=MatchValue(value=value)))
    return Filter(must=must)


async def retrieve_embeddings(
    client: AsyncQdrantClient,
    user_id: str,
    query_embedding: list[float],
    top_k: int,
    collection: str,
    topic: str | None = None,
    metadata_filter: MetadataFilter | None = None,
) -> list[dict[str, Any]]:
    collection_name = await ensure_collection(client, collection)
    response = await client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        query_filter=_build_qdrant_filter(user_id, topic, metadata_filter),
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )
    return [
        {
            "id": str(item.id),
            "score": float(item.score),
            "collection": collection,
            "topic": item.payload.get("topic") if item.payload else None,
            "text": item.payload.get("text") if item.payload else None,
            "chunks": item.payload.get("chunks", []) if item.payload else [],
            "metadata": item.payload.get("metadata", {}) if item.payload else {},
        }
        for item in response.points
    ]


async def fetch_candidates(
    client: AsyncQdrantClient,
    user_id: str,
    query_embedding: list[float],
    fetch_k: int,
    collection: str,
    topic: str | None = None,
    metadata_filter: MetadataFilter | None = None,
) -> list[dict[str, Any]]:
    collection_name = await ensure_collection(client, collection)
    response = await client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        query_filter=_build_qdrant_filter(user_id, topic, metadata_filter),
        limit=fetch_k,
        with_payload=True,
        with_vectors=True,
    )
    return [
        {
            "id": str(item.id),
            "score": float(item.score),
            "collection": collection,
            "topic": item.payload.get("topic") if item.payload else None,
            "text": item.payload.get("text") if item.payload else None,
            "chunks": item.payload.get("chunks", []) if item.payload else [],
            "metadata": item.payload.get("metadata", {}) if item.payload else {},
            "embedding": item.vector if item.vector is not None else [],
        }
        for item in response.points
    ]


