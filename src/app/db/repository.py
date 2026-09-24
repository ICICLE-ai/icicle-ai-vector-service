"""Every Qdrant read and write.

Two invariants hold throughout, enforced by ``tests/v1/test_isolation.py``:

1. Every operation is scoped. Reads, counts, scrolls and deletes go through
   :mod:`app.db.filters`. Direct point lookups cannot be filtered server-side, so
   they verify ownership on the returned payload before acting on it.
2. Nothing a user does destroys another user's data. Each user's collections are
   physically their own (see :mod:`app.db.naming`).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException, status
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    Distance,
    FilterSelector,
    KeywordIndexParams,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from ..core.settings import settings
from ..schemas.common import MetadataFilter
from . import filters
from .naming import Owner, display_name, owns, physical_name, slugify

logger = logging.getLogger(__name__)

Payload = dict[str, Any]

# Upper bound on payloads paged through when deriving distinct topics/models,
# if the server is too old for the facet API. Keeps listing bounded.
# Cap on simultaneous per-collection lookups when listing.
_LIST_CONCURRENCY = 16

# Distinct values returned per field; beyond this the list is reported truncated.
_FACET_LIMIT = 100

_FACET_SCROLL_CAP = 10_000
_SCROLL_PAGE = 512


def collection_name(owner: Owner, collection: str) -> str:
    """The physical Qdrant collection holding ``owner``'s ``collection``.

    Every storage call goes through this, so a request can only ever address a
    collection inside the caller's own namespace.
    """
    name = physical_name(owner, collection)
    if name is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Collection name '{collection}' is not valid: it must contain "
                "at least one letter or digit."
            ),
        )
    return name


async def owned_collections(client: AsyncQdrantClient, owner: Owner) -> list[str]:
    """Physical names of every collection belonging to ``owner``.

    A prefix scan over the cluster's collection list — no per-collection query,
    and collections belonging to anyone else are never even considered.
    """
    collections = await client.get_collections()
    return sorted(
        col.name
        for col in (collections.collections or [])
        if col.name.startswith(owner.prefix)
    )


async def collection_exists(client: AsyncQdrantClient, name: str) -> bool:
    collections = await client.get_collections()
    return any(col.name == name for col in collections.collections or [])


async def _create_payload_indexes(client: AsyncQdrantClient, name: str) -> None:
    """Create the keyword indexes the tenancy filters rely on.

    A collection holds exactly one user's points, so ``is_tenant`` would be
    meaningless here — the index exists to serve the topic/metadata filters and
    the defence-in-depth ``user_id`` condition.

    Best-effort — an index that already exists, or a server that rejects the
    request, must not break the write that triggered this.
    """
    index_specs: list[tuple[str, Any]] = [
        (filters.USER_ID_FIELD, PayloadSchemaType.KEYWORD),
        (filters.TOPIC_FIELD, PayloadSchemaType.KEYWORD),
        ("embedding_model", PayloadSchemaType.KEYWORD),
    ]
    for field, schema in index_specs:
        try:
            await client.create_payload_index(
                collection_name=name, field_name=field, field_schema=schema, wait=True
            )
        except Exception as exc:  # pragma: no cover - depends on server state
            logger.debug("Payload index '%s' on '%s' not created: %s", field, name, exc)


async def ensure_collection(
    client: AsyncQdrantClient,
    owner: Owner,
    collection: str,
    vector_dim: int | None = None,
) -> str:
    """Ensure the caller's collection exists, returning its physical name.

    When creating, ``vector_dim`` is required (taken from the first embedding
    stored). For reads the collection must already exist.
    """
    name = collection_name(owner, collection)
    if not await collection_exists(client, name):
        if vector_dim is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection '{collection}' not found.",
            )
        await client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )
        await _create_payload_indexes(client, name)
    return name


async def _assert_dimension_matches(
    client: AsyncQdrantClient, name: str, collection: str, vector_dim: int
) -> None:
    """Fail with an explanatory error when the vector width is wrong.

    A collection's dimension is fixed by its first embedding. Collections are
    per user, so this only ever reflects the caller's own earlier choice — but
    Qdrant's native error is opaque, so name the actual cause.
    """
    info = await client.get_collection(collection_name=name)
    existing = getattr(info.config.params.vectors, "size", None)
    if existing is not None and existing != vector_dim:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Vector dimension mismatch: collection '{collection}' stores "
                f"{existing}-dimensional vectors, but the supplied embedding has "
                f"{vector_dim}. A collection's dimension is fixed by its first "
                "embedding and cannot be changed."
            ),
        )


def _base_payload(
    user_id: str,
    collection: str,
    topic: str | None,
    token_ids: list[int],
    text: str | None,
    chunks: list[str],
    metadata: dict[str, Any],
    embedding_model: str,
    vector_dim: int,
    created_at: str,
    updated_at: str,
) -> Payload:
    return {
        filters.USER_ID_FIELD: user_id,
        "collection": collection,
        filters.TOPIC_FIELD: topic,
        "token_ids": token_ids,
        "text": text,
        "chunks": chunks,
        "metadata": metadata,
        "embedding_model": embedding_model,
        "vector_dim": vector_dim,
        "created_at": created_at,
        "updated_at": updated_at,
    }


def _record(point_id: str, payload: Payload) -> dict[str, Any]:
    """Project a stored payload onto the EmbeddingRecord shape."""
    return {
        "id": point_id,
        "user_id": payload.get(filters.USER_ID_FIELD, ""),
        "collection": payload.get("collection", ""),
        "topic": payload.get(filters.TOPIC_FIELD),
        "vector_dim": payload.get("vector_dim", 0),
        "created_at": payload.get("created_at", ""),
        "updated_at": payload.get("updated_at", ""),
        "embedding_model": payload.get("embedding_model", ""),
    }


def _hit(item: Any, collection: str, with_vector: bool = False) -> dict[str, Any]:
    """Project a search hit onto the ResultItem shape."""
    payload = item.payload or {}
    result = {
        "id": str(item.id),
        "score": float(item.score),
        "collection": collection,
        "topic": payload.get(filters.TOPIC_FIELD),
        "text": payload.get("text"),
        "chunks": payload.get("chunks", []),
        "metadata": payload.get("metadata", {}),
    }
    if with_vector:
        result["embedding"] = item.vector if item.vector is not None else []
    return result


async def _owned_point(
    client: AsyncQdrantClient,
    user_id: str,
    name: str,
    embedding_id: str,
    with_vectors: bool = False,
) -> Any | None:
    """Fetch a point only if ``user_id`` owns it, else None.

    ``retrieve`` takes ids, not filters, so ownership is checked on the returned
    payload. Returning None (rather than the point) for a foreign id is what
    makes the API answer 404 instead of leaking its existence.
    """
    found = await client.retrieve(
        collection_name=name, ids=[embedding_id], with_vectors=with_vectors
    )
    if not found:
        return None
    point = found[0]
    if (point.payload or {}).get(filters.USER_ID_FIELD) != user_id:
        logger.warning(
            "User '%s' attempted to access point %s owned by another user",
            user_id,
            embedding_id,
        )
        return None
    return point


# --- Writes --------------------------------------------------------------


async def create_embedding(
    client: AsyncQdrantClient, owner: Owner, payload: dict[str, Any]
) -> dict[str, Any]:
    collection = payload["collection"]
    vector_dim = len(payload["embedding"])
    name = await ensure_collection(client, owner, collection, vector_dim=vector_dim)
    await _assert_dimension_matches(client, name, collection, vector_dim)

    now = datetime.now(timezone.utc).isoformat()
    point_id = str(uuid.uuid4())
    point_payload = _base_payload(
        user_id=owner.username,
        collection=collection,
        topic=payload.get("topic"),
        token_ids=payload.get("token_ids", []),
        text=payload.get("text"),
        chunks=payload.get("chunks", []),
        metadata=payload.get("metadata", {}),
        embedding_model=payload["embedding_model"],
        vector_dim=vector_dim,
        created_at=now,
        updated_at=now,
    )
    await client.upsert(
        collection_name=name,
        points=[PointStruct(id=point_id, vector=payload["embedding"], payload=point_payload)],
        wait=True,
    )
    return _record(point_id, point_payload)


async def get_embedding(
    client: AsyncQdrantClient, owner: Owner, collection: str, embedding_id: str
) -> dict[str, Any] | None:
    name = collection_name(owner, collection)
    if not await collection_exists(client, name):
        return None
    point = await _owned_point(client, owner.username, name, embedding_id)
    return None if point is None else _record(str(point.id), point.payload or {})


async def update_embedding(
    client: AsyncQdrantClient,
    owner: Owner,
    collection: str,
    embedding_id: str,
    updates: dict[str, Any],
) -> dict[str, Any] | None:
    """Partial update. Returns None when the caller does not own the point.

    The rewritten payload always carries the *caller's* user_id, so an update
    can never reassign ownership even if a client sends one.
    """
    name = collection_name(owner, collection)
    if not await collection_exists(client, name):
        return None

    current = await _owned_point(
        client, owner.username, name, embedding_id, with_vectors=True
    )
    if current is None:
        return None

    current_payload: Payload = current.payload or {}
    created_at = current_payload.get("created_at") or datetime.now(timezone.utc).isoformat()
    vector = updates.get("embedding") or current.vector
    vector_dim = (
        len(vector) if isinstance(vector, list) else current_payload.get("vector_dim", 0)
    )
    if isinstance(vector, list) and updates.get("embedding"):
        await _assert_dimension_matches(client, name, collection, vector_dim)

    new_payload = _base_payload(
        user_id=owner.username,
        collection=collection,
        topic=updates.get("topic", current_payload.get(filters.TOPIC_FIELD)),
        token_ids=updates.get("token_ids", current_payload.get("token_ids", [])),
        text=updates.get("text", current_payload.get("text")),
        chunks=updates.get("chunks", current_payload.get("chunks", [])),
        metadata=updates.get("metadata", current_payload.get("metadata", {})),
        embedding_model=updates.get("embedding_model")
        or current_payload.get("embedding_model", ""),
        vector_dim=vector_dim,
        created_at=created_at,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    await client.upsert(
        collection_name=name,
        points=[PointStruct(id=embedding_id, vector=vector, payload=new_payload)],
        wait=True,
    )
    return _record(embedding_id, new_payload)


async def delete_embedding(
    client: AsyncQdrantClient, owner: Owner, collection: str, embedding_id: str
) -> bool:
    name = collection_name(owner, collection)
    if not await collection_exists(client, name):
        return False
    if await _owned_point(client, owner.username, name, embedding_id) is None:
        return False

    # Delete through the scoped id filter rather than the raw id, so the
    # ownership check and the delete cannot disagree.
    await client.delete(
        collection_name=name,
        points_selector=FilterSelector(
            filter=filters.scoped_ids(owner.username, [embedding_id])
        ),
        wait=True,
    )
    return True


# --- Search --------------------------------------------------------------


async def retrieve_embeddings(
    client: AsyncQdrantClient,
    owner: Owner,
    query_embedding: list[float],
    top_k: int,
    collection: str,
    topic: str | None = None,
    metadata_filter: MetadataFilter | None = None,
) -> list[dict[str, Any]]:
    name = await ensure_collection(client, owner, collection, vector_dim=None)
    response = await client.query_points(
        collection_name=name,
        query=query_embedding,
        query_filter=filters.scoped(owner.username, topic, metadata_filter),
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )
    return [_hit(item, collection) for item in response.points]


async def fetch_candidates(
    client: AsyncQdrantClient,
    owner: Owner,
    query_embedding: list[float],
    fetch_k: int,
    collection: str,
    topic: str | None = None,
    metadata_filter: MetadataFilter | None = None,
) -> list[dict[str, Any]]:
    """Like retrieve, but returns vectors too — the rerankers need them."""
    name = await ensure_collection(client, owner, collection, vector_dim=None)
    response = await client.query_points(
        collection_name=name,
        query=query_embedding,
        query_filter=filters.scoped(owner.username, topic, metadata_filter),
        limit=fetch_k,
        with_payload=True,
        with_vectors=True,
    )
    return [_hit(item, collection, with_vector=True) for item in response.points]


# --- Collections ---------------------------------------------------------
#
# Each user's collections are physically their own (see app.db.naming), so the
# operations below are naturally scoped: listing is a prefix scan, and deleting
# only ever touches collections inside the caller's namespace.


async def _distinct_values(
    client: AsyncQdrantClient, name: str, key: str, scope: Any
) -> tuple[list[str], bool]:
    """Distinct values of a payload key within the caller's scope.

    Prefers the server-side facet API; falls back to a bounded payload scroll
    when the Qdrant server predates it (facet landed in 1.12).
    """
    try:
        response = await client.facet(
            collection_name=name, key=key, facet_filter=scope, limit=_FACET_LIMIT
        )
        values = sorted(str(h.value) for h in response.hits if h.value is not None)
        return values, len(values) >= _FACET_LIMIT
    except Exception:
        logger.debug("facet unavailable for '%s', falling back to scroll", name)

    values: set[str] = set()
    offset: Any = None
    seen = 0
    while seen < _FACET_SCROLL_CAP:
        records, offset = await client.scroll(
            collection_name=name,
            scroll_filter=scope,
            limit=_SCROLL_PAGE,
            offset=offset,
            with_payload=[key],
            with_vectors=False,
        )
        if not records:
            break
        seen += len(records)
        for record in records:
            value = (record.payload or {}).get(key)
            if value is not None:
                values.add(str(value))
        if offset is None:
            break
    return sorted(values), seen >= _FACET_SCROLL_CAP


async def _count(client: AsyncQdrantClient, name: str, scope: Any | None = None) -> int:
    return (await client.count(collection_name=name, count_filter=scope, exact=True)).count


async def _collection_stats(
    client: AsyncQdrantClient,
    owner: Owner,
    physical: str,
    points: int,
    detail: str = "full",
) -> dict[str, Any]:
    """Stats for one collection.

    ``detail="basic"`` omits topics and embedding models, each of which costs a
    facet call, halving the Qdrant calls per collection.
    """
    info = await client.get_collection(collection_name=physical)
    stats: dict[str, Any] = {
        "collection": display_name(owner, physical),
        "points": points,
        "vector_dim": getattr(info.config.params.vectors, "size", None),
        "topics": None,
        "embedding_models": None,
    }
    stats["truncated"] = False
    if detail == "full":
        scope = filters.owned_by(owner.username)
        topics, t1 = await _distinct_values(client, physical, filters.TOPIC_FIELD, scope)
        models, t2 = await _distinct_values(client, physical, "embedding_model", scope)
        stats["topics"] = topics
        stats["embedding_models"] = models
        stats["truncated"] = t1 or t2
    return stats


async def list_user_collections(
    client: AsyncQdrantClient,
    owner: Owner,
    detail: str = "full",
    limit: int | None = None,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """List the caller's collections.

    A prefix scan, so collections belonging to other users are never inspected.

    Each collection needs four independent Qdrant calls, so they are gathered
    concurrently rather than serially, bounded by a semaphore.
    """
    physicals = await owned_collections(client, owner)
    total = len(physicals)
    # Slice before the per-collection lookups, so cost tracks the page not the total.
    page = physicals[offset : offset + limit] if limit is not None else physicals[offset:]
    if not page:
        return [], total

    limiter = asyncio.Semaphore(_LIST_CONCURRENCY)

    async def stats_for(physical: str) -> dict[str, Any]:
        async with limiter:
            points = await _count(client, physical, filters.owned_by(owner.username))
            return await _collection_stats(client, owner, physical, points, detail)

    result = await asyncio.gather(*(stats_for(name) for name in page))
    return sorted(result, key=lambda item: item["collection"] or ""), total


async def get_user_collection(
    client: AsyncQdrantClient, owner: Owner, collection: str
) -> dict[str, Any] | None:
    """Stats for one of the caller's collections, or None if they have no such one."""
    name = collection_name(owner, collection)
    if not await collection_exists(client, name):
        return None
    points = await _count(client, name, filters.owned_by(owner.username))
    return await _collection_stats(client, owner, name, points, detail="full")


async def list_collection_embeddings(
    client: AsyncQdrantClient,
    owner: Owner,
    collection: str,
    limit: int,
    offset: str | None = None,
    topic: str | None = None,
    metadata_filter: MetadataFilter | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Page through the caller's embeddings. Returns ``(records, next_offset)``."""
    name = collection_name(owner, collection)
    if not await collection_exists(client, name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{collection}' not found.",
        )

    records, next_offset = await client.scroll(
        collection_name=name,
        scroll_filter=filters.scoped(owner.username, topic, metadata_filter),
        limit=limit,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )
    return (
        [_record(str(r.id), r.payload or {}) for r in records],
        str(next_offset) if next_offset is not None else None,
    )


async def _maybe_drop_empty(
    client: AsyncQdrantClient, owner: Owner, physical: str
) -> bool:
    """Drop the collection if the caller has emptied it.

    Safe by construction: ``physical`` is inside ``owner``'s namespace, so no
    other user can be writing into it and the "is it empty?" / "drop it" pair
    has no window in which someone else's data could be destroyed. The guard
    below re-asserts ownership so a future refactor cannot turn this into a
    cross-user delete.
    """
    if not settings.drop_empty_collections:
        return False
    if not owns(owner, physical):  # pragma: no cover - defensive
        logger.error(
            "Refusing to drop '%s': outside the namespace of user '%s'",
            physical,
            owner.username,
        )
        return False
    if await _count(client, physical):
        return False
    await client.delete_collection(collection_name=physical)
    logger.info("Dropped emptied collection '%s'", physical)
    return True


async def bulk_delete_embeddings(
    client: AsyncQdrantClient,
    owner: Owner,
    collection: str,
    ids: list[str] | None = None,
    topic: str | None = None,
    metadata_filter: MetadataFilter | None = None,
    delete_all: bool = False,
) -> tuple[int, bool]:
    """Delete many of the caller's points in one of their collections.

    The collection addressed is always inside the caller's namespace, and the
    delete additionally runs through a ``user_id`` filter — including the by-id
    path, where ``HasIdCondition`` is ANDed with the owner condition.

    Returns ``(deleted_count, collection_dropped)``.
    """
    name = collection_name(owner, collection)
    if not await collection_exists(client, name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{collection}' not found.",
        )

    if delete_all:
        selector = filters.owned_by(owner.username)
    elif ids:
        selector = filters.scoped_ids(owner.username, ids)
    else:
        selector = filters.scoped(owner.username, topic, metadata_filter)

    # Count first: Qdrant's delete response reports operation status, not how
    # many points matched.
    matched = await _count(client, name, selector)
    if matched == 0:
        return 0, False

    await client.delete(
        collection_name=name, points_selector=FilterSelector(filter=selector), wait=True
    )
    return matched, await _maybe_drop_empty(client, owner, name)


async def delete_user_collection(
    client: AsyncQdrantClient, owner: Owner, collection: str
) -> tuple[int, bool]:
    """Delete one of the caller's collections outright.

    Because the collection is physically theirs, this drops it rather than
    emptying it point by point — and no other user can be affected.
    """
    name = collection_name(owner, collection)
    if not await collection_exists(client, name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{collection}' not found.",
        )

    deleted = await _count(client, name, filters.owned_by(owner.username))
    dropped = False
    if settings.drop_empty_collections:
        await client.delete_collection(collection_name=name)
        dropped = True
        logger.info("Dropped collection '%s'", name)
    else:
        await client.delete(
            collection_name=name,
            points_selector=FilterSelector(filter=filters.owned_by(owner.username)),
            wait=True,
        )
    return deleted, dropped


async def purge_user_data(
    client: AsyncQdrantClient, owner: Owner
) -> tuple[int, list[str], list[str]]:
    """Delete everything the caller owns, across all of their collections.

    Returns ``(total_deleted, collections_affected, collections_dropped)``.
    """
    physicals = await owned_collections(client, owner)
    if not physicals:
        return 0, [], []

    scope = filters.owned_by(owner.username)
    limiter = asyncio.Semaphore(_LIST_CONCURRENCY)

    async def purge_one(physical: str) -> tuple[str, int, bool]:
        async with limiter:
            matched = await _count(client, physical, scope)
            name = display_name(owner, physical) or physical
            if settings.drop_empty_collections:
                await client.delete_collection(collection_name=physical)
                return name, matched, True
            await client.delete(
                collection_name=physical,
                points_selector=FilterSelector(filter=scope),
                wait=True,
            )
            return name, matched, False

    outcomes = await asyncio.gather(*(purge_one(p) for p in physicals))
    total = sum(m for _, m, _ in outcomes)
    affected = sorted(n for n, _, _ in outcomes)
    dropped = sorted(n for n, _, d in outcomes if d)
    return total, affected, dropped


async def backfill_payload_indexes(client: AsyncQdrantClient) -> int:
    """Ensure the payload indexes exist on every collection.

    New collections get them at creation, but collections created before payload
    indexing was introduced have none — and an unindexed filter forces a full
    scan on every search. Creating an existing index is a no-op, so this is safe
    to run on every startup. Returns the number of collections seen.
    """
    collections = await client.get_collections()
    names = [col.name for col in collections.collections or []]
    for name in names:
        await _create_payload_indexes(client, name)
    return len(names)
