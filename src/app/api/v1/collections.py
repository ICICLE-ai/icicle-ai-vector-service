"""``/v1/collections`` — discover and delete collections.

Qdrant collections are shared across users; isolation is a ``user_id`` payload
filter. Two consequences drive the semantics here:

* **Listing is scoped.** A collection only appears if the caller owns at least
  one point in it, and every count/topic reflects only their points. Users do
  not learn what collections other users created.
* **Deleting is scoped.** Deleting a collection removes only the caller's
  points. The underlying Qdrant collection is dropped only once no points from
  any user remain — so one user can never destroy another user's data.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from qdrant_client import AsyncQdrantClient

from ...core import UserContext, get_current_user
from ...db import get_qdrant_client
from ...db.naming import Owner
from ...db.repository import (
    delete_user_collection,
    get_user_collection,
    list_collection_embeddings,
    list_user_collections,
    purge_user_data,
)
from ...schemas import (
    BulkDeleteResponse,
    CollectionDetail,
    CollectionInfo,
    CollectionList,
    EmbeddingListResponse,
    PurgeResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/collections", tags=["collections"])


@router.get("", response_model=CollectionList, summary="List your collections")
async def list_collections(
    detail: CollectionDetail = Query(
        "basic",
        description="'basic' returns names, point counts and dimensions. 'full' "
        "also derives topics and embedding models, at two extra Qdrant calls per "
        "collection.",
    ),
    client: AsyncQdrantClient = Depends(get_qdrant_client),
    current_user: UserContext = Depends(get_current_user),
) -> CollectionList:
    logger.info(
        "Listing collections for user '%s' (detail: %s)", current_user.username, detail
    )
    collections = await list_user_collections(
        client, Owner.from_context(current_user), detail=detail
    )
    return CollectionList(
        user_id=current_user.username,
        count=len(collections),
        detail=detail,
        collections=[CollectionInfo(**item) for item in collections],
    )


@router.delete(
    "",
    response_model=PurgeResponse,
    summary="Delete ALL your embeddings across every collection",
)
async def purge_all(
    confirm: bool = Query(
        False,
        description="Must be true. Guards against an accidental unqualified DELETE.",
    ),
    client: AsyncQdrantClient = Depends(get_qdrant_client),
    current_user: UserContext = Depends(get_current_user),
) -> PurgeResponse:
    """Irreversible: removes every point the caller owns, in every collection.

    Other users' data is untouched. Requires ``?confirm=true`` so that a stray
    ``DELETE /v1/collections`` cannot wipe an account by accident.
    """
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "This operation deletes every embedding in every collection you "
                "own and cannot be undone. Set the 'confirm' query parameter to "
                "true to proceed."
            ),
        )

    logger.warning("PURGE: deleting all data for user '%s'", current_user.username)
    deleted, affected, dropped = await purge_user_data(client, Owner.from_context(current_user))
    logger.warning(
        "PURGE complete for user '%s': %d embeddings across %d collections (%d dropped)",
        current_user.username,
        deleted,
        len(affected),
        len(dropped),
    )
    return PurgeResponse(
        user_id=current_user.username,
        deleted=deleted,
        collections_affected=affected,
        collections_dropped=dropped,
    )


@router.get(
    "/{collection}", response_model=CollectionInfo, summary="Get one collection's stats"
)
async def read_collection(
    collection: str,
    client: AsyncQdrantClient = Depends(get_qdrant_client),
    current_user: UserContext = Depends(get_current_user),
) -> CollectionInfo:
    info = await get_user_collection(client, Owner.from_context(current_user), collection)
    if info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{collection}' not found.",
        )
    return CollectionInfo(**info)


@router.get(
    "/{collection}/embeddings",
    response_model=EmbeddingListResponse,
    summary="List your embeddings in a collection",
)
async def list_embeddings(
    collection: str,
    limit: int = Query(50, ge=1, le=500, description="Page size"),
    offset: str | None = Query(
        None, description="Cursor from a previous page's next_offset"
    ),
    topic: str | None = Query(None, description="Only embeddings with this topic"),
    client: AsyncQdrantClient = Depends(get_qdrant_client),
    current_user: UserContext = Depends(get_current_user),
) -> EmbeddingListResponse:
    """Cursor-paginated listing. Returns metadata only — never vectors."""
    records, next_offset = await list_collection_embeddings(
        client,
        Owner.from_context(current_user),
        collection,
        limit=limit,
        offset=offset,
        topic=topic,
    )
    return EmbeddingListResponse(
        user_id=current_user.username,
        collection=collection,
        count=len(records),
        next_offset=next_offset,
        embeddings=records,
    )


@router.delete(
    "/{collection}",
    response_model=BulkDeleteResponse,
    summary="Delete your embeddings in one collection",
)
async def delete_collection(
    collection: str,
    client: AsyncQdrantClient = Depends(get_qdrant_client),
    current_user: UserContext = Depends(get_current_user),
) -> BulkDeleteResponse:
    """Removes every point the caller owns in this collection.

    Other users keep theirs. The Qdrant collection is dropped only if it is left
    empty for everyone — reported as ``collection_dropped``.
    """
    logger.info(
        "Deleting collection '%s' for user '%s'", collection, current_user.username
    )
    # The repository already answers 404 when the caller owns nothing here, so
    # reaching this point means at least one point was deleted.
    deleted, dropped = await delete_user_collection(
        client, Owner.from_context(current_user), collection
    )
    logger.info(
        "Deleted %d embeddings from '%s' for user '%s' (collection dropped: %s)",
        deleted,
        collection,
        current_user.username,
        dropped,
    )
    return BulkDeleteResponse(
        user_id=current_user.username,
        collection=collection,
        deleted=deleted,
        collection_dropped=dropped,
    )
