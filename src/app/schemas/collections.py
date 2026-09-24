"""Collection listing and purge models."""

from typing import Literal

from pydantic import BaseModel, Field

# "basic" returns names and point counts; "full" also derives topics and
# embedding models, which costs two extra Qdrant calls per collection.
CollectionDetail = Literal["basic", "full"]


class CollectionInfo(BaseModel):
    """A collection as seen by the requesting user.

    ``points`` counts only the caller's own points, never every user's, so this
    cannot be used to infer how much data other users hold.
    """

    collection: str
    points: int
    vector_dim: int | None = None
    topics: list[str] | None = Field(
        None, description="Null unless detail=full was requested."
    )
    embedding_models: list[str] | None = Field(
        None, description="Null unless detail=full was requested."
    )


class CollectionList(BaseModel):
    user_id: str
    count: int
    detail: CollectionDetail
    collections: list[CollectionInfo]


class PurgeResponse(BaseModel):
    """Result of deleting everything the caller owns, across all collections."""

    user_id: str
    deleted: int
    collections_affected: list[str]
    collections_dropped: list[str]
