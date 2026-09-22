"""Collection listing and purge models."""

from pydantic import BaseModel, Field


class CollectionInfo(BaseModel):
    """A collection as seen by the requesting user.

    ``points`` counts only the caller's own points, never every user's, so this
    cannot be used to infer how much data other users hold.
    """

    collection: str
    points: int
    topics: list[str] = Field(default_factory=list)
    vector_dim: int | None = None
    embedding_models: list[str] = Field(default_factory=list)


class CollectionList(BaseModel):
    user_id: str
    count: int
    collections: list[CollectionInfo]


class PurgeResponse(BaseModel):
    """Result of deleting everything the caller owns, across all collections."""

    user_id: str
    deleted: int
    collections_affected: list[str]
    collections_dropped: list[str]
