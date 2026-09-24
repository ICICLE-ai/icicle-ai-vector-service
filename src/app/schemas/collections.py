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
        None, description="Null unless detail=full. Capped at 100 distinct values."
    )
    embedding_models: list[str] | None = Field(
        None, description="Null unless detail=full. Capped at 100 distinct values."
    )
    truncated: bool = Field(
        False,
        description="True when topics or embedding_models hit the 100-value cap "
        "and the list is incomplete.",
    )


class CollectionList(BaseModel):
    user_id: str
    count: int
    total: int = Field(description="Collections you own, ignoring pagination.")
    detail: CollectionDetail
    next_offset: int | None = Field(
        None, description="Pass as ?offset= for the next page. Null on the last page."
    )
    collections: list[CollectionInfo]


class PurgeResponse(BaseModel):
    """Result of deleting everything the caller owns, across all collections."""

    user_id: str
    deleted: int
    collections_affected: list[str]
    collections_dropped: list[str]
