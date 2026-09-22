"""Embedding create / update / delete models."""

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from .common import MetadataFilter


def _require_text(value: str, field: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


class EmbeddingCreate(BaseModel):
    # Note: there is deliberately no user_id field. Ownership comes from the
    # verified token, so a client cannot write data under another user's name —
    # an extra "user_id" key in the body is ignored.
    embedding: list[float]
    collection: str
    topic: str | None = None
    chunks: list[str]
    token_ids: list[int] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding_model: str

    @field_validator("collection")
    @classmethod
    def validate_collection(cls, value: str) -> str:
        return _require_text(value, "collection")

    @field_validator("topic")
    @classmethod
    def validate_topic(cls, value: str | None) -> str | None:
        return None if value is None else _require_text(value, "topic")

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("embedding must be a non-empty list of floats")
        return value

    @field_validator("embedding_model")
    @classmethod
    def validate_embedding_model(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError(
                "embedding_model is required (e.g. 'nvidia/nvclip', 'gemini-embedding-001')"
            )
        return value

    @model_validator(mode="after")
    def check_chunks(self) -> "EmbeddingCreate":
        if not self.chunks or any(not chunk.strip() for chunk in self.chunks):
            raise ValueError("chunks must contain at least one non-empty string")
        return self


class EmbeddingUpdate(BaseModel):
    embedding: list[float] | None = None
    topic: str | None = None
    chunks: list[str] | None = None
    token_ids: list[int] | None = None
    metadata: dict[str, Any] | None = None
    embedding_model: str | None = None

    @field_validator("topic")
    @classmethod
    def validate_topic(cls, value: str | None) -> str | None:
        return None if value is None else _require_text(value, "topic")

    @model_validator(mode="after")
    def ensure_updates(self) -> "EmbeddingUpdate":
        if all(
            getattr(self, field) is None
            for field in (
                "embedding",
                "topic",
                "token_ids",
                "chunks",
                "metadata",
                "embedding_model",
            )
        ):
            raise ValueError("At least one field must be provided to update")
        if self.chunks is not None and (
            not self.chunks or any(not chunk.strip() for chunk in self.chunks)
        ):
            raise ValueError("chunks must contain at least one non-empty string")
        return self


class EmbeddingRecord(BaseModel):
    id: str
    user_id: str
    collection: str
    topic: str | None = None
    vector_dim: int
    created_at: str
    updated_at: str
    embedding_model: str


class DeleteResponse(BaseModel):
    id: str
    user_id: str
    deleted: bool


class EmbeddingListResponse(BaseModel):
    """A page of the caller's embeddings in one collection (metadata only)."""

    user_id: str
    collection: str
    count: int
    next_offset: str | None = Field(
        None,
        description="Pass as ?offset= to fetch the next page. Null on the last page.",
    )
    embeddings: list[EmbeddingRecord]


class BulkDeleteRequest(BaseModel):
    """Delete many of the caller's embeddings in one collection.

    Exactly one selector must be given: explicit ``ids``, or a
    ``topic``/``filter`` predicate. ``all=true`` deletes every point the caller
    owns in the collection.
    """

    collection: str
    ids: list[str] | None = None
    topic: str | None = None
    filter: MetadataFilter | None = None
    all: bool = False

    @field_validator("collection")
    @classmethod
    def validate_collection(cls, value: str) -> str:
        return _require_text(value, "collection")

    @model_validator(mode="after")
    def check_selector(self) -> "BulkDeleteRequest":
        has_predicate = bool(self.topic) or bool(self.filter and self.filter.conditions)
        if self.all:
            if self.ids or has_predicate:
                raise ValueError(
                    "all=true deletes everything you own in the collection; "
                    "do not combine it with ids, topic or filter"
                )
            return self
        if self.ids and has_predicate:
            raise ValueError("Provide either ids or a topic/filter predicate, not both")
        if not self.ids and not has_predicate:
            raise ValueError(
                "Nothing selected. Provide ids, a topic/filter predicate, or all=true."
            )
        if self.ids is not None and not self.ids:
            raise ValueError("ids must not be an empty list")
        return self


class BulkDeleteResponse(BaseModel):
    user_id: str
    collection: str
    deleted: int
    collection_dropped: bool = Field(
        False,
        description="True when the underlying Qdrant collection was removed "
        "because no points from any user remained. Always false unless the "
        "deployment sets DROP_EMPTY_COLLECTIONS=true.",
    )
