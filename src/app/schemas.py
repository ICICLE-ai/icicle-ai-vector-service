from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from .settings import settings


class EmbeddingCreate(BaseModel):
    embedding: list[float]
    collection: str
    topic: str | None = None
    chunks: list[str]
    token_ids: list[int] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding_model: str | None = None

    @field_validator("collection")
    @classmethod
    def validate_collection(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("collection must be a non-empty string")
        return value

    @field_validator("topic")
    @classmethod
    def validate_topic(cls, value: str | None) -> str | None:
        if value is not None:
            value = value.strip()
            if not value:
                raise ValueError("topic must be a non-empty string if provided")
        return value

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, value: list[float]) -> list[float]:
        if len(value) != settings.vector_dim:
            raise ValueError(
                f"embedding must have exactly {settings.vector_dim} dimensions, got {len(value)}"
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
        if value is not None:
            value = value.strip()
            if not value:
                raise ValueError("topic must be a non-empty string")
        return value

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, value: list[float] | None) -> list[float] | None:
        if value is not None and len(value) != settings.vector_dim:
            raise ValueError(
                f"embedding must have exactly {settings.vector_dim} dimensions, got {len(value)}"
            )
        return value

    @model_validator(mode="after")
    def ensure_updates(self) -> "EmbeddingUpdate":
        if (
            self.embedding is None
            and self.topic is None
            and self.token_ids is None
            and self.chunks is None
            and self.metadata is None
            and self.embedding_model is None
        ):
            raise ValueError("At least one field must be provided to update")
        if self.chunks is not None:
            if not self.chunks or any(not chunk.strip() for chunk in self.chunks):
                raise ValueError("chunks must contain at least one non-empty string")
        return self


class EmbeddingRecord(BaseModel):
    id: str
    user_id: str
    collection: str
    topic: str | None = None
    created_at: str
    updated_at: str
    embedding_model: str | None = None


class MetadataFilter(BaseModel):
    """Filter search results by metadata fields.

    Each key-value pair matches against the nested ``metadata`` payload field.
    All conditions are ANDed together.  Values can be:
      - a string/number/bool for exact match
      - a list for "any of" match
    """

    conditions: dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value pairs to match against the metadata payload field",
    )


class RetrieveRequest(BaseModel):
    query_embedding: list[float]
    top_k: int = Field(10, ge=1, le=100)
    collection: str
    topic: str | None = None
    filter: MetadataFilter | None = None

    @field_validator("collection")
    @classmethod
    def validate_collection(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("collection is required — specify which collection to search")
        return value

    @field_validator("query_embedding")
    @classmethod
    def validate_embedding(cls, value: list[float]) -> list[float]:
        if len(value) != settings.vector_dim:
            raise ValueError(
                f"query_embedding must have exactly {settings.vector_dim} dimensions, got {len(value)}"
            )
        return value



class RerankRequest(BaseModel):
    query_embedding: list[float]
    top_k: int = Field(10, ge=1, le=100)
    fetch_k: int = Field(50, ge=1, le=500)
    method: str = Field("mmr")
    lambda_: float = Field(0.7, ge=0.0, le=1.0, alias="lambda")
    collection: str
    topic: str | None = None
    filter: MetadataFilter | None = None

    @field_validator("collection")
    @classmethod
    def validate_collection(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("collection is required — specify which collection to rerank")
        return value

    @field_validator("query_embedding")
    @classmethod
    def validate_embedding(cls, value: list[float]) -> list[float]:
        if len(value) != settings.vector_dim:
            raise ValueError(
                f"query_embedding must have exactly {settings.vector_dim} dimensions, got {len(value)}"
            )
        return value


class ResultItem(BaseModel):
    id: str
    score: float
    collection: str
    topic: str | None = None
    text: str | None = None
    chunks: list[str]
    metadata: dict[str, Any]


class RetrieveResponse(BaseModel):
    user_id: str
    top_k: int
    results: list[ResultItem]


class DeleteResponse(BaseModel):
    id: str
    user_id: str
    deleted: bool


class RerankResponse(BaseModel):
    user_id: str
    method: str
    top_k: int
    fetch_k: int
    results: list[ResultItem]
