"""Retrieve and rerank models."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .common import MetadataFilter

# Rerank methods the service accepts. "cross_encoder" additionally requires
# query_text and a loadable reranker model.
RerankMethod = Literal["mmr", "cosine_rescore", "cross_encoder"]


class _QueryBase(BaseModel):
    query_embedding: list[float]
    collection: str
    topic: str | None = None
    filter: MetadataFilter | None = None

    @field_validator("collection")
    @classmethod
    def validate_collection(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("collection must be a non-empty string.")
        return value

    @field_validator("query_embedding")
    @classmethod
    def validate_embedding(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("query_embedding must be a non-empty list of floats.")
        return value


class RetrieveRequest(_QueryBase):
    top_k: int = Field(10, ge=1, le=100)


class RerankRequest(_QueryBase):
    query_text: str | None = Field(
        None,
        description="Raw query string. Required for method='cross_encoder', which "
        "scores the query against each candidate's text. Ignored by the "
        "vector-only methods ('mmr', 'cosine_rescore').",
    )
    top_k: int = Field(10, ge=1, le=100)
    fetch_k: int = Field(50, ge=1, le=500)
    method: RerankMethod = Field("mmr")
    rerank_model: str | None = Field(
        None,
        description="Cross-encoder model to use. Must be one of the models the "
        "deployment allows (see GET /v1/rerank/methods). Defaults to the "
        "service's configured model. Only used by method='cross_encoder'.",
    )
    lambda_: float = Field(
        0.7,
        ge=0.0,
        le=1.0,
        alias="lambda",
        description="MMR relevance/diversity trade-off. Only used by method='mmr'.",
    )

    @model_validator(mode="after")
    def check_cross_encoder(self) -> "RerankRequest":
        if self.method == "cross_encoder" and not (
            self.query_text and self.query_text.strip()
        ):
            raise ValueError(
                "query_text is required when method is 'cross_encoder', which "
                "scores the query text against each candidate's text."
            )
        return self


class ResultItem(BaseModel):
    id: str
    score: float
    collection: str
    topic: str | None = None
    text: str | None = None
    chunks: list[str]
    metadata: dict


class RerankedItem(ResultItem):
    """A result carrying both the vector score and the reranker's score.

    ``score`` stays the original cosine similarity from the vector search so
    clients can compare the two rankings; ``rerank_score`` is what the ordering
    is actually based on.
    """

    rerank_score: float


class RetrieveResponse(BaseModel):
    user_id: str
    top_k: int
    results: list[ResultItem]


class RerankResponse(BaseModel):
    user_id: str
    method: RerankMethod
    model: str | None = Field(
        None, description="Reranker model used. Null for the vector-only methods."
    )
    top_k: int
    fetch_k: int
    results: list[RerankedItem]


class RerankMethodInfo(BaseModel):
    name: RerankMethod
    kind: str = Field(description="'vector' (embedding math) or 'cross-encoder' (model).")
    available: bool = Field(
        description="False when the method's dependencies are missing from this deployment."
    )
    requires_query_text: bool
    description: str


class RerankMethodList(BaseModel):
    methods: list[RerankMethodInfo]
    default_model: str
    allowed_models: list[str]
