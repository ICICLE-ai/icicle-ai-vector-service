"""Shared request/response pieces."""

from typing import Any

from pydantic import BaseModel, Field


class MetadataFilter(BaseModel):
    """Filter results by metadata fields.

    Each key-value pair matches against the nested ``metadata`` payload field.
    All conditions are ANDed together. Values can be:
      - a string/number/bool for exact match
      - a list for "any of" match

    Keys are namespaced under ``metadata.`` when translated to Qdrant, so this
    cannot be used to address a top-level payload field such as ``user_id``.
    """

    conditions: dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value pairs to match against the metadata payload field",
    )


class HealthResponse(BaseModel):
    status: str
    version: str
    qdrant: str = Field(description="'ok' when Qdrant answered, 'unreachable' otherwise.")
    cross_encoder: bool = Field(
        description="Whether cross-encoder reranking is installed in this deployment."
    )
