"""Qdrant filter construction — the tenancy boundary.

Every query, scroll, count and delete is filtered by ``user_id``. That filter is
built here and nowhere else, so the isolation guarantee is one testable module.

The invariant: every Filter returned pins ``user_id`` to exactly one user, and no
caller-supplied input can widen, replace or escape it. Client conditions are only
ever appended to ``must`` (logical AND), never placed in ``should`` or ``must_not``,
so they can narrow a result set but never broaden it.
"""

from __future__ import annotations

from typing import Any

from qdrant_client.http.models import (
    FieldCondition,
    Filter,
    HasIdCondition,
    MatchAny,
    MatchValue,
)

from ..schemas.common import MetadataFilter

# Payload key holding the owner's username, written from the JWT on every point.
USER_ID_FIELD = "user_id"
TOPIC_FIELD = "topic"
METADATA_PREFIX = "metadata"


def owner_condition(user_id: str) -> FieldCondition:
    """The single condition that makes a query another user's data impossible."""
    if not user_id:
        # Defensive: an empty user_id would match points whose user_id is "",
        # so refuse rather than build a filter that could over-match.
        raise ValueError("user_id must not be empty.")
    return FieldCondition(key=USER_ID_FIELD, match=MatchValue(value=user_id))


def owned_by(user_id: str) -> Filter:
    """Everything ``user_id`` owns, and nothing else."""
    return Filter(must=[owner_condition(user_id)])


def _metadata_conditions(metadata_filter: MetadataFilter | None) -> list[FieldCondition]:
    """Translate the client's metadata predicate into Qdrant conditions.

    A list value becomes "any of"; anything else is an exact match. Keys are
    namespaced under ``metadata.`` so a client cannot address a top-level
    payload field such as ``user_id`` through this path.
    """
    if not metadata_filter or not metadata_filter.conditions:
        return []

    conditions: list[FieldCondition] = []
    for key, value in metadata_filter.conditions.items():
        field_path = f"{METADATA_PREFIX}.{key}"
        if isinstance(value, list):
            conditions.append(FieldCondition(key=field_path, match=MatchAny(any=value)))
        else:
            conditions.append(
                FieldCondition(key=field_path, match=MatchValue(value=value))
            )
    return conditions


def scoped(
    user_id: str,
    topic: str | None = None,
    metadata_filter: MetadataFilter | None = None,
) -> Filter:
    """A user-scoped filter, optionally narrowed by topic and metadata.

    Used by search, scroll, count and predicate deletes alike.
    """
    must: list[Any] = [owner_condition(user_id)]
    if topic:
        must.append(FieldCondition(key=TOPIC_FIELD, match=MatchValue(value=topic)))
    must.extend(_metadata_conditions(metadata_filter))
    return Filter(must=must)


def scoped_ids(user_id: str, ids: list[str]) -> Filter:
    """Those of ``ids`` that ``user_id`` actually owns.

    Deleting by raw id list would let a caller remove any point whose id they
    can guess. ANDing ``HasIdCondition`` with the owner condition means an id
    belonging to someone else simply matches nothing.
    """
    if not ids:
        raise ValueError("ids must not be empty.")
    return Filter(must=[owner_condition(user_id), HasIdCondition(has_id=list(ids))])
