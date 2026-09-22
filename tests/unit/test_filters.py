"""Unit tests for the tenancy filter primitives.

The v1 tests prove isolation holds end to end. These prove it at the level where
it is actually implemented, so a regression points at the line that caused it.

The invariant under test: every Filter these functions return pins ``user_id``
to exactly one user via a ``must`` condition, and no caller-supplied input can
remove, replace or escape it.
"""

import pytest
from qdrant_client.http.models import FieldCondition, HasIdCondition

from src.app.db import filters
from src.app.schemas.common import MetadataFilter


def user_conditions(built) -> list[FieldCondition]:
    return [
        c
        for c in (built.must or [])
        if isinstance(c, FieldCondition) and c.key == filters.USER_ID_FIELD
    ]


def assert_scoped_to(built, username: str) -> None:
    """Every filter must carry exactly one, unavoidable user_id condition."""
    matches = user_conditions(built)
    assert len(matches) == 1, f"expected exactly one user_id condition, got {len(matches)}"
    assert matches[0].match.value == username
    # Nothing may be OR'd or negated at the top level: `should` would let a
    # result match without satisfying the user condition.
    assert not built.should
    assert not built.must_not


def test_owned_by_scopes_to_the_user():
    assert_scoped_to(filters.owned_by("alice"), "alice")


def test_scoped_without_extras_is_just_the_user():
    built = filters.scoped("alice")
    assert_scoped_to(built, "alice")
    assert len(built.must) == 1


def test_scoped_with_topic_keeps_the_user_condition():
    built = filters.scoped("alice", topic="plant")
    assert_scoped_to(built, "alice")
    topics = [c for c in built.must if getattr(c, "key", None) == filters.TOPIC_FIELD]
    assert len(topics) == 1
    assert topics[0].match.value == "plant"


def test_metadata_keys_are_namespaced():
    """A metadata key must never address a top-level payload field."""
    built = filters.scoped("alice", metadata_filter=MetadataFilter(conditions={"source": "a.pdf"}))
    keys = {getattr(c, "key", None) for c in built.must}
    assert "metadata.source" in keys
    assert "source" not in keys


def test_metadata_cannot_override_the_user_condition():
    """Asking for user_id in metadata targets metadata.user_id, not user_id."""
    built = filters.scoped(
        "alice", metadata_filter=MetadataFilter(conditions={"user_id": "bob"})
    )
    assert_scoped_to(built, "alice")
    assert "metadata.user_id" in {getattr(c, "key", None) for c in built.must}


@pytest.mark.parametrize(
    "conditions",
    [
        {"user_id": "bob"},
        {"": "bob"},
        {"..": "bob"},
        {"metadata.user_id": "bob"},
        {"a": "1", "user_id": "bob"},
    ],
)
def test_no_metadata_payload_can_strip_the_user_condition(conditions):
    """Whatever a client puts in `conditions`, the owner condition survives."""
    built = filters.scoped("alice", metadata_filter=MetadataFilter(conditions=conditions))
    assert_scoped_to(built, "alice")


def test_list_values_become_any_of():
    built = filters.scoped(
        "alice", metadata_filter=MetadataFilter(conditions={"source": ["a.pdf", "b.pdf"]})
    )
    assert_scoped_to(built, "alice")
    condition = next(c for c in built.must if getattr(c, "key", None) == "metadata.source")
    assert condition.match.any == ["a.pdf", "b.pdf"]


def test_scoped_ids_ands_ownership_with_the_ids():
    built = filters.scoped_ids("alice", ["id-1", "id-2"])
    assert_scoped_to(built, "alice")
    has_id = [c for c in built.must if isinstance(c, HasIdCondition)]
    assert len(has_id) == 1
    assert has_id[0].has_id == ["id-1", "id-2"]


def test_empty_user_id_is_refused():
    """An empty username would match points stored with an empty user_id."""
    for build in (
        lambda: filters.owned_by(""),
        lambda: filters.scoped(""),
        lambda: filters.scoped_ids("", ["id-1"]),
    ):
        with pytest.raises(ValueError, match="user_id"):
            build()


def test_empty_id_list_is_refused():
    """An empty HasIdCondition would be meaningless and risks matching broadly."""
    with pytest.raises(ValueError, match="ids"):
        filters.scoped_ids("alice", [])


def test_two_users_never_produce_equal_filters():
    assert filters.owned_by("alice") != filters.owned_by("bob")
