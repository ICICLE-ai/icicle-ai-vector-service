"""Unit tests for per-user collection naming.

Naming *is* the isolation mechanism now: if two distinct users could ever
produce the same physical collection name, their data would silently merge. The
first class below is the one that matters.
"""

import pytest

from src.app.db.naming import (
    SEPARATOR,
    Owner,
    display_name,
    owns,
    physical_name,
    slugify,
)

TENANT = "icicleai"


class TestNamespaceUniqueness:
    """No two distinct identities may share a namespace."""

    def test_different_usernames_differ(self):
        assert Owner("alice", TENANT).namespace != Owner("bob", TENANT).namespace

    def test_different_tenants_differ(self):
        """Same username in two tenants must not collide."""
        assert Owner("alice", "tenant-a").namespace != Owner("alice", "tenant-b").namespace

    @pytest.mark.parametrize(
        "first,second",
        [
            ("a.b", "a-b"),      # both slugify to "a_b"
            ("a.b", "a b"),
            ("A.B", "a-b"),
            ("user!", "user?"),
            ("x__y", "x--y"),
        ],
    )
    def test_usernames_that_slugify_identically_still_differ(self, first, second):
        """The hash is what guarantees uniqueness; the readable part cannot.

        Without it, these pairs would share a namespace and merge two users'
        data into one set of collections.
        """
        assert slugify(first) == slugify(second)  # precondition of the test
        assert Owner(first, TENANT).namespace != Owner(second, TENANT).namespace

    def test_namespace_is_stable_across_calls(self):
        """It is derived, not random — the same user must resolve to the same place."""
        assert Owner("alice", TENANT).namespace == Owner("alice", TENANT).namespace

    def test_namespace_is_qdrant_safe(self):
        for username in ["alice", "a.b@example.org", "ünïcodé", "x" * 200, "!!!"]:
            namespace = Owner(username, TENANT).namespace
            assert namespace.replace("_", "").isalnum()
            assert len(namespace) <= 64
            assert SEPARATOR not in namespace


class TestOwner:
    def test_requires_a_username(self):
        with pytest.raises(ValueError, match="username"):
            Owner("", TENANT)

    def test_requires_a_tenant(self):
        with pytest.raises(ValueError, match="tenant_id"):
            Owner("alice", "")

    def test_is_frozen(self):
        owner = Owner("alice", TENANT)
        with pytest.raises(Exception):
            owner.username = "bob"

    def test_from_context(self):
        context = type("Ctx", (), {"username": "alice", "tenant_id": TENANT})()
        assert Owner.from_context(context) == Owner("alice", TENANT)

    def test_unnameable_username_still_produces_a_namespace(self):
        """A username of only punctuation must not yield an empty prefix."""
        assert Owner("!!!", TENANT).namespace.startswith("user_")


class TestPhysicalName:
    def test_same_collection_name_maps_to_different_collections_per_user(self):
        """The headline property: 'biology' is a different store for each user."""
        alice = physical_name(Owner("alice", TENANT), "biology")
        bob = physical_name(Owner("bob", TENANT), "biology")
        assert alice != bob
        assert alice.endswith("__biology")
        assert bob.endswith("__biology")

    def test_is_prefixed_with_the_owner_namespace(self):
        owner = Owner("alice", TENANT)
        assert physical_name(owner, "biology").startswith(owner.prefix)

    @pytest.mark.parametrize(
        "raw,expected_suffix",
        [
            ("biology", "biology"),
            ("Biology", "biology"),
            ("Biology Notes", "biology_notes"),
            ("bio-logy", "bio_logy"),
            ("  spaced  ", "spaced"),
        ],
    )
    def test_collection_part_is_slugified(self, raw, expected_suffix):
        name = physical_name(Owner("alice", TENANT), raw)
        assert name.endswith(f"{SEPARATOR}{expected_suffix}")

    @pytest.mark.parametrize("raw", ["", "   ", "!!!", "---"])
    def test_unusable_names_return_none(self, raw):
        assert physical_name(Owner("alice", TENANT), raw) is None

    def test_separator_splits_unambiguously(self):
        """Neither component may contain the separator, so the split is exact."""
        name = physical_name(Owner("a__b", TENANT), "c__d")
        assert name.count(SEPARATOR) == 1


class TestOwnership:
    def test_owns_own_collection(self):
        owner = Owner("alice", TENANT)
        assert owns(owner, physical_name(owner, "biology"))

    def test_does_not_own_another_users_collection(self):
        alice, bob = Owner("alice", TENANT), Owner("bob", TENANT)
        assert not owns(alice, physical_name(bob, "biology"))

    def test_does_not_own_an_unprefixed_legacy_collection(self):
        """Collections from the old shared-name scheme are not claimed by anyone."""
        assert not owns(Owner("alice", TENANT), "biology")

    def test_display_name_round_trips(self):
        owner = Owner("alice", TENANT)
        assert display_name(owner, physical_name(owner, "Biology Notes")) == "biology_notes"

    def test_display_name_is_none_for_another_users_collection(self):
        alice, bob = Owner("alice", TENANT), Owner("bob", TENANT)
        assert display_name(alice, physical_name(bob, "biology")) is None

    def test_a_prefix_lookalike_is_not_owned(self):
        """A namespace that merely starts with another's text must not match."""
        alice = Owner("alice", TENANT)
        assert not owns(alice, alice.namespace + "x__biology")
