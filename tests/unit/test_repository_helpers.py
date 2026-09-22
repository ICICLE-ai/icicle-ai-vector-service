"""Unit tests for repository helpers that need no Qdrant."""

import pytest
from fastapi import HTTPException

from src.app.db.naming import Owner
from src.app.db.repository import _hit, _record, collection_name


class TestCollectionName:
    """collection_name() resolves a user-facing name inside the caller's namespace."""

    def test_resolves_within_the_owner_namespace(self):
        owner = Owner("alice", "icicleai")
        assert collection_name(owner, "biology").startswith(owner.prefix)

    def test_two_users_get_different_physical_collections(self):
        alice = collection_name(Owner("alice", "icicleai"), "biology")
        bob = collection_name(Owner("bob", "icicleai"), "biology")
        assert alice != bob

    @pytest.mark.parametrize("raw", ["", "   ", "!!!", "---"])
    def test_names_with_no_usable_characters_are_rejected(self, raw):
        with pytest.raises(HTTPException) as excinfo:
            collection_name(Owner("alice", "icicleai"), raw)
        assert excinfo.value.status_code == 422


class TestProjections:
    def test_record_defaults_missing_fields(self):
        record = _record("abc", {})
        assert record["id"] == "abc"
        assert record["user_id"] == ""
        assert record["vector_dim"] == 0

    def test_record_reads_payload(self):
        record = _record("abc", {"user_id": "alice", "topic": "plant", "vector_dim": 8})
        assert record["user_id"] == "alice"
        assert record["topic"] == "plant"
        assert record["vector_dim"] == 8

    def test_hit_omits_embedding_by_default(self):
        item = type("P", (), {"id": 1, "score": 0.5, "payload": {}, "vector": [1.0]})()
        assert "embedding" not in _hit(item, "c")

    def test_hit_includes_embedding_when_asked(self):
        item = type("P", (), {"id": 1, "score": 0.5, "payload": {}, "vector": [1.0]})()
        assert _hit(item, "c", with_vector=True)["embedding"] == [1.0]

    def test_hit_handles_missing_vector(self):
        item = type("P", (), {"id": 1, "score": 0.5, "payload": {}, "vector": None})()
        assert _hit(item, "c", with_vector=True)["embedding"] == []


class TestDistinctValuesFallback:
    """The facet API is only on Qdrant >= 1.12; older servers must still work."""

    async def test_falls_back_to_scroll_when_facet_is_unavailable(self):
        from qdrant_client import AsyncQdrantClient

        from src.app.db import filters
        from src.app.db.repository import _distinct_values

        memory = AsyncQdrantClient(location=":memory:")
        try:
            # In-memory Qdrant has no facet support, so this exercises the
            # fallback path without having to stub anything.
            await memory.create_collection(
                collection_name="c",
                vectors_config=__import__(
                    "qdrant_client.http.models", fromlist=["VectorParams"]
                ).VectorParams(size=2, distance="Cosine"),
            )
            from qdrant_client.http.models import PointStruct

            await memory.upsert(
                collection_name="c",
                points=[
                    PointStruct(
                        id=1, vector=[1.0, 0.0], payload={"user_id": "alice", "topic": "a"}
                    ),
                    PointStruct(
                        id=2, vector=[0.0, 1.0], payload={"user_id": "alice", "topic": "b"}
                    ),
                    PointStruct(
                        id=3, vector=[0.0, 1.0], payload={"user_id": "bob", "topic": "secret"}
                    ),
                ],
                wait=True,
            )

            topics = await _distinct_values(memory, "c", "topic", filters.owned_by("alice"))
            assert topics == ["a", "b"]
            assert "secret" not in topics
        finally:
            await memory.close()


class TestFacetFallback:
    """Older Qdrant servers (< 1.12) have no facet API; the scroll path must work."""

    async def test_scroll_fallback_is_scoped_and_paginates(self, monkeypatch):
        from qdrant_client import AsyncQdrantClient
        from qdrant_client.http.models import PointStruct, VectorParams

        from src.app.db import filters, repository

        memory = AsyncQdrantClient(location=":memory:")
        try:
            await memory.create_collection(
                collection_name="c",
                vectors_config=VectorParams(size=2, distance="Cosine"),
            )
            # More points than one scroll page, to exercise the paging loop.
            monkeypatch.setattr(repository, "_SCROLL_PAGE", 2)
            await memory.upsert(
                collection_name="c",
                points=[
                    PointStruct(
                        id=i,
                        vector=[1.0, 0.0],
                        payload={"user_id": "alice", "topic": f"t{i % 3}"},
                    )
                    for i in range(7)
                ]
                + [
                    PointStruct(
                        id=99,
                        vector=[0.0, 1.0],
                        payload={"user_id": "bob", "topic": "bob-secret"},
                    )
                ],
                wait=True,
            )

            async def _no_facet(*args, **kwargs):
                raise RuntimeError("facet not supported on this server")

            monkeypatch.setattr(memory, "facet", _no_facet)

            topics = await repository._distinct_values(
                memory, "c", "topic", filters.owned_by("alice")
            )
            assert topics == ["t0", "t1", "t2"]
            assert "bob-secret" not in topics
        finally:
            await memory.close()

    async def test_scroll_fallback_respects_the_cap(self, monkeypatch):
        """A huge collection must not be scrolled end to end just to list topics."""
        from qdrant_client import AsyncQdrantClient
        from qdrant_client.http.models import PointStruct, VectorParams

        from src.app.db import filters, repository

        memory = AsyncQdrantClient(location=":memory:")
        try:
            await memory.create_collection(
                collection_name="c", vectors_config=VectorParams(size=2, distance="Cosine")
            )
            await memory.upsert(
                collection_name="c",
                points=[
                    PointStruct(
                        id=i, vector=[1.0, 0.0], payload={"user_id": "alice", "topic": f"t{i}"}
                    )
                    for i in range(10)
                ],
                wait=True,
            )

            async def _no_facet(*args, **kwargs):
                raise RuntimeError("no facet")

            monkeypatch.setattr(memory, "facet", _no_facet)
            monkeypatch.setattr(repository, "_SCROLL_PAGE", 2)
            monkeypatch.setattr(repository, "_FACET_SCROLL_CAP", 4)

            topics = await repository._distinct_values(
                memory, "c", "topic", filters.owned_by("alice")
            )
            assert len(topics) == 4  # stopped at the cap rather than reading all 10
        finally:
            await memory.close()
