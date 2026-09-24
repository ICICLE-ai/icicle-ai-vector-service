"""Collection discovery, pagination, bulk delete and purge."""

import pytest

from tests.conftest import store, vector

pytestmark = pytest.mark.asyncio


class TestListing:
    async def test_empty_when_nothing_stored(self, client):
        body = (await client.get("/v1/collections")).json()
        assert body == {"user_id": "alice", "count": 0, "collections": []}

    async def test_reports_points_topics_dim_and_models(self, client):
        await store(client, collection="biology", topic="plant")
        await store(client, collection="biology", topic="human")
        await store(client, collection="chemistry", topic="organic")

        body = (await client.get("/v1/collections")).json()
        assert body["count"] == 2
        by_name = {c["collection"]: c for c in body["collections"]}
        assert by_name["biology"]["points"] == 2
        assert sorted(by_name["biology"]["topics"]) == ["human", "plant"]
        assert by_name["biology"]["vector_dim"] == 4
        assert by_name["biology"]["embedding_models"] == ["test-model"]

    async def test_sorted_by_name(self, client):
        for name in ("zebra", "apple", "mango"):
            await store(client, collection=name)
        names = [c["collection"] for c in (await client.get("/v1/collections")).json()["collections"]]
        assert names == sorted(names)

    async def test_collection_names_are_slugified(self, client):
        await store(client, collection="Biology Notes")
        assert (await client.get("/v1/collections")).json()["collections"][0][
            "collection"
        ] == "biology_notes"

    async def test_listed_name_can_be_used_as_a_path_parameter(self, client):
        """Round-trip: whatever listing returns must address the same collection."""
        await store(client, collection="Biology Notes")
        name = (await client.get("/v1/collections")).json()["collections"][0]["collection"]
        assert (await client.get(f"/v1/collections/{name}")).status_code == 200


class TestSingleCollection:
    async def test_returns_scoped_stats(self, client):
        await store(client, topic="plant")
        body = (await client.get("/v1/collections/biology")).json()
        assert body["points"] == 1
        assert body["topics"] == ["plant"]

    async def test_unknown_collection_is_404(self, client):
        assert (await client.get("/v1/collections/nope")).status_code == 404


class TestEmbeddingListing:
    async def test_paginates_with_a_cursor(self, client):
        for i in range(5):
            await store(client, embedding=vector(float(i + 1)))

        first = (await client.get("/v1/collections/biology/embeddings?limit=2")).json()
        assert first["count"] == 2
        assert first["next_offset"] is not None

        second = (
            await client.get(
                f"/v1/collections/biology/embeddings?limit=2&offset={first['next_offset']}"
            )
        ).json()
        assert second["count"] == 2
        assert not {e["id"] for e in first["embeddings"]} & {
            e["id"] for e in second["embeddings"]
        }

    async def test_last_page_has_no_next_offset(self, client):
        await store(client)
        body = (await client.get("/v1/collections/biology/embeddings?limit=50")).json()
        assert body["count"] == 1
        assert body["next_offset"] is None

    async def test_filters_by_topic(self, client):
        await store(client, topic="plant")
        await store(client, topic="human", embedding=vector(2.0))
        body = (await client.get("/v1/collections/biology/embeddings?topic=plant")).json()
        assert body["count"] == 1
        assert body["embeddings"][0]["topic"] == "plant"

    async def test_never_returns_vectors(self, client):
        await store(client)
        body = (await client.get("/v1/collections/biology/embeddings")).json()
        assert "embedding" not in body["embeddings"][0]

    @pytest.mark.parametrize("limit", [0, 501])
    async def test_limit_bounds(self, client, limit):
        await store(client)
        assert (
            await client.get(f"/v1/collections/biology/embeddings?limit={limit}")
        ).status_code == 422

    async def test_unknown_collection_is_404(self, client):
        assert (await client.get("/v1/collections/ghost/embeddings")).status_code == 404


class TestBulkDelete:
    async def test_by_ids(self, client):
        ids = [await store(client, embedding=vector(float(i + 1))) for i in range(4)]
        body = (
            await client.post(
                "/v1/embeddings/bulk-delete", json={"collection": "biology", "ids": ids[:2]}
            )
        ).json()
        assert body["deleted"] == 2
        assert (await client.get("/v1/collections/biology")).json()["points"] == 2

    async def test_by_topic(self, client):
        await store(client, topic="plant")
        await store(client, topic="plant", embedding=vector(2.0))
        await store(client, topic="human", embedding=vector(3.0))

        body = (
            await client.post(
                "/v1/embeddings/bulk-delete", json={"collection": "biology", "topic": "plant"}
            )
        ).json()
        assert body["deleted"] == 2

    async def test_by_topic_and_metadata_together(self, client):
        await store(client, topic="plant", metadata={"source": "old.pdf"})
        await store(client, topic="plant", metadata={"source": "new.pdf"}, embedding=vector(2.0))
        await store(client, topic="human", metadata={"source": "old.pdf"}, embedding=vector(3.0))

        body = (
            await client.post(
                "/v1/embeddings/bulk-delete",
                json={
                    "collection": "biology",
                    "topic": "plant",
                    "filter": {"conditions": {"source": "old.pdf"}},
                },
            )
        ).json()
        assert body["deleted"] == 1
        assert (await client.get("/v1/collections/biology")).json()["points"] == 2

    async def test_all(self, client):
        await store(client)
        await store(client, embedding=vector(2.0))
        body = (
            await client.post(
                "/v1/embeddings/bulk-delete", json={"collection": "biology", "all": True}
            )
        ).json()
        assert body["deleted"] == 2
        assert (await client.get("/v1/collections")).json()["count"] == 0

    async def test_unknown_id_is_a_noop(self, client):
        await store(client)
        body = (
            await client.post(
                "/v1/embeddings/bulk-delete",
                json={
                    "collection": "biology",
                    "ids": ["00000000-0000-0000-0000-000000000000"],
                },
            )
        ).json()
        assert body["deleted"] == 0

    @pytest.mark.parametrize(
        "body",
        [
            {"collection": "biology"},
            {"collection": "biology", "all": True, "topic": "plant"},
            {"collection": "biology", "ids": ["a"], "topic": "plant"},
            {"collection": "biology", "ids": []},
            {"collection": "  "},
            {"collection": "biology", "filter": {"conditions": {}}},
        ],
    )
    async def test_invalid_selectors_rejected(self, client, body):
        await store(client)
        assert (
            await client.post("/v1/embeddings/bulk-delete", json=body)
        ).status_code == 422

    async def test_unknown_collection_is_404(self, client):
        assert (
            await client.post(
                "/v1/embeddings/bulk-delete", json={"collection": "ghost", "all": True}
            )
        ).status_code == 404


class TestDeleteCollection:
    async def test_deletes_everything_the_caller_owns(self, client):
        await store(client)
        await store(client, embedding=vector(2.0))

        body = (await client.delete("/v1/collections/biology")).json()
        assert body["deleted"] == 2
        assert (await client.get("/v1/collections")).json()["count"] == 0

    async def test_drops_the_collection_by_default(self, client):
        """Safe now that the collection is the caller's own, so it is the default."""
        await store(client)
        assert (await client.delete("/v1/collections/biology")).json()[
            "collection_dropped"
        ] is True
        assert (await client.get("/v1/collections")).json()["count"] == 0

    async def test_unknown_collection_is_404(self, client):
        assert (await client.delete("/v1/collections/ghost")).status_code == 404


class TestPurge:
    async def test_requires_confirmation(self, client):
        await store(client)
        assert (await client.delete("/v1/collections")).status_code == 400
        # The rejected call must not have deleted anything.
        assert (await client.get("/v1/collections")).json()["count"] == 1

    async def test_confirm_false_is_also_rejected(self, client):
        await store(client)
        assert (await client.delete("/v1/collections?confirm=false")).status_code == 400

    async def test_removes_everything_across_collections(self, client):
        await store(client, collection="biology")
        await store(client, collection="chemistry")

        body = (await client.delete("/v1/collections?confirm=true")).json()
        assert body["deleted"] == 2
        assert sorted(body["collections_affected"]) == ["biology", "chemistry"]
        assert (await client.get("/v1/collections")).json()["count"] == 0

    async def test_purging_nothing_is_still_ok(self, client):
        body = (await client.delete("/v1/collections?confirm=true")).json()
        assert body["deleted"] == 0
        assert body["collections_affected"] == []


class TestListingScales:
    """Listing gathers per-collection stats concurrently."""

    async def test_many_collections_are_all_listed(self, client):
        for i in range(25):
            await store(client, collection=f"c{i:02d}")
        body = (await client.get("/v1/collections")).json()
        assert body["count"] == 25
        assert len(body["collections"]) == 25

    async def test_listing_stays_sorted_despite_concurrency(self, client):
        """gather() completes out of order; the response must not."""
        for name in ("zebra", "apple", "mango", "kiwi", "banana"):
            await store(client, collection=name)
        names = [c["collection"] for c in (await client.get("/v1/collections")).json()["collections"]]
        assert names == sorted(names)

    async def test_stats_stay_attached_to_the_right_collection(self, client):
        """Concurrency must not shuffle points/topics between collections."""
        await store(client, collection="one", topic="alpha")
        for _ in range(3):
            await store(client, collection="two", topic="beta")

        by_name = {c["collection"]: c for c in (await client.get("/v1/collections")).json()["collections"]}
        assert by_name["one"]["points"] == 1
        assert by_name["one"]["topics"] == ["alpha"]
        assert by_name["two"]["points"] == 3
        assert by_name["two"]["topics"] == ["beta"]
