"""Vector similarity search."""

import pytest

from tests.conftest import store, vector

pytestmark = pytest.mark.asyncio


class TestRetrieve:
    async def test_returns_results_with_scores(self, client):
        await store(client, text="photosynthesis")
        body = (
            await client.post(
                "/v1/retrieve", json={"query_embedding": vector(), "collection": "biology"}
            )
        ).json()
        assert body["user_id"] == "alice"
        assert len(body["results"]) == 1
        result = body["results"][0]
        assert result["chunks"] == ["photosynthesis"]
        assert isinstance(result["score"], float)
        assert "embedding" not in result

    async def test_respects_top_k(self, client):
        for i in range(5):
            await store(client, embedding=vector(float(i + 1)))
        body = (
            await client.post(
                "/v1/retrieve",
                json={"query_embedding": vector(), "collection": "biology", "top_k": 2},
            )
        ).json()
        assert len(body["results"]) == 2

    async def test_orders_by_similarity(self, client):
        await store(client, text="far", embedding=[0.0, 1.0, 0.0, 0.0])
        await store(client, text="near", embedding=[1.0, 0.0, 0.0, 0.0])
        body = (
            await client.post(
                "/v1/retrieve",
                json={"query_embedding": [1.0, 0.0, 0.0, 0.0], "collection": "biology"},
            )
        ).json()
        assert body["results"][0]["chunks"] == ["near"]

    async def test_filters_by_topic(self, client):
        await store(client, topic="plant")
        await store(client, topic="human", embedding=vector(2.0))
        body = (
            await client.post(
                "/v1/retrieve",
                json={
                    "query_embedding": vector(),
                    "collection": "biology",
                    "topic": "plant",
                },
            )
        ).json()
        assert len(body["results"]) == 1
        assert body["results"][0]["topic"] == "plant"

    async def test_filters_by_metadata_exact_match(self, client):
        await store(client, metadata={"source": "a.pdf"})
        await store(client, metadata={"source": "b.pdf"}, embedding=vector(2.0))
        body = (
            await client.post(
                "/v1/retrieve",
                json={
                    "query_embedding": vector(),
                    "collection": "biology",
                    "filter": {"conditions": {"source": "b.pdf"}},
                },
            )
        ).json()
        assert len(body["results"]) == 1
        assert body["results"][0]["metadata"]["source"] == "b.pdf"

    async def test_filters_by_metadata_any_of(self, client):
        await store(client, metadata={"source": "a.pdf"})
        await store(client, metadata={"source": "b.pdf"}, embedding=vector(2.0))
        await store(client, metadata={"source": "c.pdf"}, embedding=vector(3.0))
        body = (
            await client.post(
                "/v1/retrieve",
                json={
                    "query_embedding": vector(),
                    "collection": "biology",
                    "filter": {"conditions": {"source": ["a.pdf", "b.pdf"]}},
                },
            )
        ).json()
        assert len(body["results"]) == 2

    async def test_multiple_metadata_conditions_are_anded(self, client):
        await store(client, metadata={"source": "a.pdf", "page": 1})
        await store(client, metadata={"source": "a.pdf", "page": 2}, embedding=vector(2.0))
        body = (
            await client.post(
                "/v1/retrieve",
                json={
                    "query_embedding": vector(),
                    "collection": "biology",
                    "filter": {"conditions": {"source": "a.pdf", "page": 1}},
                },
            )
        ).json()
        assert len(body["results"]) == 1

    async def test_unknown_collection_is_404(self, client):
        response = await client.post(
            "/v1/retrieve", json={"query_embedding": vector(), "collection": "ghost"}
        )
        assert response.status_code == 404

    async def test_no_matches_is_an_empty_list(self, client):
        await store(client, topic="plant")
        body = (
            await client.post(
                "/v1/retrieve",
                json={
                    "query_embedding": vector(),
                    "collection": "biology",
                    "topic": "nonexistent",
                },
            )
        ).json()
        assert body["results"] == []

    @pytest.mark.parametrize(
        "body",
        [
            {"collection": "biology"},
            {"query_embedding": [], "collection": "biology"},
            {"query_embedding": [1.0]},
            {"query_embedding": [1.0], "collection": "biology", "top_k": 0},
            {"query_embedding": [1.0], "collection": "biology", "top_k": 101},
        ],
    )
    async def test_invalid_requests_rejected(self, client, body):
        assert (await client.post("/v1/retrieve", json=body)).status_code == 422
