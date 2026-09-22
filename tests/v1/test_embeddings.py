"""CRUD on individual embeddings."""

import pytest

from tests.conftest import store, vector

pytestmark = pytest.mark.asyncio


class TestStore:
    async def test_returns_a_record(self, client):
        response = await client.post(
            "/v1/embeddings",
            json={
                "embedding": vector(),
                "collection": "biology",
                "topic": "plant",
                "chunks": ["photosynthesis"],
                "embedding_model": "test-model",
            },
        )
        body = response.json()
        assert response.status_code == 201
        assert body["user_id"] == "alice"
        assert body["collection"] == "biology"
        assert body["topic"] == "plant"
        assert body["vector_dim"] == 4
        assert body["created_at"] == body["updated_at"]

    async def test_topic_is_optional(self, client):
        assert (await client.post(
            "/v1/embeddings",
            json={
                "embedding": vector(),
                "collection": "biology",
                "chunks": ["x"],
                "embedding_model": "m",
            },
        )).json()["topic"] is None

    async def test_creates_the_collection_on_first_store(self, client):
        assert (await client.get("/v1/collections")).json()["count"] == 0
        await store(client)
        assert (await client.get("/v1/collections")).json()["count"] == 1

    async def test_dimension_mismatch_is_explained(self, client):
        """The error must name the real cause, not surface a raw Qdrant failure."""
        await store(client, embedding=[1.0, 2.0, 3.0, 4.0])

        response = await client.post(
            "/v1/embeddings",
            json={
                "embedding": [1.0, 2.0],
                "collection": "biology",
                "chunks": ["x"],
                "embedding_model": "m",
            },
        )
        assert response.status_code == 409
        detail = response.json()["detail"]
        assert "4-dimensional" in detail and "2" in detail

    @pytest.mark.parametrize(
        "body",
        [
            {"collection": "c", "chunks": ["x"], "embedding_model": "m"},
            {"embedding": [1.0], "chunks": ["x"], "embedding_model": "m"},
            {"embedding": [1.0], "collection": "c", "embedding_model": "m"},
            {"embedding": [1.0], "collection": "c", "chunks": ["x"]},
            {"embedding": [], "collection": "c", "chunks": ["x"], "embedding_model": "m"},
            {"embedding": [1.0], "collection": "c", "chunks": [" "], "embedding_model": "m"},
        ],
    )
    async def test_invalid_payloads_rejected(self, client, body):
        assert (await client.post("/v1/embeddings", json=body)).status_code == 422


class TestGet:
    async def test_returns_the_record(self, client):
        embedding_id = await store(client, topic="plant")
        response = await client.get(f"/v1/embeddings/{embedding_id}?collection=biology")
        assert response.status_code == 200
        assert response.json()["id"] == embedding_id
        assert response.json()["topic"] == "plant"

    async def test_unknown_id_is_404(self, client):
        await store(client)
        response = await client.get(
            "/v1/embeddings/00000000-0000-0000-0000-000000000000?collection=biology"
        )
        assert response.status_code == 404

    async def test_unknown_collection_is_404(self, client):
        embedding_id = await store(client)
        assert (
            await client.get(f"/v1/embeddings/{embedding_id}?collection=ghost")
        ).status_code == 404

    async def test_collection_param_is_required(self, client):
        embedding_id = await store(client)
        assert (await client.get(f"/v1/embeddings/{embedding_id}")).status_code == 422


class TestUpdate:
    async def test_partial_update_preserves_untouched_fields(self, client):
        embedding_id = await store(client, topic="plant")
        before = (await client.get(f"/v1/embeddings/{embedding_id}?collection=biology")).json()

        response = await client.put(
            f"/v1/embeddings/{embedding_id}?collection=biology",
            json={"topic": "molecular"},
        )
        body = response.json()
        assert response.status_code == 200
        assert body["topic"] == "molecular"
        assert body["created_at"] == before["created_at"]
        assert body["updated_at"] != before["updated_at"]
        assert body["embedding_model"] == before["embedding_model"]

    async def test_can_replace_the_vector(self, client):
        embedding_id = await store(client)
        response = await client.put(
            f"/v1/embeddings/{embedding_id}?collection=biology",
            json={"embedding": [9.0, 9.0, 9.0, 9.0]},
        )
        assert response.status_code == 200
        assert response.json()["vector_dim"] == 4

    async def test_replacing_with_a_wrong_dimension_is_rejected(self, client):
        embedding_id = await store(client)
        response = await client.put(
            f"/v1/embeddings/{embedding_id}?collection=biology",
            json={"embedding": [1.0, 2.0]},
        )
        assert response.status_code == 409

    async def test_empty_update_rejected(self, client):
        embedding_id = await store(client)
        assert (
            await client.put(f"/v1/embeddings/{embedding_id}?collection=biology", json={})
        ).status_code == 422

    async def test_unknown_id_is_404(self, client):
        await store(client)
        response = await client.put(
            "/v1/embeddings/00000000-0000-0000-0000-000000000000?collection=biology",
            json={"topic": "x"},
        )
        assert response.status_code == 404


class TestDelete:
    async def test_deletes_and_is_then_gone(self, client):
        embedding_id = await store(client)
        response = await client.delete(f"/v1/embeddings/{embedding_id}?collection=biology")
        assert response.status_code == 200
        assert response.json() == {
            "id": embedding_id,
            "user_id": "alice",
            "deleted": True,
        }
        assert (
            await client.get(f"/v1/embeddings/{embedding_id}?collection=biology")
        ).status_code == 404

    async def test_deleting_twice_is_404(self, client):
        embedding_id = await store(client)
        await store(client)  # keep the collection non-empty
        await client.delete(f"/v1/embeddings/{embedding_id}?collection=biology")
        assert (
            await client.delete(f"/v1/embeddings/{embedding_id}?collection=biology")
        ).status_code == 404

    async def test_unknown_collection_is_404(self, client):
        embedding_id = await store(client)
        assert (
            await client.delete(f"/v1/embeddings/{embedding_id}?collection=ghost")
        ).status_code == 404
