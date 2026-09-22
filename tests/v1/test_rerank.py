"""Reranking through the API.

Cross-encoder tests skip when the ``[rerank]`` extra is absent, which is the
slim-install configuration the service is designed to tolerate.
"""

import pytest

from src.app.reranking import is_available
from tests.conftest import store, vector

pytestmark = pytest.mark.asyncio

requires_model = pytest.mark.skipif(
    not is_available(), reason="[rerank] extra not installed"
)

# The vector scores rank the Eiffel Tower passage first, but only the second
# passage answers the question — so a true reranker must reorder them.
FACTS = [
    ("The Eiffel Tower in Paris is 330 metres tall and was completed in 1889.", 0.99),
    ("Paris is the capital and most populous city of France.", 0.95),
    ("Berlin is the capital of Germany and its largest city.", 0.90),
]
QUERY = "What is the capital of France?"
QUERY_VECTOR = [1.0, 0.0, 0.0, 0.0]


async def seed_facts(client):
    for text, seed in FACTS:
        await store(client, collection="facts", text=text, embedding=[seed, 0.1, 0.0, 0.0])


class TestMethodsEndpoint:
    async def test_lists_all_three_methods(self, client):
        body = (await client.get("/v1/rerank/methods")).json()
        assert {m["name"] for m in body["methods"]} == {
            "mmr",
            "cosine_rescore",
            "cross_encoder",
        }

    async def test_describes_query_text_requirements(self, client):
        by_name = {
            m["name"]: m for m in (await client.get("/v1/rerank/methods")).json()["methods"]
        }
        assert by_name["mmr"]["requires_query_text"] is False
        assert by_name["cosine_rescore"]["requires_query_text"] is False
        assert by_name["cross_encoder"]["requires_query_text"] is True

    async def test_reports_actual_availability(self, client):
        by_name = {
            m["name"]: m for m in (await client.get("/v1/rerank/methods")).json()["methods"]
        }
        assert by_name["mmr"]["available"] is True
        assert by_name["cross_encoder"]["available"] is is_available()

    async def test_default_model_is_in_the_allowlist(self, client):
        body = (await client.get("/v1/rerank/methods")).json()
        assert body["default_model"] in body["allowed_models"]


class TestVectorMethods:
    @pytest.mark.parametrize("method", ["mmr", "cosine_rescore"])
    async def test_returns_both_scores(self, client, method):
        await seed_facts(client)
        body = (
            await client.post(
                "/v1/rerank",
                json={
                    "query_embedding": QUERY_VECTOR,
                    "collection": "facts",
                    "method": method,
                    "top_k": 3,
                    "fetch_k": 10,
                },
            )
        ).json()
        assert body["method"] == method
        assert body["model"] is None
        for item in body["results"]:
            assert "score" in item and "rerank_score" in item
            assert "embedding" not in item

    async def test_cosine_rescore_sorts_descending(self, client):
        await seed_facts(client)
        body = (
            await client.post(
                "/v1/rerank",
                json={
                    "query_embedding": QUERY_VECTOR,
                    "collection": "facts",
                    "method": "cosine_rescore",
                    "top_k": 3,
                    "fetch_k": 10,
                },
            )
        ).json()
        scores = [item["rerank_score"] for item in body["results"]]
        assert scores == sorted(scores, reverse=True)

    async def test_respects_top_k(self, client):
        await seed_facts(client)
        body = (
            await client.post(
                "/v1/rerank",
                json={
                    "query_embedding": QUERY_VECTOR,
                    "collection": "facts",
                    "method": "mmr",
                    "top_k": 2,
                    "fetch_k": 10,
                },
            )
        ).json()
        assert len(body["results"]) == 2
        assert body["top_k"] == 2

    async def test_unknown_collection_is_404(self, client):
        response = await client.post(
            "/v1/rerank", json={"query_embedding": vector(), "collection": "ghost"}
        )
        assert response.status_code == 404


class TestValidation:
    async def test_cross_encoder_without_query_text_is_422(self, client):
        await seed_facts(client)
        response = await client.post(
            "/v1/rerank",
            json={
                "query_embedding": QUERY_VECTOR,
                "collection": "facts",
                "method": "cross_encoder",
            },
        )
        assert response.status_code == 422

    async def test_model_outside_the_allowlist_is_400(self, client):
        await seed_facts(client)
        response = await client.post(
            "/v1/rerank",
            json={
                "query_embedding": QUERY_VECTOR,
                "query_text": QUERY,
                "collection": "facts",
                "method": "cross_encoder",
                "rerank_model": "attacker/backdoored-model",
            },
        )
        assert response.status_code == 400
        assert "not allowed" in response.json()["detail"]

    async def test_bad_model_is_rejected_before_the_search_runs(self, client):
        """Fail fast: a bad model name must not cost a Qdrant query first."""
        response = await client.post(
            "/v1/rerank",
            json={
                "query_embedding": QUERY_VECTOR,
                "query_text": QUERY,
                "collection": "never-created",
                "method": "cross_encoder",
                "rerank_model": "attacker/model",
            },
        )
        # 400 for the model, not 404 for the missing collection.
        assert response.status_code == 400

    async def test_unknown_method_is_422(self, client):
        response = await client.post(
            "/v1/rerank",
            json={
                "query_embedding": QUERY_VECTOR,
                "collection": "facts",
                "method": "magic",
            },
        )
        assert response.status_code == 422


@requires_model
class TestCrossEncoder:
    async def test_promotes_the_passage_that_answers_the_query(self, client):
        await seed_facts(client)

        baseline = (
            await client.post(
                "/v1/retrieve",
                json={"query_embedding": QUERY_VECTOR, "collection": "facts", "top_k": 3},
            )
        ).json()
        assert baseline["results"][0]["chunks"][0].startswith("The Eiffel Tower")

        body = (
            await client.post(
                "/v1/rerank",
                json={
                    "query_embedding": QUERY_VECTOR,
                    "query_text": QUERY,
                    "collection": "facts",
                    "method": "cross_encoder",
                    "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    "top_k": 3,
                    "fetch_k": 10,
                },
            )
        ).json()
        assert body["results"][0]["chunks"][0].startswith("Paris is the capital")
        assert body["model"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    async def test_results_sorted_by_rerank_score(self, client):
        await seed_facts(client)
        body = (
            await client.post(
                "/v1/rerank",
                json={
                    "query_embedding": QUERY_VECTOR,
                    "query_text": QUERY,
                    "collection": "facts",
                    "method": "cross_encoder",
                    "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    "top_k": 3,
                    "fetch_k": 10,
                },
            )
        ).json()
        scores = [item["rerank_score"] for item in body["results"]]
        assert scores == sorted(scores, reverse=True)
        assert all("embedding" not in item for item in body["results"])

    async def test_defaults_to_the_configured_model(self, client):
        from src.app.core.settings import settings

        await seed_facts(client)
        body = (
            await client.post(
                "/v1/rerank",
                json={
                    "query_embedding": QUERY_VECTOR,
                    "query_text": QUERY,
                    "collection": "facts",
                    "method": "cross_encoder",
                    "top_k": 2,
                },
            )
        ).json()
        assert body["model"] == settings.rerank_model
