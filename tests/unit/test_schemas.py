"""Unit tests for request validation.

These are the checks that stop malformed or ambiguous requests before they ever
reach Qdrant.
"""

import pytest
from pydantic import ValidationError

from src.app.schemas import (
    BulkDeleteRequest,
    EmbeddingCreate,
    EmbeddingUpdate,
    RerankRequest,
    RetrieveRequest,
)


class TestEmbeddingCreate:
    def _valid(self, **overrides):
        body = {
            "embedding": [1.0, 2.0],
            "collection": "biology",
            "chunks": ["text"],
            "embedding_model": "m",
        }
        body.update(overrides)
        return body

    def test_minimal_valid_payload(self):
        model = EmbeddingCreate(**self._valid())
        assert model.collection == "biology"
        assert model.topic is None

    def test_user_id_in_body_is_ignored(self):
        """Ownership comes from the token; an extra key must not become a field."""
        model = EmbeddingCreate(**self._valid(user_id="bob"))
        assert not hasattr(model, "user_id")

    @pytest.mark.parametrize("bad", ["", "   "])
    def test_blank_collection_rejected(self, bad):
        with pytest.raises(ValidationError, match="collection"):
            EmbeddingCreate(**self._valid(collection=bad))

    def test_collection_is_trimmed(self):
        assert EmbeddingCreate(**self._valid(collection="  biology  ")).collection == "biology"

    def test_empty_embedding_rejected(self):
        with pytest.raises(ValidationError, match="embedding"):
            EmbeddingCreate(**self._valid(embedding=[]))

    def test_missing_embedding_model_rejected(self):
        body = self._valid()
        del body["embedding_model"]
        with pytest.raises(ValidationError):
            EmbeddingCreate(**body)

    @pytest.mark.parametrize("bad", [[], ["   "], ["ok", ""]])
    def test_blank_chunks_rejected(self, bad):
        with pytest.raises(ValidationError, match="chunks"):
            EmbeddingCreate(**self._valid(chunks=bad))

    def test_blank_topic_rejected_when_given(self):
        with pytest.raises(ValidationError, match="topic"):
            EmbeddingCreate(**self._valid(topic="  "))


class TestEmbeddingUpdate:
    def test_requires_at_least_one_field(self):
        with pytest.raises(ValidationError, match="At least one field"):
            EmbeddingUpdate()

    def test_single_field_is_enough(self):
        assert EmbeddingUpdate(topic="plant").topic == "plant"

    def test_blank_chunks_rejected(self):
        with pytest.raises(ValidationError, match="chunks"):
            EmbeddingUpdate(chunks=["  "])


class TestRetrieveRequest:
    def test_defaults(self):
        request = RetrieveRequest(query_embedding=[1.0], collection="c")
        assert request.top_k == 10

    @pytest.mark.parametrize("top_k", [0, 101, -1])
    def test_top_k_bounds(self, top_k):
        with pytest.raises(ValidationError):
            RetrieveRequest(query_embedding=[1.0], collection="c", top_k=top_k)

    def test_collection_required(self):
        with pytest.raises(ValidationError):
            RetrieveRequest(query_embedding=[1.0], collection="  ")


class TestRerankRequest:
    def _valid(self, **overrides):
        body = {"query_embedding": [1.0], "collection": "c"}
        body.update(overrides)
        return body

    def test_defaults_to_mmr(self):
        assert RerankRequest(**self._valid()).method == "mmr"

    def test_lambda_uses_its_alias(self):
        assert RerankRequest(**self._valid(**{"lambda": 0.2})).lambda_ == 0.2

    @pytest.mark.parametrize("value", [-0.1, 1.1])
    def test_lambda_bounds(self, value):
        with pytest.raises(ValidationError):
            RerankRequest(**self._valid(**{"lambda": value}))

    def test_unknown_method_rejected(self):
        with pytest.raises(ValidationError):
            RerankRequest(**self._valid(method="magic"))

    @pytest.mark.parametrize("query_text", [None, "", "   "])
    def test_cross_encoder_requires_query_text(self, query_text):
        with pytest.raises(ValidationError, match="query_text"):
            RerankRequest(**self._valid(method="cross_encoder", query_text=query_text))

    def test_cross_encoder_accepts_query_text(self):
        request = RerankRequest(**self._valid(method="cross_encoder", query_text="hi"))
        assert request.query_text == "hi"

    def test_vector_methods_do_not_need_query_text(self):
        for method in ("mmr", "cosine_rescore"):
            assert RerankRequest(**self._valid(method=method)).query_text is None

    @pytest.mark.parametrize("fetch_k", [0, 501])
    def test_fetch_k_bounds(self, fetch_k):
        with pytest.raises(ValidationError):
            RerankRequest(**self._valid(fetch_k=fetch_k))


class TestBulkDeleteRequest:
    def test_ids_selector(self):
        assert BulkDeleteRequest(collection="c", ids=["a"]).ids == ["a"]

    def test_topic_selector(self):
        assert BulkDeleteRequest(collection="c", topic="t").topic == "t"

    def test_all_selector(self):
        assert BulkDeleteRequest(collection="c", all=True).all is True

    def test_no_selector_rejected(self):
        with pytest.raises(ValidationError, match="Nothing selected"):
            BulkDeleteRequest(collection="c")

    def test_ids_plus_predicate_rejected(self):
        with pytest.raises(ValidationError, match="not both"):
            BulkDeleteRequest(collection="c", ids=["a"], topic="t")

    @pytest.mark.parametrize("extra", [{"ids": ["a"]}, {"topic": "t"}])
    def test_all_combined_with_anything_rejected(self, extra):
        with pytest.raises(ValidationError, match="do not combine"):
            BulkDeleteRequest(collection="c", all=True, **extra)

    def test_empty_ids_list_rejected(self):
        with pytest.raises(ValidationError):
            BulkDeleteRequest(collection="c", ids=[])

    def test_empty_metadata_filter_is_not_a_selector(self):
        """`filter: {conditions: {}}` selects everything — must not be allowed."""
        with pytest.raises(ValidationError, match="Nothing selected"):
            BulkDeleteRequest(collection="c", filter={"conditions": {}})
