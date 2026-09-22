"""Unit tests for the vector-space rerankers."""

import pytest

from src.app.reranking.vector import cosine_rescore, cosine_sim, mmr_rerank


def candidate(id_: str, score: float, embedding: list[float], text: str = "t") -> dict:
    return {
        "id": id_,
        "score": score,
        "embedding": embedding,
        "collection": "c",
        "topic": None,
        "text": text,
        "chunks": [text],
        "metadata": {},
    }


class TestCosineSim:
    def test_identical_vectors_score_one(self):
        assert cosine_sim([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self):
        assert cosine_sim([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors_score_minus_one(self):
        assert cosine_sim([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_magnitude_is_ignored(self):
        assert cosine_sim([1.0, 0.0], [5.0, 0.0]) == pytest.approx(1.0)

    def test_zero_vector_scores_zero_rather_than_dividing_by_zero(self):
        assert cosine_sim([0.0, 0.0], [1.0, 0.0]) == 0.0
        assert cosine_sim([1.0, 0.0], [0.0, 0.0]) == 0.0


class TestMmr:
    def test_empty_input(self):
        assert mmr_rerank([], [1.0, 0.0], 5, 0.7) == []

    def test_lambda_one_preserves_relevance_order(self):
        """Pure relevance must reproduce the incoming vector ranking."""
        candidates = [
            candidate("a", 0.9, [1.0, 0.0]),
            candidate("b", 0.8, [1.0, 0.0]),
            candidate("c", 0.7, [0.0, 1.0]),
        ]
        result = mmr_rerank(candidates, [1.0, 0.0], 3, lambda_=1.0)
        assert [item["id"] for item in result] == ["a", "b", "c"]

    def test_lambda_zero_prefers_diversity_over_relevance(self):
        """With pure diversity, the near-duplicate of the first pick is demoted."""
        candidates = [
            candidate("a", 0.9, [1.0, 0.0]),
            candidate("duplicate", 0.85, [1.0, 0.0]),
            candidate("different", 0.5, [0.0, 1.0]),
        ]
        result = mmr_rerank(candidates, [1.0, 0.0], 2, lambda_=0.0)
        assert [item["id"] for item in result] == ["a", "different"]

    def test_respects_top_k(self):
        candidates = [candidate(str(i), 0.9 - i / 10, [1.0, float(i)]) for i in range(5)]
        assert len(mmr_rerank(candidates, [1.0, 0.0], 2, 0.7)) == 2

    def test_top_k_larger_than_input_returns_everything(self):
        candidates = [candidate("a", 0.9, [1.0, 0.0])]
        assert len(mmr_rerank(candidates, [1.0, 0.0], 50, 0.7)) == 1

    def test_embeddings_are_stripped_from_results(self):
        """Vectors must never travel back to the client."""
        candidates = [candidate("a", 0.9, [1.0, 0.0])]
        assert "embedding" not in mmr_rerank(candidates, [1.0, 0.0], 1, 0.7)[0]

    def test_every_result_carries_a_rerank_score(self):
        candidates = [candidate("a", 0.9, [1.0, 0.0]), candidate("b", 0.8, [0.0, 1.0])]
        assert all("rerank_score" in item for item in mmr_rerank(candidates, [1.0, 0.0], 2, 0.7))


class TestCosineRescore:
    def test_empty_input(self):
        assert cosine_rescore([], [1.0, 0.0], 5) == []

    def test_reorders_by_exact_similarity(self):
        """The incoming order is wrong; exact rescoring must fix it."""
        candidates = [
            candidate("approx_first", 0.99, [0.0, 1.0]),
            candidate("actually_closest", 0.98, [1.0, 0.0]),
        ]
        result = cosine_rescore(candidates, [1.0, 0.0], 2)
        assert [item["id"] for item in result] == ["actually_closest", "approx_first"]

    def test_is_sorted_descending(self):
        candidates = [
            candidate("a", 0.5, [0.2, 1.0]),
            candidate("b", 0.5, [1.0, 0.0]),
            candidate("c", 0.5, [0.7, 0.7]),
        ]
        scores = [i["rerank_score"] for i in cosine_rescore(candidates, [1.0, 0.0], 3)]
        assert scores == sorted(scores, reverse=True)

    def test_respects_top_k(self):
        candidates = [candidate(str(i), 0.9, [1.0, float(i)]) for i in range(5)]
        assert len(cosine_rescore(candidates, [1.0, 0.0], 2)) == 2

    def test_embeddings_are_stripped_from_results(self):
        candidates = [candidate("a", 0.9, [1.0, 0.0])]
        assert "embedding" not in cosine_rescore(candidates, [1.0, 0.0], 1)[0]

    def test_missing_embedding_falls_back_to_the_stored_score(self):
        candidates = [candidate("a", 0.42, [])]
        assert cosine_rescore(candidates, [1.0, 0.0], 1)[0]["rerank_score"] == 0.42
