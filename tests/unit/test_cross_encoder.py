"""Unit tests for the cross-encoder wrapper.

The model itself is stubbed, so these run in milliseconds and without the
optional dependency: what is under test is the wrapper's own logic — the model
allowlist, the candidate cap, text selection and result shaping.
"""

import pytest
from fastapi import HTTPException

from src.app.core.settings import settings
from src.app.reranking import cross_encoder


def candidate(id_: str, score: float, text: str | None = None, chunks=None) -> dict:
    return {
        "id": id_,
        "score": score,
        "embedding": [1.0, 0.0],
        "collection": "c",
        "topic": None,
        "text": text,
        "chunks": chunks if chunks is not None else ["chunk text"],
        "metadata": {},
    }


class StubModel:
    """Scores each pair by its position, so ordering is predictable."""

    def __init__(self, scores=None):
        self.calls: list[list[tuple[str, str]]] = []
        self._scores = scores

    def predict(self, pairs, show_progress_bar=False):
        self.calls.append(list(pairs))
        if self._scores is not None:
            return self._scores[: len(pairs)]
        return [float(len(pairs) - i) for i in range(len(pairs))]


@pytest.fixture
def stub(monkeypatch):
    model = StubModel()
    monkeypatch.setattr(cross_encoder, "_load", lambda name: model)
    return model


class TestResolveModel:
    def test_none_yields_the_configured_default(self):
        assert cross_encoder.resolve_model(None) == settings.rerank_model

    def test_allowed_model_passes_through(self):
        allowed = settings.rerank_allowed_models[0]
        assert cross_encoder.resolve_model(allowed) == allowed

    def test_whitespace_is_trimmed(self):
        allowed = settings.rerank_allowed_models[0]
        assert cross_encoder.resolve_model(f"  {allowed}  ") == allowed

    @pytest.mark.parametrize(
        "name",
        [
            "attacker/backdoored-model",
            "../../etc/passwd",
            "BAAI/bge-reranker-large",
            "",
        ],
    )
    def test_disallowed_models_are_refused(self, name):
        with pytest.raises(HTTPException) as excinfo:
            cross_encoder.resolve_model(name)
        assert excinfo.value.status_code == 400
        assert "not allowed" in excinfo.value.detail

    def test_error_lists_the_permitted_models(self):
        """Operators need to know what they *can* pick."""
        with pytest.raises(HTTPException) as excinfo:
            cross_encoder.resolve_model("nope/nope")
        for allowed in settings.rerank_allowed_models:
            assert allowed in excinfo.value.detail


class TestCandidateText:
    def test_prefers_the_text_field(self):
        assert cross_encoder._candidate_text(candidate("a", 1.0, text="the text")) == "the text"

    def test_falls_back_to_joined_chunks(self):
        item = candidate("a", 1.0, chunks=["one", "two"])
        assert cross_encoder._candidate_text(item) == "one two"

    def test_handles_no_text_at_all(self):
        assert cross_encoder._candidate_text(candidate("a", 1.0, chunks=[])) == ""


class TestRerank:
    async def test_empty_candidates_short_circuits(self, stub):
        assert await cross_encoder.cross_encoder_rerank([], "q", 5, "m") == []
        assert stub.calls == []

    async def test_sorts_by_model_score_not_vector_score(self, monkeypatch):
        """The model's judgement must override the incoming vector order."""
        model = StubModel(scores=[0.1, 0.9])
        monkeypatch.setattr(cross_encoder, "_load", lambda name: model)

        result = await cross_encoder.cross_encoder_rerank(
            [candidate("vector_first", 0.99), candidate("truly_relevant", 0.10)],
            "q",
            2,
            "m",
        )
        assert [item["id"] for item in result] == ["truly_relevant", "vector_first"]
        assert result[0]["rerank_score"] == 0.9

    async def test_respects_top_k(self, stub):
        candidates = [candidate(str(i), 0.5) for i in range(10)]
        assert len(await cross_encoder.cross_encoder_rerank(candidates, "q", 3, "m")) == 3

    async def test_strips_embeddings(self, stub):
        result = await cross_encoder.cross_encoder_rerank([candidate("a", 0.5)], "q", 1, "m")
        assert "embedding" not in result[0]

    async def test_pairs_the_query_with_each_candidate(self, stub):
        await cross_encoder.cross_encoder_rerank(
            [candidate("a", 0.5, text="alpha"), candidate("b", 0.5, text="beta")],
            "my query",
            2,
            "m",
        )
        assert stub.calls[0] == [("my query", "alpha"), ("my query", "beta")]

    async def test_caps_candidates_to_protect_the_pod(self, stub, monkeypatch):
        """One user must not be able to occupy the CPU with a huge fetch_k."""
        monkeypatch.setattr(settings, "rerank_max_candidates", 3)
        candidates = [candidate(str(i), 1.0 - i / 100) for i in range(50)]

        await cross_encoder.cross_encoder_rerank(candidates, "q", 50, "m")
        assert len(stub.calls[0]) == 3

    async def test_cap_keeps_the_best_vector_scores(self, stub, monkeypatch):
        """The dropped tail must be the least similar, not an arbitrary slice."""
        monkeypatch.setattr(settings, "rerank_max_candidates", 2)
        candidates = [candidate("best", 0.9), candidate("mid", 0.5), candidate("worst", 0.1)]

        await cross_encoder.cross_encoder_rerank(candidates, "q", 2, "m")
        scored_texts = [pair[1] for pair in stub.calls[0]]
        assert len(scored_texts) == 2


class TestAvailability:
    def test_is_available_matches_the_import(self):
        try:
            import sentence_transformers  # noqa: F401

            expected = True
        except ImportError:
            expected = False
        assert cross_encoder.is_available() is expected

    async def test_preload_is_a_noop_when_disabled(self, monkeypatch):
        monkeypatch.setattr(settings, "rerank_preload", False)
        loaded: list[str] = []
        monkeypatch.setattr(cross_encoder, "_load", lambda name: loaded.append(name))
        await cross_encoder.preload()
        assert loaded == []


class TestDispatch:
    async def test_unknown_method_raises(self):
        from src.app.reranking import rerank

        with pytest.raises(ValueError, match="Unknown rerank method"):
            await rerank("nonsense", [], [1.0], 5)

    async def test_preload_loads_when_enabled_and_available(self, monkeypatch):
        monkeypatch.setattr(settings, "rerank_preload", True)
        monkeypatch.setattr(cross_encoder, "is_available", lambda: True)
        loaded: list[str] = []
        monkeypatch.setattr(cross_encoder, "_load", lambda name: loaded.append(name))

        await cross_encoder.preload()
        assert loaded == [settings.rerank_model]

    async def test_preload_skips_when_dependency_missing(self, monkeypatch):
        monkeypatch.setattr(settings, "rerank_preload", True)
        monkeypatch.setattr(cross_encoder, "is_available", lambda: False)
        loaded: list[str] = []
        monkeypatch.setattr(cross_encoder, "_load", lambda name: loaded.append(name))

        await cross_encoder.preload()
        assert loaded == []

    def test_unavailable_error_is_a_503(self):
        error = cross_encoder.CrossEncoderUnavailable("nope")
        assert error.status_code == 503
