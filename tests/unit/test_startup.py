"""Startup, shutdown and the shared client."""

import pytest
from qdrant_client import AsyncQdrantClient

from src.app import main
from src.app.core.settings import settings
from src.app.db import client as db_client


class TestSharedClient:
    def test_is_cached(self, monkeypatch):
        db_client.get_shared_client.cache_clear()
        assert db_client.get_shared_client() is db_client.get_shared_client()
        db_client.get_shared_client.cache_clear()

    async def test_dependency_yields_the_shared_client(self):
        db_client.get_shared_client.cache_clear()
        async for yielded in db_client.get_qdrant_client():
            assert yielded is db_client.get_shared_client()
        db_client.get_shared_client.cache_clear()

    async def test_close_is_safe_when_never_opened(self):
        db_client.get_shared_client.cache_clear()
        await db_client.close_client()  # must not raise


class TestLifespan:
    async def test_startup_verifies_qdrant_and_indexes(self, monkeypatch):
        memory = AsyncQdrantClient(location=":memory:")
        monkeypatch.setattr(main, "get_shared_client", lambda: memory)
        monkeypatch.setattr(main, "close_client", _noop)

        async with main.lifespan(main.app):
            pass  # startup + shutdown must both complete

        await memory.close()

    async def test_startup_aborts_when_qdrant_is_unreachable(self, monkeypatch):
        """A pod that cannot reach Qdrant must fail fast, not serve errors."""

        class Dead:
            async def get_collections(self):
                raise ConnectionError("refused")

        monkeypatch.setattr(main, "get_shared_client", lambda: Dead())

        with pytest.raises(SystemExit) as excinfo:
            async with main.lifespan(main.app):
                pass
        assert "not reachable" in str(excinfo.value)

    async def test_index_backfill_failure_does_not_block_startup(self, monkeypatch):
        """Index creation is an optimisation; it must not take the service down."""
        memory = AsyncQdrantClient(location=":memory:")
        monkeypatch.setattr(main, "get_shared_client", lambda: memory)
        monkeypatch.setattr(main, "close_client", _noop)

        async def _boom(_client):
            raise RuntimeError("index backend unavailable")

        monkeypatch.setattr(main, "backfill_payload_indexes", _boom)

        async with main.lifespan(main.app):
            pass

        await memory.close()

    async def test_preload_runs_when_enabled(self, monkeypatch):
        memory = AsyncQdrantClient(location=":memory:")
        monkeypatch.setattr(main, "get_shared_client", lambda: memory)
        monkeypatch.setattr(main, "close_client", _noop)
        monkeypatch.setattr(settings, "rerank_preload", True)

        called: list[bool] = []

        async def _preload():
            called.append(True)

        monkeypatch.setattr(main.reranking, "is_available", lambda: True)
        monkeypatch.setattr(main.reranking, "preload", _preload)

        async with main.lifespan(main.app):
            pass

        assert called == [True]
        await memory.close()


async def _noop():
    return None
