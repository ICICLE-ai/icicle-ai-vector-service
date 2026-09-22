"""Shared fixtures for the whole suite.

The suite is hermetic: an in-memory Qdrant and a stubbed current-user
dependency, so no Qdrant server, no Tapis token and no network are needed
(beyond a one-time model download for the cross-encoder tests).

Authentication itself is stubbed on purpose. ``core.security`` is covered by its
own unit tests; what the v1 tests exercise is *authorisation* — that given a
correctly identified user, no endpoint will touch another user's data.
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
import pytest_asyncio

# Settings are constructed at import time, so these must exist before any
# src.app import happens.
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("TAPIS_ISSUER", "https://example.tapis.io/v3/tokens")
os.environ.setdefault("TAPIS_JWKS_URL", "https://example.tapis.io/jwks.json")
os.environ.setdefault("TAPIS_TENANT_ID", "icicleai")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from httpx import ASGITransport, AsyncClient  # noqa: E402
from qdrant_client import AsyncQdrantClient  # noqa: E402

from src.app.core.security import UserContext, get_current_user  # noqa: E402
from src.app.db import get_qdrant_client  # noqa: E402
from src.app.main import app  # noqa: E402

ALICE = "alice"
BOB = "bob"


class Client:
    """HTTP client bound to a switchable current user.

    ``client.as_user("bob").get(...)`` makes the next request as bob, which is
    what lets a single test drive two tenants against one store.
    """

    def __init__(self, http: AsyncClient) -> None:
        self._http = http
        self.username = ALICE

    def as_user(self, username: str) -> "Client":
        self.username = username
        return self

    def __getattr__(self, name: str):
        return getattr(self._http, name)


@pytest_asyncio.fixture
async def qdrant() -> AsyncQdrantClient:
    """A fresh in-memory Qdrant per test, so tests cannot leak into each other."""
    client = AsyncQdrantClient(location=":memory:")
    yield client
    await client.close()


@pytest_asyncio.fixture
async def client(qdrant: AsyncQdrantClient) -> Client:
    async def _qdrant():
        yield qdrant

    holder: dict[str, Client] = {}

    async def _user() -> UserContext:
        return UserContext(
            username=holder["client"].username, tenant_id="icicleai", claims={}
        )

    app.dependency_overrides[get_qdrant_client] = _qdrant
    app.dependency_overrides[get_current_user] = _user

    # The real lifespan dials Qdrant and exits the process if it can't connect.
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def _noop(_app):
        yield

    app.router.lifespan_context = _noop

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as http:
        holder["client"] = Client(http)
        yield holder["client"]

    app.dependency_overrides.clear()
    app.router.lifespan_context = original_lifespan


# --- helpers -------------------------------------------------------------


def vector(seed: float = 1.0, dim: int = 4) -> list[float]:
    """A deterministic test vector. ``seed`` varies the first component."""
    return [seed, 0.5, 0.2, 0.1][:dim] + [0.0] * max(0, dim - 4)


async def store(
    client: Client,
    collection: str = "biology",
    topic: str | None = None,
    text: str = "a chunk",
    embedding: list[float] | None = None,
    metadata: dict | None = None,
    expect: int = 201,
) -> str:
    """Store one embedding and return its id."""
    body = {
        "embedding": embedding if embedding is not None else vector(),
        "collection": collection,
        "chunks": [text],
        "embedding_model": "test-model",
    }
    if topic:
        body["topic"] = topic
    if metadata:
        body["metadata"] = metadata

    response = await client.post("/v1/embeddings", json=body)
    assert response.status_code == expect, response.text
    return response.json()["id"] if response.status_code == 201 else ""


@pytest.fixture
def anyio_backend():
    return "asyncio"
