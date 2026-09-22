"""Behaviour that changes with configuration.

Chiefly DROP_EMPTY_COLLECTIONS. Collections are per user, so dropping one is
always safe; the setting exists for operators who would rather keep an emptied
collection around — for instance to preserve its vector dimension across a full
delete and re-ingest.
"""

import pytest

from src.app.core.settings import settings
from tests.conftest import ALICE, BOB, store

pytestmark = pytest.mark.asyncio


@pytest.fixture
def keep_empty(monkeypatch):
    monkeypatch.setattr(settings, "drop_empty_collections", False)
    yield


async def test_dropping_is_the_default(client):
    """Per-user collections make this safe, so it is on."""
    assert settings.drop_empty_collections is True


async def test_emptied_collection_is_removed_by_default(client):
    await store(client)
    assert (await client.delete("/v1/collections/biology")).json()[
        "collection_dropped"
    ] is True
    assert (await client.get("/v1/collections")).json()["count"] == 0


async def test_collection_is_kept_when_dropping_is_disabled(client, keep_empty):
    await store(client)
    body = (await client.delete("/v1/collections/biology")).json()
    assert body["deleted"] == 1
    assert body["collection_dropped"] is False

    # Kept, and therefore still holding its original dimension.
    assert (await client.get("/v1/collections/biology")).json()["vector_dim"] == 4


async def test_keeping_collections_preserves_the_dimension_on_re_ingest(client, keep_empty):
    """The reason an operator might turn dropping off."""
    await store(client, embedding=[1.0, 2.0, 3.0, 4.0])
    await client.delete("/v1/collections/biology")

    # A re-ingest at a different width is now refused rather than silently
    # redefining the collection.
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


async def test_purge_drops_every_collection_the_caller_owns(client):
    await store(client, collection="one")
    await store(client, collection="two")

    body = (await client.delete("/v1/collections?confirm=true")).json()
    assert sorted(body["collections_dropped"]) == ["one", "two"]
    assert (await client.get("/v1/collections")).json()["count"] == 0


async def test_purge_keeps_collections_when_dropping_is_disabled(client, keep_empty):
    await store(client, collection="one")

    body = (await client.delete("/v1/collections?confirm=true")).json()
    assert body["deleted"] == 1
    assert body["collections_dropped"] == []
    assert body["collections_affected"] == ["one"]


async def test_dropping_never_crosses_users_either_way(client, keep_empty):
    """Whatever the setting, another user's identically-named collection survives."""
    await store(client.as_user(ALICE), collection="biology")
    bob_id = await store(client.as_user(BOB), collection="biology")

    await client.as_user(ALICE).delete("/v1/collections/biology")
    assert (
        await client.as_user(BOB).get(f"/v1/embeddings/{bob_id}?collection=biology")
    ).status_code == 200
