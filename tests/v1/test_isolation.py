"""Cross-tenant isolation: one user must never affect another.

Collections are physically shared between users, so isolation rests entirely on
the ``user_id`` filter in ``app.db.filters``. This module is the executable
statement of that guarantee, organised by the kind of damage being ruled out:

* **Confidentiality** — B's data never appears in A's responses.
* **Integrity**       — A cannot modify B's data.
* **Availability**    — A cannot delete or destroy B's data.
* **Metadata**        — A cannot even learn what B has.

Several tests assert a *positive* control alongside the negative one (e.g. "1 of
2 ids deleted") so that a filter which silently matched nothing would fail the
test rather than pass it.
"""

import pytest

from tests.conftest import ALICE, BOB, store, vector

pytestmark = pytest.mark.asyncio

ENDPOINTS_REQUIRING_OWNERSHIP = ["get", "put", "delete"]


# --- Confidentiality -----------------------------------------------------


async def test_retrieve_never_returns_another_users_points(client):
    await store(client.as_user(ALICE), text="alice private note")
    await store(client.as_user(BOB), text="bob private note")

    response = await client.as_user(ALICE).post(
        "/v1/retrieve",
        json={"query_embedding": vector(), "collection": "biology", "top_k": 100},
    )
    results = response.json()["results"]
    assert len(results) == 1
    assert "bob" not in results[0]["chunks"][0]


async def test_rerank_never_returns_another_users_points(client):
    """Rerank fetches its own candidates, so it needs its own proof."""
    await store(client.as_user(ALICE), text="alice note")
    await store(client.as_user(BOB), text="bob note")

    for method in ("mmr", "cosine_rescore"):
        response = await client.as_user(ALICE).post(
            "/v1/rerank",
            json={
                "query_embedding": vector(),
                "collection": "biology",
                "method": method,
                "top_k": 100,
                "fetch_k": 100,
            },
        )
        results = response.json()["results"]
        assert len(results) == 1, method
        assert "bob" not in results[0]["chunks"][0], method


async def test_metadata_filter_cannot_escape_the_user_scope(client):
    """A crafted filter key must not be able to address a top-level field.

    Metadata keys are namespaced under ``metadata.``, so asking for
    ``user_id: bob`` looks for ``metadata.user_id`` and simply matches nothing.
    """
    await store(client.as_user(BOB), text="bob note")
    await store(client.as_user(ALICE), text="alice note")

    response = await client.as_user(ALICE).post(
        "/v1/retrieve",
        json={
            "query_embedding": vector(),
            "collection": "biology",
            "filter": {"conditions": {"user_id": BOB}},
        },
    )
    assert response.status_code == 200
    assert response.json()["results"] == []


async def test_listing_embeddings_is_scoped(client):
    await store(client.as_user(ALICE))
    await store(client.as_user(BOB))
    await store(client.as_user(BOB))

    body = (
        await client.as_user(ALICE).get("/v1/collections/biology/embeddings")
    ).json()
    assert body["count"] == 1
    assert all(e["user_id"] == ALICE for e in body["embeddings"])


# --- Integrity -----------------------------------------------------------


async def test_cannot_read_update_or_delete_another_users_embedding(client):
    bob_id = await store(client.as_user(BOB))
    alice = client.as_user(ALICE)

    assert (
        await alice.get(f"/v1/embeddings/{bob_id}?collection=biology")
    ).status_code == 404
    assert (
        await alice.put(
            f"/v1/embeddings/{bob_id}?collection=biology", json={"topic": "hijacked"}
        )
    ).status_code == 404
    assert (
        await alice.delete(f"/v1/embeddings/{bob_id}?collection=biology")
    ).status_code == 404

    # Bob's point is untouched and unmodified.
    still = await client.as_user(BOB).get(f"/v1/embeddings/{bob_id}?collection=biology")
    assert still.status_code == 200
    assert still.json()["topic"] is None
    assert still.json()["user_id"] == BOB


async def test_storing_cannot_forge_ownership(client):
    """A user_id in the request body is ignored; the token decides ownership."""
    response = await client.as_user(ALICE).post(
        "/v1/embeddings",
        json={
            "embedding": vector(),
            "collection": "biology",
            "chunks": ["x"],
            "embedding_model": "test-model",
            "user_id": BOB,
        },
    )
    assert response.json()["user_id"] == ALICE

    # And it is really stored as alice's, not merely reported that way.
    assert (await client.as_user(BOB).get("/v1/collections")).json()["count"] == 0


async def test_update_cannot_reassign_ownership(client):
    """Rewriting the payload must not let a user hand their point to someone else."""
    alice_id = await store(client.as_user(ALICE))

    await client.as_user(ALICE).put(
        f"/v1/embeddings/{alice_id}?collection=biology",
        json={"topic": "still-mine", "user_id": BOB},
    )

    assert (
        await client.as_user(BOB).get(f"/v1/embeddings/{alice_id}?collection=biology")
    ).status_code == 404
    mine = await client.as_user(ALICE).get(
        f"/v1/embeddings/{alice_id}?collection=biology"
    )
    assert mine.json()["user_id"] == ALICE


# --- Availability --------------------------------------------------------


async def test_bulk_delete_by_id_cannot_touch_another_users_points(client):
    # Alice needs her own point here, otherwise the request 404s for not owning
    # anything in the collection and never reaches the id filter under test.
    await store(client.as_user(ALICE))
    bob_id = await store(client.as_user(BOB))

    response = await client.as_user(ALICE).post(
        "/v1/embeddings/bulk-delete", json={"collection": "biology", "ids": [bob_id]}
    )
    assert response.status_code == 200
    assert response.json()["deleted"] == 0

    assert (
        await client.as_user(BOB).get(f"/v1/embeddings/{bob_id}?collection=biology")
    ).status_code == 200


async def test_bulk_delete_mixed_batch_deletes_only_own(client):
    """Positive control: exactly 1 of 2 ids must go, proving the filter is live."""
    alice_id = await store(client.as_user(ALICE))
    bob_id = await store(client.as_user(BOB))

    response = await client.as_user(ALICE).post(
        "/v1/embeddings/bulk-delete",
        json={"collection": "biology", "ids": [alice_id, bob_id]},
    )
    assert response.json()["deleted"] == 1

    assert (
        await client.as_user(BOB).get(f"/v1/embeddings/{bob_id}?collection=biology")
    ).status_code == 200
    assert (
        await client.as_user(ALICE).get(f"/v1/embeddings/{alice_id}?collection=biology")
    ).status_code == 404


async def test_bulk_delete_all_spares_other_users(client):
    await store(client.as_user(ALICE))
    await store(client.as_user(ALICE))
    bob_id = await store(client.as_user(BOB))

    response = await client.as_user(ALICE).post(
        "/v1/embeddings/bulk-delete", json={"collection": "biology", "all": True}
    )
    assert response.json()["deleted"] == 2

    assert (
        await client.as_user(BOB).get(f"/v1/embeddings/{bob_id}?collection=biology")
    ).status_code == 200


async def test_bulk_delete_by_predicate_spares_other_users(client):
    await store(client.as_user(ALICE), topic="plant")
    bob_id = await store(client.as_user(BOB), topic="plant")

    response = await client.as_user(ALICE).post(
        "/v1/embeddings/bulk-delete", json={"collection": "biology", "topic": "plant"}
    )
    assert response.json()["deleted"] == 1

    assert (
        await client.as_user(BOB).get(f"/v1/embeddings/{bob_id}?collection=biology")
    ).status_code == 200


async def test_deleting_a_collection_spares_other_users(client):
    await store(client.as_user(ALICE))
    bob_id = await store(client.as_user(BOB))

    response = await client.as_user(ALICE).delete("/v1/collections/biology")
    assert response.json()["deleted"] == 1

    assert (
        await client.as_user(BOB).get(f"/v1/embeddings/{bob_id}?collection=biology")
    ).status_code == 200


async def test_purge_spares_other_users(client):
    await store(client.as_user(ALICE), collection="biology")
    await store(client.as_user(ALICE), collection="chemistry")
    bob_id = await store(client.as_user(BOB), collection="biology")

    response = await client.as_user(ALICE).delete("/v1/collections?confirm=true")
    assert response.json()["deleted"] == 2

    bob_view = await client.as_user(BOB).get("/v1/collections")
    assert bob_view.json()["count"] == 1
    assert (
        await client.as_user(BOB).get(f"/v1/embeddings/{bob_id}?collection=biology")
    ).status_code == 200


async def test_purge_cannot_reach_another_users_collections(client):
    """Purge drops collections outright — it must only ever see the caller's own."""
    await store(client.as_user(ALICE), collection="biology")
    bob_id = await store(client.as_user(BOB), collection="biology")

    body = (await client.as_user(ALICE).delete("/v1/collections?confirm=true")).json()
    assert body["collections_dropped"] == ["biology"]  # alice's own, not bob's

    # Bob's identically-named collection is untouched.
    assert (
        await client.as_user(BOB).get(f"/v1/embeddings/{bob_id}?collection=biology")
    ).status_code == 200
    assert (await client.as_user(BOB).get("/v1/collections")).json()["count"] == 1


async def test_dropping_a_collection_cannot_reach_the_same_name_elsewhere(client):
    """The case that used to be a race: two users, one collection name.

    Because each user's 'biology' is a physically separate Qdrant collection,
    dropping one cannot destroy the other, with no ordering or timing caveat.
    """
    await store(client.as_user(ALICE), collection="biology")
    bob_id = await store(client.as_user(BOB), collection="biology")

    body = (await client.as_user(ALICE).delete("/v1/collections/biology")).json()
    assert body["collection_dropped"] is True

    assert (
        await client.as_user(BOB).get(f"/v1/embeddings/{bob_id}?collection=biology")
    ).status_code == 200


async def test_users_may_hold_the_same_collection_name_at_different_dimensions(client):
    """Different domains bring different embedding models, hence different widths.

    Under a shared collection the second store would fail with a 409; per-user
    collections make the two independent.
    """
    await store(client.as_user(ALICE), collection="biology", embedding=[1.0, 0.0, 0.0, 0.0])

    response = await client.as_user(BOB).post(
        "/v1/embeddings",
        json={
            "embedding": [1.0, 0.0],  # a narrower model
            "collection": "biology",
            "chunks": ["bob's domain"],
            "embedding_model": "other-model",
        },
    )
    assert response.status_code == 201
    assert response.json()["vector_dim"] == 2

    assert (await client.as_user(ALICE).get("/v1/collections/biology")).json()[
        "vector_dim"
    ] == 4
    assert (await client.as_user(BOB).get("/v1/collections/biology")).json()[
        "vector_dim"
    ] == 2


# --- Metadata ------------------------------------------------------------


async def test_collection_listing_hides_other_users_collections(client):
    await store(client.as_user(ALICE), collection="alice-work")
    await store(client.as_user(BOB), collection="bob-secret")

    names = {
        c["collection"]
        for c in (await client.as_user(ALICE).get("/v1/collections")).json()["collections"]
    }
    assert names == {"alice_work"}


async def test_collection_stats_are_scoped(client):
    """Counts, topics and models must describe only the caller's own data."""
    for _ in range(3):
        await store(client.as_user(ALICE), topic="plant")
    await store(client.as_user(BOB), topic="bob-only-topic")

    stats = (await client.as_user(ALICE).get("/v1/collections/biology")).json()
    assert stats["points"] == 3
    assert stats["topics"] == ["plant"]
    assert "bob-only-topic" not in stats["topics"]


async def test_cannot_probe_a_collection_owned_only_by_another_user(client):
    """A collection you have no data in is a 404, not an empty 200.

    Otherwise the status code alone would reveal that the collection exists.
    """
    await store(client.as_user(BOB), collection="bob-secret")

    alice = client.as_user(ALICE)
    assert (await alice.get("/v1/collections/bob-secret")).status_code == 404
    assert (await alice.delete("/v1/collections/bob-secret")).status_code == 404


async def test_embedding_listing_does_not_leak_collection_existence(client):
    """Probing the listing endpoint must not distinguish "not yours" from "no such thing".

    Regression test: this endpoint used to answer 200 with an empty page for a
    collection owned entirely by someone else, while the sibling stats endpoint
    answered 404 — so comparing the two revealed other users' collection names.
    """
    await store(client.as_user(BOB), collection="bob-secret")
    alice = client.as_user(ALICE)

    real = await alice.get("/v1/collections/bob-secret/embeddings")
    imaginary = await alice.get("/v1/collections/does-not-exist-at-all/embeddings")

    assert real.status_code == imaginary.status_code == 404
    # Same message shape too, with only the caller's own input echoed back.
    assert real.json()["detail"].replace("bob-secret", "X") == imaginary.json()[
        "detail"
    ].replace("does-not-exist-at-all", "X")


async def test_bulk_delete_does_not_leak_collection_existence(client):
    """Same probe, via the bulk-delete endpoint."""
    await store(client.as_user(BOB), collection="bob-secret")
    alice = client.as_user(ALICE)

    real = await alice.post(
        "/v1/embeddings/bulk-delete", json={"collection": "bob-secret", "all": True}
    )
    imaginary = await alice.post(
        "/v1/embeddings/bulk-delete", json={"collection": "no-such-thing", "all": True}
    )
    # Both must be indistinguishable to the caller.
    assert real.status_code == imaginary.status_code == 404
