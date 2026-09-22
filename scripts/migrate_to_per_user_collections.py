#!/usr/bin/env python3
"""Migrate shared-name collections to the per-user layout.

Before v1.0.0 every user's points lived in one collection per domain
(``biology``), separated only by a ``user_id`` payload filter. They now live in
one collection *per user* per domain (``alice_9f2a1c4e__biology``), which is
what makes deletion safe and lets each user pick their own vector dimension.

This script reads each legacy collection, splits its points by ``user_id``, and
writes them into the correct per-user collections. It is:

* **Non-destructive.** Legacy collections are never modified or deleted. Verify
  the result, then remove them yourself.
* **Re-runnable.** Points keep their ids, so a second run overwrites rather than
  duplicates. A run interrupted halfway can simply be repeated.
* **Dry by default.** Nothing is written unless you pass ``--apply``.

    python scripts/migrate_to_per_user_collections.py                 # plan only
    python scripts/migrate_to_per_user_collections.py --apply
    python scripts/migrate_to_per_user_collections.py --apply --tenant icicleai

The tenant defaults to ``TAPIS_TENANT_ID``; it must match the tenant those users
authenticate with, because it is part of the namespace hash.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qdrant_client import AsyncQdrantClient  # noqa: E402
from qdrant_client.http.models import (  # noqa: E402
    Distance,
    PointStruct,
    VectorParams,
)

from src.app.core.settings import settings  # noqa: E402
from src.app.db.naming import SEPARATOR, Owner, physical_name  # noqa: E402

SCROLL_PAGE = 256


def is_legacy(name: str) -> bool:
    """A collection created before the per-user layout has no owner prefix."""
    return SEPARATOR not in name


async def read_points(client: AsyncQdrantClient, name: str):
    """Yield every point in a collection, with payload and vector."""
    offset = None
    while True:
        records, offset = await client.scroll(
            collection_name=name,
            limit=SCROLL_PAGE,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        for record in records:
            yield record
        if offset is None:
            break


async def ensure_target(
    client: AsyncQdrantClient, name: str, vector_dim: int
) -> None:
    existing = await client.get_collections()
    if any(col.name == name for col in existing.collections or []):
        return
    await client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
    )


async def migrate(tenant: str, apply: bool) -> int:
    client = AsyncQdrantClient(
        url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=60
    )
    try:
        collections = await client.get_collections()
        legacy = sorted(
            col.name for col in (collections.collections or []) if is_legacy(col.name)
        )
        if not legacy:
            print("No legacy collections found — nothing to migrate.")
            return 0

        print(f"Found {len(legacy)} legacy collection(s): {', '.join(legacy)}")
        print(f"Tenant for namespace hashing: {tenant}")
        print(f"Mode: {'APPLY' if apply else 'DRY RUN (pass --apply to write)'}\n")

        grand_total = 0
        for name in legacy:
            info = await client.get_collection(collection_name=name)
            vector_dim = getattr(info.config.params.vectors, "size", None)
            if vector_dim is None:
                print(f"  {name}: skipped (named vectors are not handled)")
                continue

            by_user: dict[str, list[PointStruct]] = defaultdict(list)
            orphans = 0
            async for record in read_points(client, name):
                payload = record.payload or {}
                username = payload.get("user_id")
                if not username:
                    orphans += 1
                    continue
                by_user[username].append(
                    PointStruct(id=record.id, vector=record.vector, payload=payload)
                )

            print(f"  {name} ({vector_dim}d): {len(by_user)} user(s)")
            if orphans:
                print(f"    !! {orphans} point(s) have no user_id and were left behind")

            for username, points in sorted(by_user.items()):
                target = physical_name(Owner(username, tenant), name)
                print(f"    {username:<24} -> {target}  ({len(points)} points)")
                if not apply:
                    continue
                await ensure_target(client, target, vector_dim)
                for start in range(0, len(points), SCROLL_PAGE):
                    await client.upsert(
                        collection_name=target,
                        points=points[start : start + SCROLL_PAGE],
                        wait=True,
                    )
                grand_total += len(points)

        print()
        if apply:
            print(f"Migrated {grand_total} point(s).")
            print(
                "Legacy collections were left untouched. Verify with "
                "GET /v1/collections as each user, then drop them manually."
            )
        else:
            print("Dry run complete. Re-run with --apply to write.")
        return 0
    finally:
        await client.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true", help="actually write (default is a dry run)"
    )
    parser.add_argument(
        "--tenant",
        default=settings.tapis_tenant_id,
        help="tenant id used in the namespace hash (default: TAPIS_TENANT_ID)",
    )
    args = parser.parse_args()
    return asyncio.run(migrate(args.tenant, args.apply))


if __name__ == "__main__":
    raise SystemExit(main())
