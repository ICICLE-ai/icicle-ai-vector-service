"""The shared Qdrant client."""

from functools import lru_cache
from typing import AsyncGenerator

from qdrant_client import AsyncQdrantClient

from ..core.settings import settings


@lru_cache(maxsize=1)
def get_shared_client() -> AsyncQdrantClient:
    """One client per process, so the HTTP connection pool stays warm."""
    return AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        prefer_grpc=False,
        timeout=30,
    )


async def get_qdrant_client() -> AsyncGenerator[AsyncQdrantClient, None]:
    """FastAPI dependency. Overridden in tests to inject an in-memory client."""
    yield get_shared_client()


async def close_client() -> None:
    """Close the shared client and drop it from the cache (used at shutdown)."""
    if get_shared_client.cache_info().currsize:
        await get_shared_client().close()
        get_shared_client.cache_clear()
