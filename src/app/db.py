from functools import lru_cache
from typing import AsyncGenerator

from qdrant_client import AsyncQdrantClient

from .settings import settings


@lru_cache(maxsize=1)
def _client() -> AsyncQdrantClient:
    # Single shared client keeps the HTTP connection pool warm for FastAPI.
    return AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        prefer_grpc=False,
        timeout=30,
    )


async def get_qdrant_client() -> AsyncGenerator[AsyncQdrantClient, None]:
    yield _client()
