"""Application entrypoint.

Wiring only — settings, CORS, lifespan and the versioned router. Endpoint logic
lives in :mod:`app.api.v1`, data access in :mod:`app.db.repository`.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import AsyncQdrantClient

from . import __version__, reranking
from .api.v1 import router as v1_router
from .core.settings import settings
from .db import close_client, get_qdrant_client, get_shared_client
from .db.repository import backfill_payload_indexes
from .schemas import HealthResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

DESCRIPTION = """\
Vector storage and retrieval service for the ICICLE AI tenant. Clients provide
their own pre-computed embeddings; the service handles storage, filtered search
and reranking.

**Collections** are broad domains (e.g. `biology`, `chemistry`), each backed by
its own Qdrant collection and HNSW index. **Topics** are optional sub-categories
within a collection (e.g. `human`, `plant`).

**User isolation.** Every request is authenticated with a Tapis access token via
the `X-Tapis-Token` header, and `user_id` is taken from the token's
`tapis/username` claim — never from the request body. Collections are physically
shared, but every read, write, search and delete is filtered by `user_id`, so
users never see or affect each other's data.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Connecting to Qdrant at %s ...", settings.qdrant_url)
    try:
        client = get_shared_client()
        collections = await client.get_collections()
        logger.info(
            "Qdrant is reachable (%d collections found)",
            len(collections.collections or []),
        )
    except Exception as exc:
        logger.error("Failed to connect to Qdrant at %s: %s", settings.qdrant_url, exc)
        raise SystemExit(
            f"Qdrant is not reachable at {settings.qdrant_url}. "
            "Check QDRANT_URL and QDRANT_API_KEY in your .env file."
        ) from exc

    # Collections created before payload indexing existed have no index on
    # user_id/topic, which turns every filtered search into a full scan.
    # Creating an existing index is a no-op, so this just converges on startup.
    try:
        processed = await backfill_payload_indexes(client)
        logger.info("Tenancy indexes verified on %d collection(s)", processed)
    except Exception as exc:
        logger.warning("Could not verify payload indexes (continuing): %s", exc)

    if reranking.is_available():
        logger.info(
            "Cross-encoder reranking available (default model: %s, preload: %s)",
            settings.rerank_model,
            settings.rerank_preload,
        )
        await reranking.preload()
    else:
        logger.info(
            "Cross-encoder reranking unavailable (sentence-transformers not installed); "
            "'mmr' and 'cosine_rescore' still work"
        )

    yield

    await close_client()


app = FastAPI(
    title="ICICLE AI Vector Service",
    version=__version__,
    description=DESCRIPTION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["X-Tapis-Token", "Content-Type", "Authorization"],
)


@app.get("/healthz", response_model=HealthResponse, tags=["health"])
async def health_check(
    client: AsyncQdrantClient = Depends(get_qdrant_client),
) -> HealthResponse:
    """Liveness/readiness probe. The only endpoint that needs no token."""
    qdrant_status = "ok"
    try:
        await client.get_collections()
    except Exception as exc:
        logger.warning("Health check: Qdrant unreachable: %s", exc)
        qdrant_status = "unreachable"

    return HealthResponse(
        status="ok",
        version=__version__,
        qdrant=qdrant_status,
        cross_encoder=reranking.is_available(),
    )


app.include_router(v1_router)
