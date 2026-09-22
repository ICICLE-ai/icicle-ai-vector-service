"""API v1 router.

All versioned endpoints are mounted under ``/v1`` from here, so the version
prefix lives in exactly one place. Adding a v2 later means a sibling package,
not edits scattered across handlers.
"""

from fastapi import APIRouter

from . import collections, embeddings, search

router = APIRouter(prefix="/v1")
router.include_router(embeddings.router)
router.include_router(collections.router)
router.include_router(search.router)

__all__ = ["router"]
