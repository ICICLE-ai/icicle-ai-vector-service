"""The unauthenticated health probe."""

import pytest

from src.app import __version__
from src.app.reranking import is_available

pytestmark = pytest.mark.asyncio


async def test_health_reports_status_and_version(client):
    body = (await client.get("/healthz")).json()
    assert body["status"] == "ok"
    assert body["version"] == __version__


async def test_health_reports_qdrant_reachability(client):
    assert (await client.get("/healthz")).json()["qdrant"] == "ok"


async def test_health_reports_cross_encoder_availability(client):
    assert (await client.get("/healthz")).json()["cross_encoder"] is is_available()
