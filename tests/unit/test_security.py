"""Unit tests for Tapis token validation.

This is the gate that decides *who* the caller is; every isolation guarantee
downstream depends on it returning the right username or refusing outright. The
JWKS lookup and signature check are stubbed — what is tested is the claim
validation the service layers on top, and that each failure mode is refused.
"""

import jwt
import pytest
from fastapi import HTTPException

from src.app.core import security
from src.app.core.settings import settings


def claims(**overrides) -> dict:
    base = {
        "tapis/username": "alice",
        "tapis/tenant_id": settings.tapis_tenant_id,
        "tapis/token_type": "access",
        "sub": "alice@icicleai",
    }
    base.update(overrides)
    return base


@pytest.fixture
def decodes_to(monkeypatch):
    """Stub JWKS + signature verification, returning chosen claims."""

    def _install(payload=None, raises=None):
        monkeypatch.setattr(
            security, "_jwk_client", lambda: type("K", (), {
                "get_signing_key_from_jwt": lambda self, token: type("S", (), {"key": "k"})()
            })()
        )

        def _decode(token, key, **kwargs):
            if raises is not None:
                raise raises
            return payload

        monkeypatch.setattr(jwt, "decode", _decode)

    return _install


class TestAcceptedTokens:
    def test_valid_token_yields_the_username(self, decodes_to):
        decodes_to(claims())
        assert security._decode_token("t")["tapis/username"] == "alice"

    def test_username_claim_is_preferred_over_sub(self, decodes_to):
        decodes_to(claims(**{"tapis/username": "real", "sub": "other"}))
        result = security._decode_token("t")
        assert result["tapis/username"] == "real"


class TestRejectedTokens:
    def test_expired_token(self, decodes_to):
        decodes_to(raises=jwt.ExpiredSignatureError())
        with pytest.raises(HTTPException) as excinfo:
            security._decode_token("t")
        assert excinfo.value.status_code == 401
        assert "expired" in excinfo.value.detail.lower()

    def test_wrong_issuer(self, decodes_to):
        decodes_to(raises=jwt.InvalidIssuerError())
        with pytest.raises(HTTPException) as excinfo:
            security._decode_token("t")
        assert excinfo.value.status_code == 401
        assert "issuer" in excinfo.value.detail.lower()

    def test_malformed_token(self, decodes_to):
        decodes_to(raises=jwt.DecodeError())
        with pytest.raises(HTTPException) as excinfo:
            security._decode_token("t")
        assert excinfo.value.status_code == 401

    def test_refresh_token_is_not_an_access_token(self, decodes_to):
        decodes_to(claims(**{"tapis/token_type": "refresh"}))
        with pytest.raises(HTTPException) as excinfo:
            security._decode_token("t")
        assert excinfo.value.status_code == 401
        assert "access token" in excinfo.value.detail.lower()

    def test_token_from_another_tenant_is_forbidden(self, decodes_to):
        """A valid token from a different Tapis tenant must not grant access."""
        decodes_to(claims(**{"tapis/tenant_id": "some-other-tenant"}))
        with pytest.raises(HTTPException) as excinfo:
            security._decode_token("t")
        assert excinfo.value.status_code == 403
        assert "some-other-tenant" in excinfo.value.detail


class TestGetCurrentUser:
    async def test_returns_the_user_context(self, decodes_to):
        decodes_to(claims())
        user = await security.get_current_user("token")
        assert user.username == "alice"
        assert user.tenant_id == settings.tapis_tenant_id

    async def test_falls_back_to_sub_when_username_is_absent(self, decodes_to):
        payload = claims()
        del payload["tapis/username"]
        decodes_to(payload)
        user = await security.get_current_user("token")
        assert user.username == "alice@icicleai"

    async def test_token_without_any_identity_is_refused(self, decodes_to):
        payload = claims()
        del payload["tapis/username"]
        del payload["sub"]
        decodes_to(payload)
        with pytest.raises(HTTPException) as excinfo:
            await security.get_current_user("token")
        assert excinfo.value.status_code == 401

    async def test_user_context_is_immutable(self, decodes_to):
        """A handler must not be able to widen its own access mid-request."""
        decodes_to(claims())
        user = await security.get_current_user("token")
        with pytest.raises(Exception):
            user.username = "bob"


class TestAuthIsActuallyRequired:
    """Without the test override, every v1 route must demand a token."""

    async def test_requests_without_a_token_are_refused(self, qdrant):
        from contextlib import asynccontextmanager

        from httpx import ASGITransport, AsyncClient

        from src.app.db import get_qdrant_client
        from src.app.main import app

        async def _qdrant():
            yield qdrant

        app.dependency_overrides = {get_qdrant_client: _qdrant}
        original = app.router.lifespan_context

        @asynccontextmanager
        async def _noop(_app):
            yield

        app.router.lifespan_context = _noop
        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as http:
                # No X-Tapis-Token header on any of these.
                assert (await http.get("/v1/collections")).status_code == 422
                assert (await http.post("/v1/retrieve", json={})).status_code == 422
                assert (await http.delete("/v1/collections?confirm=true")).status_code == 422
                # The health probe stays open.
                assert (await http.get("/healthz")).status_code == 200
        finally:
            app.dependency_overrides.clear()
            app.router.lifespan_context = original
