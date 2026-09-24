"""Environment-backed configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # extra="ignore": a .env or pod environment routinely carries variables this
    # service does not own; rejecting them would fail at import time.
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="", extra="ignore"
    )

    # --- Qdrant ---
    qdrant_url: str
    qdrant_api_key: str | None = None

    # --- App ---
    app_env: str = "dev"
    allowed_origins: list[str] = ["*"]

    # --- Tapis auth (ICICLE AI tenant) ---
    tapis_issuer: str
    tapis_jwks_url: str
    tapis_tenant_id: str

    # --- Storage ---
    # Whether emptying a collection also drops it. Safe because collections are
    # per user; set false to preserve a collection's vector dimension.
    drop_empty_collections: bool = True

    # --- Cross-encoder reranking (optional; without it that method returns 503) ---
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Models a request may select. Anything else is rejected, so a client cannot
    # make the pod fetch arbitrary weights.
    rerank_allowed_models: list[str] = [
        "BAAI/bge-reranker-base",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ]
    # True warms the model during startup instead of on first request.
    rerank_preload: bool = False
    # Torch intra-op threads. 0 lets torch decide, which in a container reads the
    # host's CPU count rather than the cgroup quota — set this explicitly.
    rerank_threads: int = 0
    # Caps per-request cross-encoder work so one caller cannot monopolise the CPU.
    rerank_max_candidates: int = 128
    rerank_max_length: int = 512


settings = Settings()  # type: ignore[call-arg]
