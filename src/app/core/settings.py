"""Environment-backed configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # extra="ignore": a .env or a pod's environment routinely carries variables
    # this service does not own (e.g. a client-side TAPIS_TOKEN). Rejecting them
    # would crash the app at import time, which is how the stale VECTOR_DIM key
    # left over from an earlier release broke local startup.
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

    # --- Storage hygiene ---
    # Whether emptying a collection also removes it from Qdrant. ON by default:
    # collections are per user (see app.db.naming), so the collection being
    # dropped is always the caller's own and no other user can be writing into
    # it. Set false to keep emptied collections around — e.g. to preserve a
    # collection's vector dimension across a full delete and re-ingest.
    drop_empty_collections: bool = True

    # --- Cross-encoder reranking ---
    # The cross-encoder is optional: the service runs fine without torch /
    # sentence-transformers installed, and the "cross_encoder" rerank method
    # returns 503 in that case. Everything else keeps working.
    # Default model used when a request does not name one.
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Models a request is allowed to select via "rerank_model". Anything outside
    # this list is rejected, so a client cannot make the pod download and load
    # arbitrary weights from the internet. Each selected model is loaded once and
    # cached for the life of the process.
    rerank_allowed_models: list[str] = [
        "BAAI/bge-reranker-base",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ]
    # Loaded on first use rather than at startup so a pod that never reranks
    # never pays the memory cost. Set true to warm it during lifespan instead.
    rerank_preload: bool = False
    # Torch intra-op threads. Defaults to 0 = let torch decide from the cgroup.
    rerank_threads: int = 0
    # Max candidates a single cross-encoder call will score. This is a fairness
    # guard as much as a performance one: without it, one user sending fetch_k=500
    # repeatedly would monopolise the pod's CPU and slow every other user down.
    rerank_max_candidates: int = 128
    # Truncation length for the (query, passage) pair fed to the model.
    rerank_max_length: int = 512


settings = Settings()  # type: ignore[call-arg]
