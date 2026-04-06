from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="")

    qdrant_url: str
    qdrant_api_key: str | None = None
    vector_dim: int = 768
    collection_prefix: str = "embeddings"
    app_env: str = "dev"
    tapis_issuer: str
    tapis_jwks_url: str
    tapis_tenant_id: str
    allowed_origins: list[str] = ["*"]


settings = Settings()  # type: ignore[call-arg]
