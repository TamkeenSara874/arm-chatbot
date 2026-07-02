from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Application
    app_env: str = "development"
    debug: bool = False
    api_key: str = "change-me-local-dev-key"
    log_level: str = "INFO"
    allowed_origins: str = "http://localhost:5173,http://localhost:3000"
    sentry_dsn: str = ""

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/armchatbot"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection_reviews: str = "review_chunks"
    qdrant_collection_corrections: str = "correction_embeddings"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600

    # LLM providers
    groq_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # Model names — change these env vars to switch GPT-4 <-> GPT-5, no code edits needed
    openai_simple_model: str = "gpt-4o-mini"
    openai_complex_model: str = "gpt-4o"
    openai_embed_model: str = "text-embedding-3-large"
    groq_decomp_model: str = "llama-3.3-70b-versatile"
    anthropic_fallback_model: str = "claude-sonnet-4-6"

    # Embeddings
    embedding_dim: int = 3072

    # Rate limiting (per API key)
    rate_limit_chat: str = "10/minute"
    rate_limit_ingest: str = "5/minute"
    rate_limit_correct: str = "20/minute"
    rate_limit_read: str = "60/minute"

    # Session
    session_ttl_days: int = 30
    session_summary_trigger: int = 6

    # Ingestion
    chunk_size_tokens: int = 256
    chunk_overlap_tokens: int = 32
    ingest_batch_size: int = 100
    ingest_max_file_size_mb: int = 10
    entity_extraction_batch_size: int = 10

    # Ranking weights (must sum to 1.0)
    ranking_weight_rrf: float = 0.5
    ranking_weight_recency: float = 0.3
    ranking_weight_rating: float = 0.2

    # Correction
    correction_sim_threshold: float = 0.85

    # Evidence quality
    data_staleness_days: int = 365

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    model_config = {"env_file": ".env", "case_sensitive": False, "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
