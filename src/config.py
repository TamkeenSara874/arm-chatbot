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
    qdrant_collection_session_memory: str = "session_memory"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    # 24h: review data only changes on re-ingestion (not continuously), and
    # corrections already bust their own cache entry via
    # RedisCache.invalidate_query(), so a longer TTL is pure cost savings
    # without a staleness downside.
    cache_ttl_seconds: int = 86400

    # LLM providers
    groq_api_key: str = ""
    # Optional comma-separated list of additional free-tier Groq keys (each
    # has its own daily token quota). When set, decomposition rotates across
    # all of them (groq_api_key + these) instead of falling back to paid
    # OpenAI the moment a single key's quota is exhausted.
    groq_api_keys: str = ""
    openai_api_key: str = ""

    # Model names — change these env vars to switch models, no code edits needed
    # gpt-4o-mini is the approved simple-query and entity-extraction model
    # gpt-4.1 replaces gpt-4o for complex queries (same capability tier, allowed by project key)
    openai_simple_model: str = "gpt-4o-mini"
    openai_complex_model: str = "gpt-4.1"
    openai_embed_model: str = "text-embedding-3-large"
    groq_decomp_model: str = "llama-3.3-70b-versatile"
    # Cross-encoder reranker (runs locally, no API cost). ms-marco-MiniLM-L-6-v2
    # (~22M params) over bge-reranker-base (~278M params) -- the larger model
    # measured at 13-22s of CPU inference for 24-30 candidates in production,
    # the dominant end-to-end latency cost by a wide margin.
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Embeddings
    embedding_dim: int = 3072

    # Rate limiting (per API key)
    rate_limit_chat: str = "10/minute"
    rate_limit_ingest: str = "5/minute"
    rate_limit_correct: str = "20/minute"
    rate_limit_read: str = "60/minute"

    # Session
    session_ttl_days: int = 30
    session_recent_messages: int = 5
    session_relevant_k: int = 3
    session_summary_trigger: int = 50
    session_context_token_budget: int = 6000

    # Ingestion
    chunk_size_tokens: int = 256
    chunk_overlap_tokens: int = 32
    ingest_batch_size: int = 100
    ingest_max_file_size_mb: int = 10
    entity_extraction_batch_size: int = 10
    entity_extraction_concurrency: int = 8

    # Ranking weights (must sum to 1.0)
    ranking_weight_rrf: float = 0.5
    ranking_weight_recency: float = 0.3
    ranking_weight_rating: float = 0.2

    # Correction
    correction_sim_threshold: float = 0.85

    # Evidence quality
    data_staleness_days: int = 365

    # JWT (restaurant-scoped tokens issued after API key auth)
    jwt_secret: str = "change-me-jwt-secret-32-chars-min"
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def groq_api_key_list(self) -> list[str]:
        """groq_api_key plus any additional keys in groq_api_keys, deduplicated, order preserved."""
        keys = [self.groq_api_key] if self.groq_api_key else []
        keys += [k.strip() for k in self.groq_api_keys.split(",") if k.strip()]
        seen: set[str] = set()
        deduped = []
        for k in keys:
            if k not in seen:
                seen.add(k)
                deduped.append(k)
        return deduped

    model_config = {"env_file": ".env", "case_sensitive": False, "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
