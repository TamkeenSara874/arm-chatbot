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
    qdrant_collection_chat_cache: str = "chat_cache"

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

    # Speech-to-text (voice mode dictation). Groq's free tier (2,000
    # requests/day as of writing) comfortably covers real usage at
    # chatbot-question audio lengths -- same cost/latency tradeoff already
    # made for decomposition (Groq over a paid alternative).
    stt_provider: str = "groq"
    groq_stt_model: str = "whisper-large-v3-turbo"
    # Multipart audio uploads for voice dictation are short spoken questions,
    # not files -- a much tighter cap than the batch review-ingest limit.
    voice_max_upload_mb: int = 10

    # Model names — change these env vars to switch models, no code edits needed
    # gpt-4o-mini is the approved simple-query and entity-extraction model
    # gpt-4.1 replaces gpt-4o for complex queries (same capability tier, allowed by project key)
    openai_simple_model: str = "gpt-4o-mini"
    openai_complex_model: str = "gpt-4.1"
    openai_embed_model: str = "text-embedding-3-large"
    groq_decomp_model: str = "llama-3.3-70b-versatile"
    # Cross-encoder reranker (runs locally, no API cost). ms-marco-MiniLM-L6-v2
    # (~22M params) over bge-reranker-base (~278M params) -- the larger model
    # measured at 13-22s of CPU inference for 24-30 candidates in production,
    # the dominant end-to-end latency cost by a wide margin.
    # NOTE: must be the canonical repo id "ms-marco-MiniLM-L6-v2" (no hyphen
    # between L and 6) -- "ms-marco-MiniLM-L-6-v2" (hyphenated) is a HF Hub
    # redirect/alias to this same repo, and current transformers/huggingface_hub
    # releases fail AutoConfig resolution through that redirect (confirmed live:
    # a fresh `pip install -e .` today resolves transformers==5.13.0, which
    # raises "Unrecognized model... should have a model_type key" for the
    # hyphenated alias even though the config.json is valid -- the canonical
    # non-hyphenated id loads correctly on the exact same library versions).
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    # Live traces confirmed reranking at ~43-47% of total request latency.
    # ONNX int8 dynamic quantization only pays off once reranker_batch_size
    # (below) keeps candidate batches length-homogeneous -- verified live with
    # both fixes together: rerank_ms dropped from ~2.1-2.9s to ~0.8-1.0s, no
    # score/rank-order change (unit tests + isolated diff check), and the
    # LLM-judge eval harness (tests/e2e/test_eval_fixture.py) showed the exact
    # same failures with quantization on vs off -- all pre-existing/test-timing
    # issues, none attributable to the reranker backend. Explicit rollback
    # toggle either way -- flip via env var alone, no code deploy needed.
    # avx2 (not avx512_vnni, what's pre-published upstream) is the safe
    # default -- avx512_vnni needs recent server-class Intel CPUs and would be
    # the wrong choice on a typical dev machine or generic cloud VM.
    reranker_onnx_quantized: bool = True
    reranker_onnx_quantization_config: str = "avx2"
    # CrossEncoder.predict() already sorts candidates by length internally, but
    # only within a single batch -- with the default batch_size=32 and our
    # top_k always <=20, every request's candidates land in one batch anyway,
    # so the sort has no effect and one long review pads every candidate in
    # the request to its length. Confirmed live: dropping batch_size to 8 cut
    # rerank_ms by 2.5-4x on mixed-length candidates (e.g. 20 short + 4 long),
    # for both the torch and onnx-quantized backends, with rank order and
    # scores unchanged (max diff ~1e-6, floating-point noise from padding).
    reranker_batch_size: int = 8

    # Embeddings
    embedding_dim: int = 3072

    # Rate limiting (per API key)
    rate_limit_chat: str = "10/minute"
    rate_limit_ingest: str = "5/minute"
    # Single-review live ingestion is a lightweight, high-frequency call (one
    # review at a time, not a whole file) -- a much looser limit than the
    # batch file upload above.
    rate_limit_ingest_review: str = "60/minute"
    rate_limit_correct: str = "20/minute"
    rate_limit_read: str = "60/minute"
    rate_limit_voice: str = "20/minute"

    # Session
    session_ttl_days: int = 30
    # How often expired sessions are swept. Purely a background reaper, so a
    # long interval is fine -- the point is that it runs at all, not that it
    # runs promptly.
    session_purge_interval_hours: int = 6
    session_recent_messages: int = 5
    session_relevant_k: int = 3
    # The recent-messages block already covers session_recent_messages pairs
    # verbatim, so a summary only starts earning its keep once a conversation
    # runs past that window. At recent_messages=5 (10 messages) the gap opens
    # around message 10, which is why this is 20 and not the old 50 -- at 50,
    # conversations between 10 and 50 messages had no summary at all.
    session_summary_trigger: int = 20
    # The summary is refreshed every N messages beyond what it already covers,
    # rather than being written once and frozen.
    session_summary_refresh_every: int = 20
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

    # Semantic cache: cosine similarity a paraphrased query must clear against
    # a previously-cached (decomposed, rephrased) query before the cached
    # response is reused. Kept high -- a false positive here silently serves
    # the wrong answer with no error signal, unlike a retrieval false positive
    # which just adds one irrelevant evidence chunk among several.
    semantic_cache_similarity_threshold: float = 0.95

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
