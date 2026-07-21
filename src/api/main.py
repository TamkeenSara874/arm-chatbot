import asyncio
from contextlib import asynccontextmanager

try:
    import sentry_sdk as _sentry_sdk
except ImportError:
    _sentry_sdk = None  # type: ignore[assignment]

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.config import Settings, get_settings
from src.utils.logging import configure_logging

# Configure structured logging before anything else
_settings = get_settings()
configure_logging(log_level=_settings.log_level, debug=_settings.debug)

if _sentry_sdk is not None and _settings.sentry_dsn:
    _sentry_sdk.init(
        dsn=_settings.sentry_dsn,
        environment=_settings.app_env,
        traces_sample_rate=0.1,
    )

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("startup_begin", env=settings.app_env, debug=settings.debug)

    # Postgres
    from sqlalchemy import text

    from src.services.database import get_engine

    engine = get_engine()
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("database_connected")
    except Exception as exc:
        logger.error("database_connection_failed", error=str(exc))
        raise  # fail fast if DB is unreachable at startup

    # Qdrant collections (idempotent create -- shared with scripts/seed.py so
    # whichever process starts first, API or seed job, creates them)
    try:
        from src.services.vector.qdrant_store import ensure_collections

        await ensure_collections(settings)
    except Exception as exc:
        logger.error("qdrant_init_failed", error=str(exc))
        raise

    # Warm up ML models so the first query doesn't incur 20-30s model-load latency
    try:
        from src.core.reranker import load_reranker
        from src.services.embedding.sparse_embedder import warmup_sparse_embedder

        await asyncio.gather(
            load_reranker(settings.reranker_model),
            warmup_sparse_embedder(),
        )
        logger.info("models_warmed_up", reranker=settings.reranker_model)
    except Exception as exc:
        logger.warning("model_warmup_failed", error=str(exc))

    # Sweep sessions past SESSION_TTL_DAYS. Runs in-process rather than as a
    # cron/worker because the app is the only deployable this repo ships; if a
    # scheduler is introduced later this should move there. Multiple uvicorn
    # workers each run their own loop -- the deletes are idempotent (an
    # absolute cutoff, not a relative one), so the duplicates are wasted work
    # rather than a correctness problem.
    purge_task = asyncio.create_task(_session_purge_loop(settings))

    logger.info("startup_complete")
    yield

    logger.info("shutdown_begin")
    purge_task.cancel()
    await engine.dispose()
    logger.info("shutdown_complete")


async def _session_purge_loop(settings: Settings) -> None:
    from src.api.dependencies import get_vector_store
    from src.core.session import purge_expired_sessions
    from src.services.database import get_session_factory

    interval_seconds = settings.session_purge_interval_hours * 3600
    while True:
        try:
            async with get_session_factory()() as db:
                await purge_expired_sessions(db, get_vector_store(), settings.session_ttl_days)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Never let a bad sweep kill the loop -- a reaper that silently
            # stopped after one transient Qdrant blip would reintroduce exactly
            # the unbounded growth it exists to prevent.
            logger.warning("session_purge_cycle_failed", error=str(exc))
        await asyncio.sleep(interval_seconds)


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="ARM Review RAG Chatbot",
        description="Production-grade RAG chatbot for restaurant review analytics.",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.debug else None,
    )

    # CORS (explicit list, never wildcard)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "Accept"],
        max_age=600,
    )

    # Rate limiting -- one shared, Redis-backed limiter (src/api/rate_limit.py),
    # keyed by restaurant_id when a JWT is present, else remote IP. Every route
    # file imports this same instance rather than constructing its own.
    from src.api.rate_limit import limiter

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Routes
    from src.api.routes import health
    from src.api.routes.auth import router as auth_router
    from src.api.routes.chat import router as chat_router
    from src.api.routes.ingest import router as ingest_router
    from src.api.routes.voice import router as voice_router

    app.include_router(health.router)
    app.include_router(auth_router)
    app.include_router(chat_router)
    app.include_router(ingest_router)
    app.include_router(voice_router)

    return app


app = create_app()
