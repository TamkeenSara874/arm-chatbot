from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.config import get_settings
from src.utils.logging import configure_logging

# Configure structured logging before anything else
_settings = get_settings()
configure_logging(log_level=_settings.log_level, debug=_settings.debug)

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

    # Qdrant collections (idempotent create)
    try:
        from qdrant_client import AsyncQdrantClient
        from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams

        qdrant = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )

        # review_chunks uses named vectors: dense (ANN) + sparse (BM25-style via fastembed)
        if not await qdrant.collection_exists(settings.qdrant_collection_reviews):
            await qdrant.create_collection(
                collection_name=settings.qdrant_collection_reviews,
                vectors_config={
                    "dense": VectorParams(size=settings.embedding_dim, distance=Distance.COSINE),
                },
                sparse_vectors_config={"sparse": SparseVectorParams()},
            )
            logger.info("qdrant_collection_created", collection=settings.qdrant_collection_reviews)
        else:
            logger.info("qdrant_collection_exists", collection=settings.qdrant_collection_reviews)

        # correction_embeddings and session_memory use flat dense vectors only
        for name in [
            settings.qdrant_collection_corrections,
            settings.qdrant_collection_session_memory,
        ]:
            if not await qdrant.collection_exists(name):
                await qdrant.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=settings.embedding_dim, distance=Distance.COSINE
                    ),
                )
                logger.info("qdrant_collection_created", collection=name)
            else:
                logger.info("qdrant_collection_exists", collection=name)

        await qdrant.close()
    except Exception as exc:
        logger.error("qdrant_init_failed", error=str(exc))
        raise

    logger.info("startup_complete")
    yield

    logger.info("shutdown_begin")
    await engine.dispose()
    logger.info("shutdown_complete")


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

    # Rate limiting (per API key via Redis, shared across workers)
    limiter = Limiter(
        key_func=get_remote_address,
        storage_uri=settings.redis_url,  # shared across workers
    )
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Routes
    from src.api.routes import health
    from src.api.routes.chat import router as chat_router
    from src.api.routes.ingest import router as ingest_router

    app.include_router(health.router)
    app.include_router(chat_router)
    app.include_router(ingest_router)

    return app


app = create_app()
