from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, Security
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import text

from src.api.dependencies import DbSession
from src.config import Settings, get_settings

router = APIRouter(tags=["health"])
logger = structlog.get_logger()
_bearer = HTTPBearer(auto_error=False)


@router.get("/health", summary="Liveness probe — no auth, no external checks")
async def liveness() -> dict:
    return {"status": "ok"}


@router.get("/health/ready", summary="Readiness probe — checks Postgres, Qdrant, Redis")
async def readiness(db: DbSession) -> JSONResponse:
    settings = get_settings()
    checks: dict[str, str | bool] = {}
    overall = "ready"

    # Postgres
    try:
        await db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:
        logger.error("readiness_db_failed", error=str(exc))
        checks["database"] = "unreachable"
        overall = "not_ready"

    # Redis
    try:
        import redis.asyncio as aioredis

        r = aioredis.from_url(settings.redis_url, decode_responses=True)
        await r.ping()
        await r.aclose()
        checks["redis"] = "ok"
    except Exception as exc:
        logger.warning("readiness_redis_failed", error=str(exc))
        checks["redis"] = "unreachable"
        overall = "not_ready"

    # Qdrant
    try:
        from qdrant_client import AsyncQdrantClient

        qdrant = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
        # AsyncQdrantClient has no health_check() method in the installed
        # qdrant-client version (confirmed via direct test -- raises
        # AttributeError, which the broad except below was silently
        # swallowing as "unreachable" even with Qdrant fully healthy).
        # get_collections() is a real, lightweight connectivity check.
        await qdrant.get_collections()
        await qdrant.close()
        checks["qdrant"] = "ok"
    except Exception as exc:
        logger.warning("readiness_qdrant_failed", error=str(exc))
        checks["qdrant"] = "unreachable"
        overall = "not_ready"

    # Model warmup status -- informational only, doesn't gate `overall`.
    # Warmup already runs (and blocks) at startup in main.py's lifespan
    # specifically so a chat query is never the first thing that triggers
    # the ~20-30s reranker/sparse-model load; surfaced here so that can be
    # confirmed at a glance instead of grepping startup logs.
    from src.core.reranker import is_warmed_up as reranker_is_warmed_up
    from src.services.embedding.sparse_embedder import is_warmed_up as sparse_is_warmed_up

    checks["reranker_warmed_up"] = reranker_is_warmed_up(settings.reranker_model)
    checks["sparse_embedder_warmed_up"] = sparse_is_warmed_up()

    status_code = 200 if overall == "ready" else 503
    return JSONResponse(content={"status": overall, **checks}, status_code=status_code)


@router.get(
    "/health/metrics",
    summary="Prometheus metrics — requires Bearer auth",
    include_in_schema=False,
)
async def metrics(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(_bearer)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> Response:
    from fastapi import HTTPException

    if credentials is None or credentials.credentials != settings.api_key:
        raise HTTPException(status_code=403, detail="Forbidden")

    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
