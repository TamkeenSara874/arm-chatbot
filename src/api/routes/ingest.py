"""Ingest and restaurant discovery routes."""

# NOTE: deliberately no `from __future__ import annotations` here. Combined
# with slowapi's @limiter.limit() decorator, postponed evaluation leaves the
# `file: UploadFile` parameter as an unresolved ForwardRef in the wrapped
# function's namespace, and FastAPI's dependant analysis crashes at import
# time with "Invalid args for response field! ... ForwardRef('UploadFile')"
# -- confirmed via isolated repro. This is the only route file with an
# UploadFile parameter, so it's the only one that needs this left off.

import uuid
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, status
from sqlalchemy import distinct, select
from sqlalchemy.exc import IntegrityError

from src.api.dependencies import (
    AuthToken,
    DbSession,
    get_cache,
    get_embedder,
    get_simple_client,
    get_vector_store,
)
from src.api.rate_limit import limiter
from src.config import get_settings
from src.models.db_entities import IngestJob, ReviewChunkMeta
from src.models.schemas import (
    IngestJobResponse,
    RestaurantListResponse,
    ReviewIngestRequest,
    ReviewIngestResponse,
)
from src.services.cache import RedisCache
from src.services.embedding.base import BaseEmbedder
from src.services.llm.base import BaseLLMClient
from src.services.vector.base import BaseVectorStore
from src.utils.background import fire_and_forget
from src.utils.security import check_file_upload
from src.workers.ingest_worker import ingest_single_review, run_ingest_job

logger = structlog.get_logger()

router = APIRouter(prefix="/api/v1", tags=["ingest"])

settings = get_settings()

SimpleClient = Annotated[BaseLLMClient, Depends(get_simple_client)]
Embedder = Annotated[BaseEmbedder, Depends(get_embedder)]
VectorStore = Annotated[BaseVectorStore, Depends(get_vector_store)]
Cache = Annotated[RedisCache, Depends(get_cache)]


@router.post("/ingest", response_model=IngestJobResponse, status_code=status.HTTP_202_ACCEPTED)
@limiter.limit(settings.rate_limit_ingest)
async def ingest_reviews(
    request: Request,
    restaurant_id: int,
    file: UploadFile,
    _: AuthToken,
    db: DbSession,
    llm_client: SimpleClient,
    embedder: Embedder,
    vector_store: VectorStore,
    cache: Cache,
) -> IngestJobResponse:
    file_bytes = await file.read()

    try:
        check_file_upload(
            filename=file.filename or "upload.json",
            content_type=file.content_type or "",
            size_bytes=len(file_bytes),
        )
    except ValueError as exc:
        status_code = (
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
            if "too large" in str(exc)
            else status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
        )
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc

    job = IngestJob(
        restaurant_id=restaurant_id,
        filename=file.filename or "upload.json",
        status="pending",
    )
    db.add(job)
    try:
        await db.commit()
    except IntegrityError as exc:
        # ix_ingest_job_one_active_per_restaurant (partial unique index on
        # restaurant_id WHERE status IN ('pending','processing')) is the
        # single source of truth here -- a pre-check SELECT would have a
        # TOCTOU race under concurrent requests, the DB constraint doesn't.
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An ingest job is already pending or processing for this restaurant",
        ) from exc
    await db.refresh(job)

    logger.info(
        "ingest_job_created",
        job_id=str(job.id),
        restaurant_id=restaurant_id,
        filename=job.filename,
        size_bytes=len(file_bytes),
    )

    fire_and_forget(
        run_ingest_job(
            job_id=job.id,
            restaurant_id=restaurant_id,
            file_content=file_bytes,
            settings=settings,
            embedder=embedder,
            vector_store=vector_store,
            reviews_collection=settings.qdrant_collection_reviews,
            llm_client=llm_client,
            cache=cache,
        ),
        name=f"ingest-{job.id}",
    )

    return IngestJobResponse(
        job_id=job.id,
        status=job.status,
        progress_pct=job.progress_pct,
    )


@router.post("/ingest/review", response_model=ReviewIngestResponse, status_code=status.HTTP_200_OK)
@limiter.limit(settings.rate_limit_ingest_review)
async def ingest_review(
    request: Request,
    body: ReviewIngestRequest,
    _: AuthToken,
    db: DbSession,
    llm_client: SimpleClient,
    embedder: Embedder,
    vector_store: VectorStore,
    cache: Cache,
) -> ReviewIngestResponse:
    """Ingest, or update, exactly one review the moment a source system has
    it -- the push-based counterpart to the batch file upload above, run
    inline (not fire_and_forget) since one review's chunk/extract/embed/
    upsert work is fast enough for a normal request-response cycle.
    """
    result = await ingest_single_review(
        restaurant_id=body.restaurant_id,
        external_review_id=body.external_review_id,
        review_text=body.review,
        rating=body.rating,
        username=body.username,
        source=body.source,
        created_at_raw=body.created_at,
        sentiment_label=body.sentiment,
        settings=settings,
        db_session=db,
        embedder=embedder,
        vector_store=vector_store,
        reviews_collection=settings.qdrant_collection_reviews,
        llm_client=llm_client,
        cache=cache,
    )
    return ReviewIngestResponse(
        review_id=result.review_id,
        status=result.status,
        chunks_written=result.chunks_written,
    )


@router.get("/ingest/{job_id}/status", response_model=IngestJobResponse)
@limiter.limit(settings.rate_limit_read)
async def get_ingest_status(
    request: Request,
    job_id: uuid.UUID,
    _: AuthToken,
    db: DbSession,
) -> IngestJobResponse:
    job = await db.get(IngestJob, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ingest job not found")

    return IngestJobResponse(
        job_id=job.id,
        status=job.status,
        progress_pct=job.progress_pct,
        total_reviews=job.total_reviews,
        total_chunks=job.total_chunks,
        skipped_empty=job.skipped_empty,
        skipped_already_processed=job.skipped_already_processed,
        error_message=job.error_message,
    )


@router.get("/restaurants", response_model=RestaurantListResponse)
@limiter.limit(settings.rate_limit_read)
async def list_restaurants(
    request: Request,
    _: AuthToken,
    db: DbSession,
) -> RestaurantListResponse:
    stmt = select(distinct(ReviewChunkMeta.restaurant_id)).order_by(ReviewChunkMeta.restaurant_id)
    result = await db.execute(stmt)
    ids = [row[0] for row in result.all()]
    return RestaurantListResponse(restaurant_ids=ids)
