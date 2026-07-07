from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import structlog
from sqlalchemy import delete, distinct, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.chunking import chunk_text
from src.models.db_entities import IngestJob, ReviewChunkMeta
from src.services.cache import RedisCache
from src.services.database import get_session_factory
from src.services.embedding.base import BaseEmbedder
from src.services.embedding.sparse_embedder import compute_sparse_vectors_batch
from src.services.llm.base import BaseLLMClient, UsageCallback
from src.services.vector.base import BaseVectorStore
from src.utils.metrics import ingest_reviews_total
from src.utils.security import flag_injection
from src.utils.tracing import IngestTrace

logger = structlog.get_logger()

# Bump this string whenever the embedding model, chunking strategy, or entity
# extraction prompt changes. The seed script uses it to detect stale vectors.
PIPELINE_VERSION = "1.0.0"

SENTIMENT_RATING_BUCKETS: dict[str, str] = {
    "positive": "Positive",
    "negative": "Negative",
    "mixed": "Mixed",
    "neutral": "Neutral",
}

RATING_TO_SENTIMENT: dict[int, str] = {
    1: "Negative",
    2: "Negative",
    3: "Mixed",
    4: "Positive",
    5: "Positive",
}

DATE_FORMATS: list[str] = [
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
]


async def run_ingest_job(
    job_id: uuid.UUID,
    restaurant_id: int,
    file_content: bytes,
    settings: object,
    embedder: BaseEmbedder,
    vector_store: BaseVectorStore,
    reviews_collection: str,
    llm_client: BaseLLMClient,
    cache: RedisCache,
) -> None:
    """Execute a full ingestion pipeline for one uploaded review file.

    Runs as a fire_and_forget() task -- never use FastAPI BackgroundTasks for
    this because BackgroundTasks blocks the event loop during heavy embedding I/O.
    Creates its own DB session to avoid sharing a request-scoped session.
    """
    session_factory = get_session_factory()
    async with session_factory() as db_session:
        await _run(
            job_id=job_id,
            restaurant_id=restaurant_id,
            file_content=file_content,
            settings=settings,
            db_session=db_session,
            embedder=embedder,
            vector_store=vector_store,
            reviews_collection=reviews_collection,
            llm_client=llm_client,
            cache=cache,
        )


async def _run(
    job_id: uuid.UUID,
    restaurant_id: int,
    file_content: bytes,
    settings: object,
    db_session: AsyncSession,
    embedder: BaseEmbedder,
    vector_store: BaseVectorStore,
    reviews_collection: str,
    llm_client: BaseLLMClient,
    cache: RedisCache,
) -> None:
    job = await db_session.get(IngestJob, job_id)
    if not job:
        logger.error("ingest_job_not_found", job_id=str(job_id))
        return

    try:
        rows = _parse_json(file_content)
    except Exception as exc:
        await _fail_job(db_session, job, str(exc))
        return

    job.total_reviews = len(rows)
    job.status = "processing"
    await db_session.commit()

    try:
        await _process_rows(
            rows=rows,
            restaurant_id=restaurant_id,
            settings=settings,
            db_session=db_session,
            job=job,
            embedder=embedder,
            vector_store=vector_store,
            reviews_collection=reviews_collection,
            llm_client=llm_client,
            cache=cache,
        )
    except Exception as exc:
        logger.error(
            "ingest_pipeline_failed",
            job_id=str(job_id),
            error=str(exc),
            exc_info=True,
        )
        await _fail_job(db_session, job, str(exc))
        raise


_META_COLUMNS = [
    "chunk_id",
    "restaurant_id",
    "review_id",
    "chunk_text",
    "full_review",
    "has_content",
    "rating",
    "sentiment_label",
    "sentiment_rating_agree",
    "review_date",
    "username",
    "source",
    "chunk_index",
    "has_injection_attempt",
    "date_inferred",
    "pipeline_version",
]


async def _process_rows(
    rows: list[dict[str, Any]],
    restaurant_id: int,
    settings: object,
    db_session: AsyncSession,
    job: IngestJob,
    embedder: BaseEmbedder,
    vector_store: BaseVectorStore,
    reviews_collection: str,
    llm_client: BaseLLMClient,
    cache: RedisCache,
) -> None:
    """Chunk, extract entities, embed, and upsert every row, one row-batch at a
    time. Raises on failure so the caller can mark the job failed with a real
    error message instead of leaving it stuck at status=processing.

    Each row-batch fully commits to both Qdrant and Postgres (in that order)
    before moving to the next batch -- unlike the old single-pass-per-store
    design, a crash partway through never leaves Qdrant ahead of Postgres by
    more than one batch. That, plus a skip check keyed on review_id +
    PIPELINE_VERSION, is what makes a retried job cheap: rows a prior attempt
    already finished are never re-chunked, re-sent to the entity-extraction
    LLM, or re-embedded.
    """
    chunk_size: int = getattr(settings, "chunk_size_tokens", 256)
    overlap: int = getattr(settings, "chunk_overlap_tokens", 32)
    batch_size: int = getattr(settings, "ingest_batch_size", 100)
    entity_batch: int = getattr(settings, "entity_extraction_batch_size", 10)
    entity_concurrency: int = getattr(settings, "entity_extraction_concurrency", 8)

    trace = IngestTrace(job_id=str(job.id), restaurant_id=restaurant_id)
    entity_semaphore = asyncio.Semaphore(entity_concurrency)
    entity_model_name = getattr(llm_client, "model", "unknown")
    embedding_model_name = getattr(embedder, "model", "unknown")

    total_chunks = 0
    skipped_empty = 0
    skipped_already_processed = 0

    for batch_start in range(0, len(rows), batch_size):
        batch_rows = rows[batch_start : batch_start + batch_size]

        review_ids: list[str] = []
        row_infos: list[dict[str, Any]] = []
        for offset, raw_row in enumerate(batch_rows):
            row_idx = batch_start + offset
            row = _correct_row(raw_row)

            review_text: str = row.get("review", "")
            has_content = bool(review_text.strip())

            created_at_raw = row.get("createdAt") or row.get("created_at", "")
            review_date, date_inferred = _parse_date(created_at_raw)

            rating_raw = row.get("rating")
            rating: float | None = None
            if rating_raw is not None:
                try:
                    rating = float(rating_raw)
                except (ValueError, TypeError):
                    rating = None

            sentiment_label: str = row.get("sentiment") or "Neutral"
            username: str = row.get("username") or "Anonymous"
            source: str = row.get("source") or "Unknown"
            review_id = _derive_review_id(restaurant_id, username, created_at_raw, row_idx)

            review_ids.append(review_id)
            row_infos.append(
                {
                    "review_id": review_id,
                    "review_text": review_text,
                    "has_content": has_content,
                    "review_date": review_date,
                    "date_inferred": date_inferred,
                    "rating": rating,
                    "sentiment_label": sentiment_label,
                    "username": username,
                    "source": source,
                }
            )

        already_processed = await _fetch_processed_review_ids(
            db_session, restaurant_id, review_ids, PIPELINE_VERSION
        )

        batch_meta: list[ReviewChunkMeta] = []
        batch_points: list[dict[str, Any]] = []

        for info in row_infos:
            if info["review_id"] in already_processed:
                skipped_already_processed += 1
                continue

            if not info["has_content"]:
                skipped_empty += 1
                ingest_reviews_total.labels(status="skipped_empty").inc()
                batch_meta.append(
                    ReviewChunkMeta(
                        chunk_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{info['review_id']}_0")),
                        restaurant_id=restaurant_id,
                        review_id=info["review_id"],
                        chunk_text=None,
                        full_review=None,
                        has_content=False,
                        rating=info["rating"],
                        sentiment_label=info["sentiment_label"],
                        sentiment_rating_agree=_check_sentiment_rating(
                            info["rating"], info["sentiment_label"]
                        ),
                        review_date=info["review_date"],
                        username=info["username"],
                        source=info["source"],
                        chunk_index=0,
                        has_injection_attempt=False,
                        date_inferred=info["date_inferred"],
                        pipeline_version=PIPELINE_VERSION,
                    )
                )
                continue

            metas, points = _build_review_chunks(
                restaurant_id=restaurant_id,
                review_id=info["review_id"],
                review_text=info["review_text"],
                rating=info["rating"],
                sentiment_label=info["sentiment_label"],
                username=info["username"],
                source=info["source"],
                review_date=info["review_date"],
                date_inferred=info["date_inferred"],
                chunk_size=chunk_size,
                overlap=overlap,
            )
            batch_meta.extend(metas)
            batch_points.extend(points)
            ingest_reviews_total.labels(status="ingested").inc()

        # Entity extraction (batched LLM calls, bounded concurrency) for this
        # row-batch only -- a semaphore caps concurrent OpenAI calls without
        # overwhelming rate limits.
        content_points = [p for p in batch_points if p["text"]]
        t_entities = time.perf_counter()
        await _extract_entities_into_points(
            content_points,
            llm_client,
            entity_semaphore,
            entity_batch,
            usage_callback=lambda p, c, ca: trace.record_entity_tokens(entity_model_name, p, c),
        )
        trace.entity_extraction_ms += (time.perf_counter() - t_entities) * 1000.0

        # Embedding and Qdrant upsert (sub-batched) with named dense + sparse vectors
        t_embed = time.perf_counter()
        for i in range(0, len(batch_points), batch_size):
            sub_batch = batch_points[i : i + batch_size]
            texts = [p["text"] for p in sub_batch]
            dense_vectors, sparse_vectors = await asyncio.gather(
                embedder.embed(
                    texts,
                    usage_callback=lambda t: trace.record_embedding_tokens(embedding_model_name, t),
                ),
                compute_sparse_vectors_batch(texts),
            )
            qdrant_points = [
                {
                    "id": p["chunk_id"],
                    "vector": {
                        "dense": dv,
                        "sparse": {"indices": sv.indices, "values": sv.values},
                    },
                    "payload": {k: val for k, val in p.items() if k != "chunk_id"},
                }
                for p, dv, sv in zip(sub_batch, dense_vectors, sparse_vectors, strict=True)
            ]
            await vector_store.upsert(reviews_collection, qdrant_points)
        trace.embedding_upsert_ms += (time.perf_counter() - t_embed) * 1000.0

        # Postgres upsert happens right after Qdrant, in the same batch --
        # not deferred to after the whole file finishes -- so a crash on the
        # next batch leaves both stores consistent with each other instead of
        # leaving Postgres with nothing despite Qdrant having full data.
        await _upsert_chunk_meta(db_session, batch_meta, batch_size)

        total_chunks += len(batch_points)
        job.progress_pct = min(int(((batch_start + len(batch_rows)) / len(rows)) * 100), 99)
        await db_session.commit()

    job.total_chunks = total_chunks
    job.skipped_empty = skipped_empty
    job.skipped_already_processed = skipped_already_processed
    await db_session.commit()

    # Cache invalidation
    deleted = await cache.invalidate_restaurant(restaurant_id)
    logger.info(
        "ingest_caches_invalidated",
        restaurant_id=restaurant_id,
        redis_keys_deleted=deleted,
    )

    job.status = "complete"
    job.progress_pct = 100
    await db_session.commit()

    logger.info(
        "ingest_job_complete",
        job_id=str(job.id),
        restaurant_id=restaurant_id,
        total_reviews=len(rows),
        total_chunks=total_chunks,
        skipped_empty=skipped_empty,
        skipped_already_processed=skipped_already_processed,
    )

    trace.total_reviews = len(rows)
    trace.total_chunks = total_chunks
    trace.skipped_empty = skipped_empty
    trace.skipped_already_processed = skipped_already_processed
    trace.emit()


def _build_review_chunks(
    restaurant_id: int,
    review_id: str,
    review_text: str,
    rating: float | None,
    sentiment_label: str,
    username: str,
    source: str,
    review_date: datetime | None,
    date_inferred: bool,
    chunk_size: int,
    overlap: int,
) -> tuple[list[ReviewChunkMeta], list[dict[str, Any]]]:
    """Chunk one review's text and build its metadata rows + Qdrant point
    stubs (food_entities empty, vectors not yet computed). Pure, no I/O --
    shared by the batch ingest path and the single-review live-ingest path.
    """
    sentiment_rating_agree = _check_sentiment_rating(rating, sentiment_label)
    chunks = chunk_text(review_text, chunk_size=chunk_size, overlap_tokens=overlap)

    metas: list[ReviewChunkMeta] = []
    points: list[dict[str, Any]] = []
    for chunk_idx, chunk in enumerate(chunks):
        has_injection = flag_injection(chunk, restaurant_id)
        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{review_id}_{chunk_idx}"))
        metas.append(
            ReviewChunkMeta(
                chunk_id=chunk_id,
                restaurant_id=restaurant_id,
                review_id=review_id,
                chunk_text=chunk,
                full_review=review_text if chunk_idx == 0 else None,
                has_content=True,
                rating=rating,
                sentiment_label=sentiment_label,
                sentiment_rating_agree=sentiment_rating_agree,
                review_date=review_date,
                username=username,
                source=source,
                chunk_index=chunk_idx,
                has_injection_attempt=has_injection,
                date_inferred=date_inferred,
                pipeline_version=PIPELINE_VERSION,
            )
        )
        points.append(
            {
                "chunk_id": chunk_id,
                "text": chunk,
                "restaurant_id": restaurant_id,
                "review_id": review_id,
                "rating": rating,
                "username": username,
                "source": source,
                "review_date": review_date.isoformat() if review_date else None,
                "review_date_ts": int(review_date.timestamp()) if review_date else None,
                "chunk_index": chunk_idx,
                "sentiment_label": sentiment_label,
                "sentiment_rating_agree": sentiment_rating_agree,
                "has_injection_attempt": has_injection,
                "date_inferred": date_inferred,
                "food_entities": [],
            }
        )
    return metas, points


async def _extract_entities_into_points(
    content_points: list[dict[str, Any]],
    llm_client: BaseLLMClient,
    semaphore: asyncio.Semaphore,
    entity_batch: int,
    usage_callback: UsageCallback | None = None,
) -> None:
    """Run batched entity extraction over content_points, writing results into
    each point's food_entities field in place."""

    async def _extract_batch(batch: list[dict[str, Any]]) -> None:
        async with semaphore:
            entities_per_chunk = await _extract_entities(
                llm_client,
                [p["text"] for p in batch],
                usage_callback=usage_callback,
            )
        for point, entities in zip(batch, entities_per_chunk, strict=True):
            point["food_entities"] = entities

    await asyncio.gather(
        *(
            _extract_batch(content_points[i : i + entity_batch])
            for i in range(0, len(content_points), entity_batch)
        )
    )


async def _upsert_chunk_meta(
    db_session: AsyncSession, metas: list[ReviewChunkMeta], batch_size: int
) -> None:
    """Bulk upsert ReviewChunkMeta rows. ON CONFLICT DO UPDATE on the chunk_id
    primary key makes this safe to call again with the same chunk_ids (a
    retried batch, or an updated live-ingested review).
    """
    if not metas:
        return
    rows = [{col: getattr(meta, col) for col in _META_COLUMNS} for meta in metas]
    # Postgres/asyncpg cap query parameters at 32767; with 16 columns per
    # row that's ~2048 rows max in one statement, so chunk well under that.
    for i in range(0, len(rows), batch_size):
        row_chunk = rows[i : i + batch_size]
        stmt = pg_insert(ReviewChunkMeta).values(row_chunk)
        update_cols = {col: stmt.excluded[col] for col in _META_COLUMNS if col != "chunk_id"}
        stmt = stmt.on_conflict_do_update(index_elements=["chunk_id"], set_=update_cols)
        await db_session.execute(stmt)


async def _fetch_processed_review_ids(
    db_session: AsyncSession,
    restaurant_id: int,
    review_ids: list[str],
    pipeline_version: str,
) -> set[str]:
    """Which of these review_ids already have chunks written under the
    current PIPELINE_VERSION -- a resumed job skips re-chunking,
    re-extracting entities, and re-embedding these, while a row left over
    from an older pipeline version (pipeline_version mismatch, including
    legacy rows with NULL) is correctly treated as not-yet-processed.
    """
    if not review_ids:
        return set()
    stmt = select(distinct(ReviewChunkMeta.review_id)).where(
        ReviewChunkMeta.restaurant_id == restaurant_id,
        ReviewChunkMeta.review_id.in_(review_ids),
        ReviewChunkMeta.pipeline_version == pipeline_version,
    )
    result = await db_session.execute(stmt)
    return {row[0] for row in result.all()}


async def _fetch_existing_chunk_ids(db_session: AsyncSession, review_id: str) -> list[str]:
    stmt = select(ReviewChunkMeta.chunk_id).where(ReviewChunkMeta.review_id == review_id)
    result = await db_session.execute(stmt)
    return [row[0] for row in result.all()]


@dataclass
class ReviewIngestResult:
    review_id: str
    status: str  # "ingested" | "updated" | "skipped_empty"
    chunks_written: int


async def ingest_single_review(
    restaurant_id: int,
    external_review_id: str,
    review_text: str,
    rating: float | None,
    username: str | None,
    source: str | None,
    created_at_raw: str | None,
    sentiment_label: str | None,
    settings: object,
    db_session: AsyncSession,
    embedder: BaseEmbedder,
    vector_store: BaseVectorStore,
    reviews_collection: str,
    llm_client: BaseLLMClient,
    cache: RedisCache,
) -> ReviewIngestResult:
    """Ingest, or update, exactly one review -- the live/incremental
    counterpart to run_ingest_job's batch file upload, for a source system
    that calls the moment a review is added or edited instead of waiting for
    a full re-upload.

    review_id is derived from (restaurant_id, external_review_id) rather than
    a row position in a file, so calling this again with the same
    external_review_id is a genuine update: it reuses the same chunk_ids
    (naturally upserting changed text/rating/etc via ON CONFLICT DO UPDATE)
    and deletes any now-stale chunks left over if the edited review produces
    fewer chunks than the version it's replacing.
    """
    chunk_size: int = getattr(settings, "chunk_size_tokens", 256)
    overlap: int = getattr(settings, "chunk_overlap_tokens", 32)
    entity_batch: int = getattr(settings, "entity_extraction_batch_size", 10)
    entity_concurrency: int = getattr(settings, "entity_extraction_concurrency", 8)

    review_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{restaurant_id}:{external_review_id}"))
    has_content = bool(review_text.strip())
    review_date, date_inferred = _parse_date(created_at_raw)
    sentiment_label = sentiment_label or "Neutral"
    username = username or "Anonymous"
    source = source or "Unknown"

    existing_chunk_ids = await _fetch_existing_chunk_ids(db_session, review_id)
    is_update = bool(existing_chunk_ids)

    if not has_content:
        metas = [
            ReviewChunkMeta(
                chunk_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{review_id}_0")),
                restaurant_id=restaurant_id,
                review_id=review_id,
                chunk_text=None,
                full_review=None,
                has_content=False,
                rating=rating,
                sentiment_label=sentiment_label,
                sentiment_rating_agree=_check_sentiment_rating(rating, sentiment_label),
                review_date=review_date,
                username=username,
                source=source,
                chunk_index=0,
                has_injection_attempt=False,
                date_inferred=date_inferred,
                pipeline_version=PIPELINE_VERSION,
            )
        ]
        points: list[dict[str, Any]] = []
    else:
        metas, points = _build_review_chunks(
            restaurant_id=restaurant_id,
            review_id=review_id,
            review_text=review_text,
            rating=rating,
            sentiment_label=sentiment_label,
            username=username,
            source=source,
            review_date=review_date,
            date_inferred=date_inferred,
            chunk_size=chunk_size,
            overlap=overlap,
        )

    entity_semaphore = asyncio.Semaphore(entity_concurrency)
    content_points = [p for p in points if p["text"]]
    await _extract_entities_into_points(content_points, llm_client, entity_semaphore, entity_batch)

    if points:
        texts = [p["text"] for p in points]
        dense_vectors, sparse_vectors = await asyncio.gather(
            embedder.embed(texts),
            compute_sparse_vectors_batch(texts),
        )
        qdrant_points = [
            {
                "id": p["chunk_id"],
                "vector": {"dense": dv, "sparse": {"indices": sv.indices, "values": sv.values}},
                "payload": {k: val for k, val in p.items() if k != "chunk_id"},
            }
            for p, dv, sv in zip(points, dense_vectors, sparse_vectors, strict=True)
        ]
        await vector_store.upsert(reviews_collection, qdrant_points)

    # An edit that shrinks the review's chunk count leaves the extra old
    # chunk_ids as orphans in both stores unless explicitly removed here --
    # ON CONFLICT DO UPDATE only touches chunk_ids the new version still has.
    new_chunk_ids = {m.chunk_id for m in metas}
    stale_chunk_ids = [cid for cid in existing_chunk_ids if cid not in new_chunk_ids]
    if stale_chunk_ids:
        await vector_store.delete(reviews_collection, stale_chunk_ids)

    await _upsert_chunk_meta(db_session, metas, batch_size=100)
    if stale_chunk_ids:
        await db_session.execute(
            delete(ReviewChunkMeta).where(ReviewChunkMeta.chunk_id.in_(stale_chunk_ids))
        )
    await db_session.commit()

    deleted = await cache.invalidate_restaurant(restaurant_id)
    logger.info(
        "single_review_ingested",
        restaurant_id=restaurant_id,
        review_id=review_id,
        is_update=is_update,
        chunks_written=len(metas),
        stale_chunks_removed=len(stale_chunk_ids),
        redis_keys_deleted=deleted,
    )

    status = "skipped_empty" if not has_content else ("updated" if is_update else "ingested")
    return ReviewIngestResult(review_id=review_id, status=status, chunks_written=len(metas))


def _parse_json(content: bytes) -> list[dict[str, Any]]:
    """Parse the dataset JSON file.

    The outer key is a SQL query string (not a static name like 'data').
    The value under that key is the list of review rows.
    Encoding: UTF-8 BOM (utf-8-sig).
    """
    text = content.decode("utf-8-sig")
    data: dict[str, Any] = json.loads(text)
    if not isinstance(data, dict) or not data:
        raise ValueError("JSON root must be a non-empty object")
    rows = list(data.values())[0]
    if not isinstance(rows, list):
        raise ValueError("JSON data value must be an array of review objects")
    return rows


def _correct_row(row: dict[str, Any]) -> dict[str, Any]:
    """Apply defensive defaults to every field. No row is ever discarded."""
    corrected = dict(row)
    if not corrected.get("review", "").strip():
        pass  # has_content=False handled in caller
    if not corrected.get("username"):
        corrected["username"] = "Anonymous"
    if not corrected.get("sentiment"):
        corrected["sentiment"] = "Neutral"
        logger.warning("sentiment_missing_defaulting_to_neutral", row_preview=str(row)[:80])
    if not corrected.get("source"):
        corrected["source"] = "Unknown"
    return corrected


def _parse_date(value: Any) -> tuple[datetime | None, bool]:
    """Try to parse a createdAt value. Returns (datetime, date_inferred).

    date_inferred=True means the original value was unparseable and datetime.now()
    was substituted. This flag is stored so downstream queries can caveat stale dates.
    """
    if not value:
        now = datetime.now(tz=UTC)
        logger.warning("created_at_missing_using_now")
        return now, True

    str_value = str(value).strip()

    # Try ISO variants first
    try:
        dt = datetime.fromisoformat(str_value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt, False
    except (ValueError, TypeError):
        pass

    # Try known explicit formats
    for fmt in DATE_FORMATS:
        try:
            dt = datetime.strptime(str_value, fmt)
            return dt.replace(tzinfo=UTC), False
        except ValueError:
            continue

    # Try Excel serial date (integer days since 1899-12-30)
    try:
        serial = float(str_value)
        if 1 < serial < 200000:
            from datetime import timedelta

            excel_epoch = datetime(1899, 12, 30, tzinfo=UTC)
            dt = excel_epoch + timedelta(days=serial)
            return dt, False
    except (ValueError, TypeError):
        pass

    now = datetime.now(tz=UTC)
    logger.warning("created_at_unparseable_using_now", raw_value=str_value[:50])
    return now, True


def _check_sentiment_rating(rating: float | None, sentiment_label: str) -> bool | None:
    """Return True if rating bucket matches sentiment label, False if they disagree, None if unknown."""
    if rating is None:
        return None
    bucket = RATING_TO_SENTIMENT.get(int(round(rating)))
    if bucket is None:
        return None
    normalized_sentiment = sentiment_label.strip().capitalize()
    return bucket == normalized_sentiment


def _derive_review_id(restaurant_id: int, username: str, created_at_raw: Any, row_idx: int) -> str:
    import hashlib

    raw = f"{restaurant_id}:{username}:{created_at_raw}:{row_idx}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


async def _extract_entities(
    llm_client: BaseLLMClient,
    texts: list[str],
    usage_callback: UsageCallback | None = None,
) -> list[list[str]]:
    """Extract food/menu items from a batch of review texts.

    Returns a list (one element per text) of entity lists. On any failure,
    returns empty lists for the entire batch rather than aborting the job.
    """
    if not texts:
        return []

    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    prompt = (
        f"For each numbered review text below, list only food or menu items mentioned.\n"
        f"Return a JSON array with one element per review (in order). "
        f"Each element is an array of strings. Return [] for a review with no food items.\n\n"
        f"{numbered}\n\nJSON:"
    )

    try:
        raw = await llm_client.complete(
            prompt=prompt,
            system="You extract food and menu item names from restaurant reviews. Output JSON only.",
            temperature=0.0,
            max_tokens=512,
            usage_callback=usage_callback,
        )
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
        parsed = json.loads(raw)
        if isinstance(parsed, list) and len(parsed) == len(texts):
            return [[str(e) for e in item] if isinstance(item, list) else [] for item in parsed]
    except Exception as exc:
        logger.warning("entity_extraction_failed", error=str(exc), batch_size=len(texts))

    return [[] for _ in texts]


async def _fail_job(db_session: AsyncSession, job: IngestJob, error: str) -> None:
    job.status = "failed"
    job.error_message = error[:500]
    await db_session.commit()
    logger.error("ingest_job_failed", job_id=str(job.id), error=error)
