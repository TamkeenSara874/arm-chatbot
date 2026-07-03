from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import UTC, datetime
from typing import Any

import structlog
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

    Runs as asyncio.create_task() -- never use FastAPI BackgroundTasks for this
    because BackgroundTasks blocks the event loop during heavy embedding I/O.
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
    """Chunk, extract entities, embed, and upsert every row. Raises on failure

    so the caller can mark the job failed with a real error message instead of
    leaving it stuck at status=processing.
    """
    chunk_size: int = getattr(settings, "chunk_size_tokens", 256)
    overlap: int = getattr(settings, "chunk_overlap_tokens", 32)
    batch_size: int = getattr(settings, "ingest_batch_size", 100)
    entity_batch: int = getattr(settings, "entity_extraction_batch_size", 10)
    entity_concurrency: int = getattr(settings, "entity_extraction_concurrency", 8)

    trace = IngestTrace(job_id=str(job.id), restaurant_id=restaurant_id)

    all_meta: list[ReviewChunkMeta] = []
    all_points: list[dict] = []
    skipped_empty = 0

    for row_idx, raw_row in enumerate(rows):
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

        sentiment_rating_agree = _check_sentiment_rating(rating, sentiment_label)

        review_id = _derive_review_id(restaurant_id, username, created_at_raw, row_idx)

        if not has_content:
            skipped_empty += 1
            ingest_reviews_total.labels(status="skipped_empty").inc()
            meta = ReviewChunkMeta(
                chunk_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{review_id}_0")),
                restaurant_id=restaurant_id,
                review_id=review_id,
                chunk_text=None,
                full_review=None,
                has_content=False,
                rating=rating,
                sentiment_label=sentiment_label,
                sentiment_rating_agree=sentiment_rating_agree,
                review_date=review_date,
                username=username,
                source=source,
                chunk_index=0,
                has_injection_attempt=False,
                date_inferred=date_inferred,
            )
            all_meta.append(meta)
            continue

        chunks = chunk_text(review_text, chunk_size=chunk_size, overlap_tokens=overlap)

        for chunk_idx, chunk in enumerate(chunks):
            has_injection = flag_injection(chunk, restaurant_id)
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{review_id}_{chunk_idx}"))
            meta = ReviewChunkMeta(
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
            )
            all_meta.append(meta)
            all_points.append(
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

        ingest_reviews_total.labels(status="ingested").inc()

        progress = int(((row_idx + 1) / len(rows)) * 70)
        job.progress_pct = progress
        if (row_idx + 1) % batch_size == 0:
            await db_session.commit()

    # Entity extraction (batched LLM calls, bounded concurrency).
    # ~275 batches run one-at-a-time would serialize the whole job behind
    # sequential network round-trips; a semaphore caps concurrent OpenAI
    # calls without overwhelming rate limits.
    content_points = [p for p in all_points if p["text"]]
    entity_semaphore = asyncio.Semaphore(entity_concurrency)
    entity_model_name = getattr(llm_client, "model", "unknown")

    async def _extract_batch(batch: list[dict]) -> None:
        async with entity_semaphore:
            entities_per_chunk = await _extract_entities(
                llm_client,
                [p["text"] for p in batch],
                usage_callback=lambda p, c: trace.record_entity_tokens(entity_model_name, p, c),
            )
        for point, entities in zip(batch, entities_per_chunk, strict=True):
            point["food_entities"] = entities

    t_entities = time.perf_counter()
    await asyncio.gather(
        *(
            _extract_batch(content_points[i : i + entity_batch])
            for i in range(0, len(content_points), entity_batch)
        )
    )
    trace.entity_extraction_ms = (time.perf_counter() - t_entities) * 1000.0

    # Embedding and Qdrant upsert (batched) with named dense + sparse vectors
    embedding_model_name = getattr(embedder, "model", "unknown")
    t_embed = time.perf_counter()
    for i in range(0, len(all_points), batch_size):
        batch = all_points[i : i + batch_size]
        texts = [p["text"] for p in batch]
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
            for p, dv, sv in zip(batch, dense_vectors, sparse_vectors, strict=True)
        ]
        await vector_store.upsert(reviews_collection, qdrant_points)

        progress = 70 + int(((i + len(batch)) / max(len(all_points), 1)) * 20)
        job.progress_pct = min(progress, 90)
        await db_session.commit()

    # Postgres bulk upsert of ReviewChunkMeta. Uses ON CONFLICT DO UPDATE on the
    # chunk_id primary key so retrying a job (after a transient failure partway
    # through) is safe and doesn't collide with rows a prior attempt already
    # committed -- matching the idempotency Qdrant already gets from
    # deterministic point IDs.
    if all_meta:
        meta_columns = [
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
        ]
        rows = [{col: getattr(meta, col) for col in meta_columns} for meta in all_meta]
        # Postgres/asyncpg cap query parameters at 32767; with 15 columns per
        # row that's ~2184 rows max in one statement, so chunk well under
        # that instead of sending the whole dataset in a single INSERT.
        for i in range(0, len(rows), batch_size):
            row_chunk = rows[i : i + batch_size]
            stmt = pg_insert(ReviewChunkMeta).values(row_chunk)
            update_cols = {col: stmt.excluded[col] for col in meta_columns if col != "chunk_id"}
            stmt = stmt.on_conflict_do_update(index_elements=["chunk_id"], set_=update_cols)
            await db_session.execute(stmt)

    trace.embedding_upsert_ms = (time.perf_counter() - t_embed) * 1000.0

    job.total_chunks = len(all_points)
    job.skipped_empty = skipped_empty
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
        total_chunks=len(all_points),
        skipped_empty=skipped_empty,
    )

    trace.total_reviews = len(rows)
    trace.total_chunks = len(all_points)
    trace.skipped_empty = skipped_empty
    trace.emit()


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
