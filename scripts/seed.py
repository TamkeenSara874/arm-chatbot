"""Seed the vector store from the ARM review export.

Runs automatically as a docker-compose service before the API starts. Safe to
re-run at any time: exits immediately if the collection is already current.

Re-ingestion is triggered by either:
  - The content of dataset/dataset.json changing (SHA-256 hash mismatch)
  - PIPELINE_VERSION in ingest_worker.py being bumped (model/chunking change)

Transient failures (rate limits, network blips) are retried automatically up to
MAX_ATTEMPTS times with exponential backoff. Qdrant upserts use deterministic
point IDs so restarting mid-ingest is always safe.

To force a full re-seed without touching the file:
  docker compose run --rm seed python scripts/seed.py --force
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import sys
import uuid
from pathlib import Path

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from src.config import get_settings
from src.models.db_entities import IngestJob, IngestManifest, RestaurantCredential
from src.services.cache import RedisCache
from src.services.embedding.factory import create_embedder
from src.services.llm.factory import create_simple_client
from src.services.vector.factory import create_vector_store
from src.services.vector.qdrant_store import ensure_collections
from src.utils.logging import configure_logging
from src.utils.restaurant_auth import generate_restaurant_key, hash_restaurant_key
from src.workers.ingest_worker import PIPELINE_VERSION, run_ingest_job

DATASET_PATH = Path(__file__).parent.parent / "dataset" / "dataset.json"
RESTAURANT_ID = 1

MAX_ATTEMPTS = 3
RETRY_BASE_SECONDS = 60  # 60s, 120s between attempts — matches typical rate-limit windows

logger = structlog.get_logger()


class SeedError(RuntimeError):
    pass


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def _is_current(session: AsyncSession, collection: str, current_hash: str) -> bool:
    manifest = await session.scalar(
        select(IngestManifest).where(IngestManifest.collection_name == collection)
    )
    if manifest is None:
        return False
    return manifest.file_hash == current_hash and manifest.pipeline_version == PIPELINE_VERSION


async def _save_manifest(
    session: AsyncSession,
    collection: str,
    file_hash: str,
    review_count: int,
) -> None:
    manifest = await session.scalar(
        select(IngestManifest).where(IngestManifest.collection_name == collection)
    )
    if manifest is None:
        manifest = IngestManifest(
            collection_name=collection,
            file_hash=file_hash,
            pipeline_version=PIPELINE_VERSION,
            review_count=review_count,
        )
        session.add(manifest)
    else:
        manifest.file_hash = file_hash
        manifest.pipeline_version = PIPELINE_VERSION
        manifest.review_count = review_count
    await session.commit()


async def _run_once(
    file_bytes: bytes,
    current_hash: str,
    collection: str,
    settings,
    engine,
) -> int:
    """Run one full ingest attempt. Returns review_count on success, raises SeedError on failure."""
    embedder = create_embedder(settings)
    vector_store = create_vector_store(settings)
    llm_client = create_simple_client(settings)
    cache = RedisCache(settings.redis_url, settings.cache_ttl_seconds)

    async with AsyncSession(engine) as session:
        job = IngestJob(
            id=uuid.uuid4(),
            restaurant_id=RESTAURANT_ID,
            filename=DATASET_PATH.name,
            status="pending",
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)

    await run_ingest_job(
        job_id=job.id,
        restaurant_id=RESTAURANT_ID,
        file_content=file_bytes,
        settings=settings,
        embedder=embedder,
        vector_store=vector_store,
        reviews_collection=collection,
        llm_client=llm_client,
        cache=cache,
    )

    async with AsyncSession(engine) as session:
        updated_job = await session.get(IngestJob, job.id)
        if updated_job.status == "failed":
            raise SeedError(updated_job.error_message or "ingest_worker reported failure")
        return updated_job.total_reviews or 0


async def _ensure_dev_restaurant_credential(restaurant_id: int) -> None:
    """Create a restaurant_credential row for local dev if one doesn't exist yet.

    Deliberately create-if-missing, not overwrite-on-every-run: unlike
    scripts/create_restaurant_credential.py (which reissues on demand), this
    runs on every `docker compose up`, and rotating the key each time would
    invalidate whatever a developer already has configured in their local
    frontend .env.
    """
    settings = get_settings()
    engine = create_async_engine(settings.database_url)
    async with AsyncSession(engine) as session:
        existing = await session.get(RestaurantCredential, restaurant_id)
        if existing is not None:
            await engine.dispose()
            return
        key = generate_restaurant_key()
        session.add(
            RestaurantCredential(restaurant_id=restaurant_id, key_hash=hash_restaurant_key(key))
        )
        await session.commit()
    await engine.dispose()
    logger.info(
        "dev_restaurant_credential_created",
        restaurant_id=restaurant_id,
        restaurant_key=key,
        note="Set this as VITE_RESTAURANT_KEY in frontend/.env for local login",
    )


async def seed(force: bool = False) -> None:
    settings = get_settings()
    configure_logging(log_level=settings.log_level, debug=settings.debug)

    if not DATASET_PATH.exists():
        logger.error("dataset_not_found", path=str(DATASET_PATH))
        sys.exit(1)

    file_bytes = DATASET_PATH.read_bytes()
    current_hash = _file_hash(DATASET_PATH)
    collection = settings.qdrant_collection_reviews

    # Qdrant collections must exist before any upsert. This service can start
    # before the backend (whose lifespan normally creates them), so create
    # them here too -- idempotent, safe if the backend already did it.
    await ensure_collections(settings)

    await _ensure_dev_restaurant_credential(RESTAURANT_ID)

    engine = create_async_engine(settings.database_url)

    async with AsyncSession(engine) as session:
        if not force and await _is_current(session, collection, current_hash):
            logger.info(
                "seed_skip",
                reason="file_hash and pipeline_version unchanged",
                collection=collection,
            )
            await engine.dispose()
            return

    logger.info(
        "seed_start",
        collection=collection,
        pipeline_version=PIPELINE_VERSION,
        file_hash=current_hash[:12],
        forced=force,
        max_attempts=MAX_ATTEMPTS,
    )

    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            review_count = await _run_once(
                file_bytes, current_hash, collection, settings, engine
            )
            break
        except Exception as exc:
            last_error = exc
            if attempt == MAX_ATTEMPTS:
                logger.error(
                    "seed_all_attempts_failed",
                    attempts=MAX_ATTEMPTS,
                    error=str(exc),
                )
                await engine.dispose()
                sys.exit(1)

            delay = RETRY_BASE_SECONDS * attempt
            logger.warning(
                "seed_attempt_failed",
                attempt=attempt,
                retry_in_seconds=delay,
                error=str(exc),
            )
            await asyncio.sleep(delay)
    else:
        # Should never reach here given sys.exit above, but satisfies type checker.
        await engine.dispose()
        sys.exit(1)

    async with AsyncSession(engine) as session:
        await _save_manifest(session, collection, current_hash, review_count)

    logger.info("seed_complete", reviews=review_count, collection=collection)
    await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest even if hash and pipeline version match",
    )
    args = parser.parse_args()
    asyncio.run(seed(force=args.force))
