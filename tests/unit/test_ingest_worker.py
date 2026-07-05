"""Unit tests for src/workers/ingest_worker.py -- resumability (skip-if-already-
processed, per-batch incremental Postgres writes) and the single-review live
ingestion path. Previously zero direct test coverage existed for this module.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.db_entities import IngestJob, ReviewChunkMeta
from src.services.embedding.sparse_embedder import SparseVector
from src.workers.ingest_worker import (
    PIPELINE_VERSION,
    _build_review_chunks,
    _derive_review_id,
    _fetch_existing_chunk_ids,
    _fetch_processed_review_ids,
    _process_rows,
    _upsert_chunk_meta,
    ingest_single_review,
)


def _mock_settings() -> MagicMock:
    return MagicMock(
        chunk_size_tokens=256,
        chunk_overlap_tokens=32,
        ingest_batch_size=100,
        entity_extraction_batch_size=10,
        entity_extraction_concurrency=8,
    )


def _mock_embedder(dim: int = 8) -> MagicMock:
    embedder = MagicMock()
    embedder.model = "text-embedding-3-small"
    embedder.embed = AsyncMock(
        side_effect=lambda texts, usage_callback=None: [[0.1] * dim] * len(texts)
    )
    return embedder


def _mock_llm_client() -> MagicMock:
    client = MagicMock()
    client.model = "gpt-4o-mini"
    client.complete = AsyncMock(return_value="[]")
    return client


def _mock_vector_store() -> MagicMock:
    store = MagicMock()
    store.upsert = AsyncMock()
    store.delete = AsyncMock()
    return store


def _mock_db_session() -> MagicMock:
    session = MagicMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    return session


def _mock_cache() -> MagicMock:
    cache = MagicMock()
    cache.invalidate_restaurant = AsyncMock(return_value=0)
    return cache


def _sparse_patch():
    sv = SparseVector(indices=[0, 1], values=[0.5, 0.5])
    return patch(
        "src.workers.ingest_worker.compute_sparse_vectors_batch",
        AsyncMock(side_effect=lambda texts: [sv] * len(texts)),
    )


class TestDeriveReviewId:
    def test_same_inputs_produce_same_id(self) -> None:
        a = _derive_review_id(1, "alice", "2024-01-01", 0)
        b = _derive_review_id(1, "alice", "2024-01-01", 0)
        assert a == b

    def test_different_row_idx_produces_different_id(self) -> None:
        a = _derive_review_id(1, "alice", "2024-01-01", 0)
        b = _derive_review_id(1, "alice", "2024-01-01", 1)
        assert a != b


class TestBuildReviewChunks:
    def test_short_review_produces_one_chunk_with_pipeline_version(self) -> None:
        metas, points = _build_review_chunks(
            restaurant_id=1,
            review_id="r1",
            review_text="Great food, will come back.",
            rating=5.0,
            sentiment_label="Positive",
            username="alice",
            source="Google",
            review_date=None,
            date_inferred=False,
            chunk_size=256,
            overlap=32,
        )
        assert len(metas) == 1
        assert len(points) == 1
        assert metas[0].pipeline_version == PIPELINE_VERSION
        assert metas[0].chunk_index == 0
        assert points[0]["text"] == "Great food, will come back."


class TestFetchProcessedReviewIds:
    @pytest.mark.asyncio
    async def test_empty_review_ids_short_circuits_without_query(self) -> None:
        db = _mock_db_session()
        result = await _fetch_processed_review_ids(db, 1, [], PIPELINE_VERSION)
        assert result == set()
        db.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_review_ids_from_query_result(self) -> None:
        db = _mock_db_session()
        exec_result = MagicMock()
        exec_result.all.return_value = [("r1",), ("r2",)]
        db.execute.return_value = exec_result
        result = await _fetch_processed_review_ids(db, 1, ["r1", "r2", "r3"], PIPELINE_VERSION)
        assert result == {"r1", "r2"}


class TestFetchExistingChunkIds:
    @pytest.mark.asyncio
    async def test_returns_chunk_ids_from_query_result(self) -> None:
        db = _mock_db_session()
        exec_result = MagicMock()
        exec_result.all.return_value = [("c1",), ("c2",)]
        db.execute.return_value = exec_result
        result = await _fetch_existing_chunk_ids(db, "r1")
        assert result == ["c1", "c2"]


class TestUpsertChunkMeta:
    @pytest.mark.asyncio
    async def test_empty_list_is_noop(self) -> None:
        db = _mock_db_session()
        await _upsert_chunk_meta(db, [], batch_size=100)
        db.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_batches_beyond_batch_size(self) -> None:
        db = _mock_db_session()
        metas = [
            ReviewChunkMeta(
                chunk_id=f"c{i}",
                restaurant_id=1,
                review_id="r1",
                chunk_text="text",
                has_content=True,
                chunk_index=i,
                pipeline_version=PIPELINE_VERSION,
            )
            for i in range(5)
        ]
        await _upsert_chunk_meta(db, metas, batch_size=2)
        assert db.execute.call_count == 3  # ceil(5/2)


class TestIngestSingleReview:
    @pytest.mark.asyncio
    async def test_new_review_is_ingested(self) -> None:
        db = _mock_db_session()
        exec_result = MagicMock()
        exec_result.all.return_value = []  # no existing chunks
        db.execute.return_value = exec_result
        vector_store = _mock_vector_store()

        with _sparse_patch():
            result = await ingest_single_review(
                restaurant_id=1,
                external_review_id="ext-1",
                review_text="The food was great and the service was fast.",
                rating=5.0,
                username="alice",
                source="Google",
                created_at_raw="2024-01-01",
                sentiment_label="Positive",
                settings=_mock_settings(),
                db_session=db,
                embedder=_mock_embedder(),
                vector_store=vector_store,
                reviews_collection="review_chunks",
                llm_client=_mock_llm_client(),
                cache=_mock_cache(),
            )

        assert result.status == "ingested"
        assert result.chunks_written >= 1
        vector_store.upsert.assert_called_once()
        vector_store.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_review_text_is_skipped_empty(self) -> None:
        db = _mock_db_session()
        exec_result = MagicMock()
        exec_result.all.return_value = []
        db.execute.return_value = exec_result
        vector_store = _mock_vector_store()

        with _sparse_patch():
            result = await ingest_single_review(
                restaurant_id=1,
                external_review_id="ext-2",
                review_text="   ",
                rating=None,
                username=None,
                source=None,
                created_at_raw=None,
                sentiment_label=None,
                settings=_mock_settings(),
                db_session=db,
                embedder=_mock_embedder(),
                vector_store=vector_store,
                reviews_collection="review_chunks",
                llm_client=_mock_llm_client(),
                cache=_mock_cache(),
            )

        assert result.status == "skipped_empty"
        vector_store.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_repeat_call_same_external_id_is_update(self) -> None:
        db = _mock_db_session()
        # Pretend this review_id already has one chunk written.
        with (
            patch(
                "src.workers.ingest_worker._fetch_existing_chunk_ids",
                AsyncMock(return_value=["existing-chunk-0"]),
            ),
            _sparse_patch(),
        ):
            result = await ingest_single_review(
                restaurant_id=1,
                external_review_id="ext-3",
                review_text="Updated review text about the food.",
                rating=4.0,
                username="bob",
                source="Yelp",
                created_at_raw="2024-02-01",
                sentiment_label="Positive",
                settings=_mock_settings(),
                db_session=db,
                embedder=_mock_embedder(),
                vector_store=_mock_vector_store(),
                reviews_collection="review_chunks",
                llm_client=_mock_llm_client(),
                cache=_mock_cache(),
            )
        assert result.status == "updated"

    @pytest.mark.asyncio
    async def test_shrinking_review_deletes_stale_chunks(self) -> None:
        """An edit that produces fewer chunks than before must delete the
        leftover chunk_ids from both Qdrant and Postgres, not just leave them
        as orphans that ON CONFLICT DO UPDATE never touches."""
        db = _mock_db_session()
        vector_store = _mock_vector_store()

        # Simulate 3 pre-existing chunks; the new (short) text will produce
        # only 1 chunk, so chunk indices 1 and 2 should be deleted.
        review_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "1:ext-4"))
        stale_ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{review_id}_1")),
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{review_id}_2")),
        ]
        kept_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{review_id}_0"))

        with (
            patch(
                "src.workers.ingest_worker._fetch_existing_chunk_ids",
                AsyncMock(return_value=[kept_id, *stale_ids]),
            ),
            _sparse_patch(),
        ):
            result = await ingest_single_review(
                restaurant_id=1,
                external_review_id="ext-4",
                review_text="Short edit.",
                rating=4.0,
                username="carol",
                source="Google",
                created_at_raw="2024-03-01",
                sentiment_label="Positive",
                settings=_mock_settings(),
                db_session=db,
                embedder=_mock_embedder(),
                vector_store=vector_store,
                reviews_collection="review_chunks",
                llm_client=_mock_llm_client(),
                cache=_mock_cache(),
            )

        assert result.status == "updated"
        vector_store.delete.assert_called_once()
        deleted_ids = vector_store.delete.call_args[0][1]
        assert set(deleted_ids) == set(stale_ids)


class TestProcessRowsResumability:
    @pytest.mark.asyncio
    async def test_already_processed_review_is_skipped_before_llm_or_embed(self) -> None:
        rows = [
            {
                "review": "Already handled review text.",
                "username": "alice",
                "createdAt": "2024-01-01",
                "rating": 5,
                "sentiment": "Positive",
                "source": "Google",
            },
            {
                "review": "Brand new review text.",
                "username": "bob",
                "createdAt": "2024-01-02",
                "rating": 4,
                "sentiment": "Positive",
                "source": "Yelp",
            },
        ]
        already_processed_id = _derive_review_id(1, "alice", "2024-01-01", 0)

        db = _mock_db_session()
        job = IngestJob(id=uuid.uuid4(), restaurant_id=1, filename="test.json", status="processing")
        llm_client = _mock_llm_client()
        vector_store = _mock_vector_store()

        with (
            patch(
                "src.workers.ingest_worker._fetch_processed_review_ids",
                AsyncMock(return_value={already_processed_id}),
            ),
            _sparse_patch(),
        ):
            await _process_rows(
                rows=rows,
                restaurant_id=1,
                settings=_mock_settings(),
                db_session=db,
                job=job,
                embedder=_mock_embedder(),
                vector_store=vector_store,
                reviews_collection="review_chunks",
                llm_client=llm_client,
                cache=_mock_cache(),
            )

        assert job.skipped_already_processed == 1
        # Only the non-skipped row's text should ever reach the entity
        # extraction LLM call.
        all_prompts = " ".join(
            str(call.kwargs.get("prompt", call.args[0] if call.args else ""))
            for call in llm_client.complete.call_args_list
        )
        assert "Already handled review text" not in all_prompts
        assert "Brand new review text" in all_prompts
        assert job.status == "complete"
