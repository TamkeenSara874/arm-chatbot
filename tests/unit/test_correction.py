"""Unit tests for correction lookup and storage."""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.correction import (
    CONSENSUS_THRESHOLD,
    CORRECTION_COLLECTION,
    find_correction,
    store_correction,
)
from src.services.vector.base import SearchResult


def _make_embedder(vector: list[float] | None = None) -> MagicMock:
    embedder = MagicMock()
    embedder.embed_one = AsyncMock(return_value=vector or [0.1] * 3072)
    return embedder


def _make_vector_store(search_results: list[SearchResult] | None = None) -> MagicMock:
    store = MagicMock()
    store.search = AsyncMock(return_value=search_results or [])
    store.upsert = AsyncMock()
    store.update_payload = AsyncMock()
    return store


def _make_db_session() -> MagicMock:
    session = MagicMock()
    session.get = AsyncMock(return_value=None)
    session.add = MagicMock()
    session.commit = AsyncMock()
    return session


class TestFindCorrection:
    @pytest.mark.asyncio
    async def test_returns_none_when_no_match(self) -> None:
        embedder = _make_embedder()
        store = _make_vector_store(search_results=[])
        result = await find_correction("query", 1, "factual", embedder, store)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_corrected_response_on_match(self) -> None:
        sr = SearchResult(
            id="c1",
            score=0.92,
            payload={
                "restaurant_id": 1,
                "intent": "factual",
                "corrected_response": "The best dish is biryani.",
            },
        )
        embedder = _make_embedder()
        store = _make_vector_store(search_results=[sr])
        result = await find_correction("What is best?", 1, "factual", embedder, store)
        assert result.text == "The best dish is biryani."

    @pytest.mark.asyncio
    async def test_missing_is_consensus_in_payload_defaults_false(self) -> None:
        sr = SearchResult(
            id="c1",
            score=0.92,
            payload={
                "restaurant_id": 1,
                "intent": "factual",
                "corrected_response": "The best dish is biryani.",
            },
        )
        embedder = _make_embedder()
        store = _make_vector_store(search_results=[sr])
        result = await find_correction("What is best?", 1, "factual", embedder, store)
        assert result.is_consensus is False

    @pytest.mark.asyncio
    async def test_is_consensus_true_is_surfaced_from_payload(self) -> None:
        sr = SearchResult(
            id="c1",
            score=0.92,
            payload={
                "restaurant_id": 1,
                "intent": "factual",
                "corrected_response": "The food is now excellent after a menu overhaul.",
                "is_consensus": True,
            },
        )
        embedder = _make_embedder()
        store = _make_vector_store(search_results=[sr])
        result = await find_correction("How is the food?", 1, "factual", embedder, store)
        assert result.is_consensus is True
        assert result.text == "The food is now excellent after a menu overhaul."

    @pytest.mark.asyncio
    async def test_intent_mismatch_is_skipped(self) -> None:
        sr = SearchResult(
            id="c1",
            score=0.95,
            payload={
                "restaurant_id": 1,
                "intent": "best_item",
                "corrected_response": "Should not be returned",
            },
        )
        embedder = _make_embedder()
        store = _make_vector_store(search_results=[sr])
        result = await find_correction("How to improve?", 1, "improvement", embedder, store)
        assert result is None

    @pytest.mark.asyncio
    async def test_no_intent_in_payload_passes_through(self) -> None:
        sr = SearchResult(
            id="c1",
            score=0.90,
            payload={
                "restaurant_id": 1,
                "corrected_response": "This has no intent field.",
            },
        )
        embedder = _make_embedder()
        store = _make_vector_store(search_results=[sr])
        result = await find_correction("Some query", 1, "factual", embedder, store)
        assert result.text == "This has no intent field."

    @pytest.mark.asyncio
    async def test_lookup_failure_returns_none(self) -> None:
        embedder = _make_embedder()
        store = MagicMock()
        store.search = AsyncMock(side_effect=RuntimeError("Qdrant down"))
        result = await find_correction("query", 1, "factual", embedder, store)
        assert result is None


class TestStoreCorrection:
    @pytest.mark.asyncio
    async def test_new_correction_creates_qdrant_point(self) -> None:
        embedder = _make_embedder()
        store = _make_vector_store(search_results=[])
        db = _make_db_session()

        correction_id, is_consensus = await store_correction(
            session_id=None,
            restaurant_id=1,
            original_query="What is best?",
            original_response="I don't know",
            corrected_response="Biryani is best",
            intent="best_item",
            embedder=embedder,
            vector_store=store,
            db_session=db,
        )

        assert isinstance(correction_id, uuid.UUID)
        assert is_consensus is False
        store.upsert.assert_called_once()
        upsert_args = store.upsert.call_args[0]
        assert upsert_args[0] == CORRECTION_COLLECTION

    @pytest.mark.asyncio
    async def test_duplicate_correction_increments_count(self) -> None:
        existing = SearchResult(
            id=str(uuid.uuid4()),
            score=0.95,
            payload={
                "restaurant_id": 1,
                "intent": "best_item",
                "corrected_response": "Old correction",
                "correction_count": 2,
                "is_consensus": False,
            },
        )
        embedder = _make_embedder()
        store = _make_vector_store(search_results=[existing])
        db = _make_db_session()

        correction_id, is_consensus = await store_correction(
            session_id=None,
            restaurant_id=1,
            original_query="What is best?",
            original_response="I don't know",
            corrected_response="New correction",
            intent="best_item",
            embedder=embedder,
            vector_store=store,
            db_session=db,
        )

        store.update_payload.assert_called_once()
        payload = store.update_payload.call_args[0][2]
        assert payload["correction_count"] == 3

    @pytest.mark.asyncio
    async def test_consensus_reached_at_threshold(self) -> None:
        existing = SearchResult(
            id=str(uuid.uuid4()),
            score=0.95,
            payload={
                "restaurant_id": 1,
                "intent": "factual",
                "corrected_response": "Old",
                "correction_count": CONSENSUS_THRESHOLD - 1,
                "is_consensus": False,
            },
        )
        embedder = _make_embedder()
        store = _make_vector_store(search_results=[existing])
        db = _make_db_session()

        _, is_consensus = await store_correction(
            session_id=None,
            restaurant_id=1,
            original_query="query",
            original_response="old response",
            corrected_response="corrected",
            intent="factual",
            embedder=embedder,
            vector_store=store,
            db_session=db,
        )

        assert is_consensus is True
        payload = store.update_payload.call_args[0][2]
        assert payload["is_consensus"] is True
