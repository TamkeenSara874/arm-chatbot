"""Unit tests for correction lookup, storage, and its anti-poisoning guardrails."""

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.exc import IntegrityError

from src.core.correction import (
    CONSENSUS_MIN_SPAN_SECONDS,
    CONSENSUS_THRESHOLD,
    CORRECTION_COLLECTION,
    SUBMISSION_COOLDOWN_SECONDS,
    _record_vote,
    _vote_stats,
    check_stat_contradiction,
    find_correction,
    reject_correction,
    scan_for_stat_contradiction,
    session_in_cooldown,
    store_correction,
)
from src.core.review_stats import PeriodStats
from src.models.db_entities import ChatCorrection
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
    store.delete = AsyncMock()
    return store


def _make_db_session() -> MagicMock:
    session = MagicMock()
    session.get = AsyncMock(return_value=None)
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
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


class TestScanForStatContradiction:
    def test_no_claim_returns_none(self) -> None:
        assert scan_for_stat_contradiction("The staff were rude.", 3.97, 2753) is None

    def test_rating_claim_within_tolerance_returns_none(self) -> None:
        assert scan_for_stat_contradiction("We have a 4 star rating", 3.97, 2753) is None

    def test_rating_claim_contradicting_real_value_is_rejected(self) -> None:
        reason = scan_for_stat_contradiction(
            "We have a perfect 5 star rating", real_avg_rating=3.97, real_count=2753
        )
        assert reason is not None
        assert "5" in reason and "3.97" in reason

    def test_rating_claim_various_phrasings_detected(self) -> None:
        for phrase in ["5 stars", "5/5", "5 out of 5"]:
            assert scan_for_stat_contradiction(phrase, 3.97, 2753) is not None

    def test_count_claim_within_tolerance_returns_none(self) -> None:
        # 2753 * 0.15 ~= 413 tolerance
        assert scan_for_stat_contradiction("We have 2800 reviews", None, 2753) is None

    def test_count_claim_contradicting_real_value_is_rejected(self) -> None:
        reason = scan_for_stat_contradiction("We have 10000 reviews", None, 2753)
        assert reason is not None
        assert "10000" in reason and "2753" in reason

    def test_no_real_avg_rating_skips_rating_check(self) -> None:
        # Restaurant with no ratings at all yet -- nothing to contradict.
        assert scan_for_stat_contradiction("5 star rating", None, 0) is None


class TestCheckStatContradiction:
    @pytest.mark.asyncio
    async def test_delegates_to_compute_period_stats(self) -> None:
        db = MagicMock()
        with patch(
            "src.core.correction.compute_period_stats",
            new=AsyncMock(
                return_value=PeriodStats(count=2753, avg_rating=3.97, sentiment_counts={})
            ),
        ) as mock_stats:
            reason = await check_stat_contradiction("perfect 5 star rating", db, restaurant_id=1)
        mock_stats.assert_called_once_with(db, 1, date_from=None, date_to=None)
        assert reason is not None


class TestRecordVote:
    @pytest.mark.asyncio
    async def test_new_vote_returns_true(self) -> None:
        db = _make_db_session()
        result = await _record_vote(db, uuid.uuid4(), uuid.uuid4(), "1.2.3.4")
        assert result is True
        db.add.assert_called_once()
        db.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_duplicate_vote_returns_false_and_rolls_back(self) -> None:
        db = _make_db_session()
        db.commit = AsyncMock(side_effect=IntegrityError("stmt", {}, Exception("dup key")))
        result = await _record_vote(db, uuid.uuid4(), uuid.uuid4(), None)
        assert result is False
        db.rollback.assert_awaited_once()


class TestVoteStats:
    @pytest.mark.asyncio
    async def test_no_votes_returns_zero(self) -> None:
        db = MagicMock()
        exec_result = MagicMock()
        exec_result.one.return_value = (0, None, None)
        db.execute = AsyncMock(return_value=exec_result)
        count, span = await _vote_stats(db, uuid.uuid4())
        assert count == 0
        assert span == 0.0

    @pytest.mark.asyncio
    async def test_computes_distinct_count_and_span(self) -> None:
        earliest = datetime.now(tz=UTC) - timedelta(hours=2)
        latest = datetime.now(tz=UTC)
        db = MagicMock()
        exec_result = MagicMock()
        exec_result.one.return_value = (3, earliest, latest)
        db.execute = AsyncMock(return_value=exec_result)
        count, span = await _vote_stats(db, uuid.uuid4())
        assert count == 3
        assert abs(span - 7200.0) < 1.0


class TestSessionInCooldown:
    @pytest.mark.asyncio
    async def test_no_recent_vote_returns_false(self) -> None:
        db = MagicMock()
        exec_result = MagicMock()
        exec_result.scalar_one_or_none.return_value = None
        db.execute = AsyncMock(return_value=exec_result)
        assert await session_in_cooldown(db, uuid.uuid4()) is False

    @pytest.mark.asyncio
    async def test_recent_vote_returns_true(self) -> None:
        db = MagicMock()
        exec_result = MagicMock()
        exec_result.scalar_one_or_none.return_value = uuid.uuid4()
        db.execute = AsyncMock(return_value=exec_result)
        assert await session_in_cooldown(db, uuid.uuid4()) is True

    def test_cooldown_window_is_reasonably_short(self) -> None:
        # A sanity bound on the constant itself -- long enough to matter,
        # short enough not to block a legitimate user correcting a typo.
        assert 0 < SUBMISSION_COOLDOWN_SECONDS <= 300


class TestStoreCorrection:
    @pytest.mark.asyncio
    async def test_new_correction_creates_qdrant_point_and_records_vote(self) -> None:
        embedder = _make_embedder()
        store = _make_vector_store(search_results=[])
        db = _make_db_session()

        with (
            patch(
                "src.core.correction._record_vote", new=AsyncMock(return_value=True)
            ) as vote_mock,
            patch("src.core.correction._vote_stats", new=AsyncMock(return_value=(1, 0.0))),
        ):
            correction_id, is_consensus = await store_correction(
                session_id=uuid.uuid4(),
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
        assert store.upsert.call_args[0][0] == CORRECTION_COLLECTION
        vote_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_matching_existing_correction_records_vote_against_it_not_a_new_point(
        self,
    ) -> None:
        existing_id = str(uuid.uuid4())
        existing = SearchResult(
            id=existing_id,
            score=0.95,
            payload={"restaurant_id": 1, "intent": "best_item", "corrected_response": "Old"},
        )
        embedder = _make_embedder()
        store = _make_vector_store(search_results=[existing])
        db = _make_db_session()

        with (
            patch("src.core.correction._record_vote", new=AsyncMock(return_value=True)),
            patch("src.core.correction._vote_stats", new=AsyncMock(return_value=(2, 100.0))),
        ):
            correction_id, _ = await store_correction(
                session_id=uuid.uuid4(),
                restaurant_id=1,
                original_query="What is best?",
                original_response="I don't know",
                corrected_response="New correction",
                intent="best_item",
                embedder=embedder,
                vector_store=store,
                db_session=db,
            )

        assert str(correction_id) == existing_id
        store.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_repeat_vote_from_same_session_does_not_update_stored_text(self) -> None:
        """vote_is_new=False (this session already voted) must not let a
        resubmission silently swap in different wording while riding on
        other sessions' vote count."""
        existing_id = str(uuid.uuid4())
        existing = SearchResult(
            id=existing_id,
            score=0.95,
            payload={"restaurant_id": 1, "intent": "best_item", "corrected_response": "Old"},
        )
        embedder = _make_embedder()
        store = _make_vector_store(search_results=[existing])
        db = _make_db_session()

        with (
            patch("src.core.correction._record_vote", new=AsyncMock(return_value=False)),
            patch("src.core.correction._vote_stats", new=AsyncMock(return_value=(1, 0.0))),
        ):
            await store_correction(
                session_id=uuid.uuid4(),
                restaurant_id=1,
                original_query="What is best?",
                original_response="I don't know",
                corrected_response="Attacker-controlled text",
                intent="best_item",
                embedder=embedder,
                vector_store=store,
                db_session=db,
            )

        # update_payload is still called once at the end to sync
        # correction_count/is_consensus, but never with corrected_response.
        for call in store.update_payload.call_args_list:
            assert "corrected_response" not in call[0][2]

    @pytest.mark.asyncio
    async def test_consensus_requires_both_distinct_sessions_and_min_span(self) -> None:
        embedder = _make_embedder()
        store = _make_vector_store(search_results=[])
        db = _make_db_session()

        with (
            patch("src.core.correction._record_vote", new=AsyncMock(return_value=True)),
            patch(
                "src.core.correction._vote_stats",
                new=AsyncMock(return_value=(CONSENSUS_THRESHOLD, CONSENSUS_MIN_SPAN_SECONDS - 1)),
            ),
        ):
            _, is_consensus = await store_correction(
                session_id=uuid.uuid4(),
                restaurant_id=1,
                original_query="query",
                original_response="old",
                corrected_response="corrected",
                intent="factual",
                embedder=embedder,
                vector_store=store,
                db_session=db,
            )

        assert is_consensus is False, (
            "3 distinct sessions within too short a window must not reach consensus -- "
            "otherwise scripting 3 sessions back to back still trivially poisons it."
        )

    @pytest.mark.asyncio
    async def test_consensus_reached_with_enough_sessions_and_span(self) -> None:
        embedder = _make_embedder()
        store = _make_vector_store(search_results=[])
        db = _make_db_session()

        with (
            patch("src.core.correction._record_vote", new=AsyncMock(return_value=True)),
            patch(
                "src.core.correction._vote_stats",
                new=AsyncMock(return_value=(CONSENSUS_THRESHOLD, CONSENSUS_MIN_SPAN_SECONDS + 1)),
            ),
        ):
            _, is_consensus = await store_correction(
                session_id=uuid.uuid4(),
                restaurant_id=1,
                original_query="query",
                original_response="old",
                corrected_response="corrected",
                intent="factual",
                embedder=embedder,
                vector_store=store,
                db_session=db,
            )

        assert is_consensus is True
        payload = store.update_payload.call_args[0][2]
        assert payload["is_consensus"] is True


class TestRejectCorrection:
    @pytest.mark.asyncio
    async def test_rejects_and_deletes_qdrant_point(self) -> None:
        correction_id = uuid.uuid4()
        row = ChatCorrection(
            id=correction_id,
            qdrant_point_id=str(correction_id),
            restaurant_id=1,
            original_query="q",
            original_response="a",
            corrected_response="c",
            correction_count=3,
            is_consensus=True,
        )
        db = MagicMock()
        db.get = AsyncMock(return_value=row)
        db.commit = AsyncMock()
        store = _make_vector_store()

        ok = await reject_correction(db, store, correction_id, restaurant_id=1)

        assert ok is True
        assert row.is_rejected is True
        assert row.is_consensus is False
        store.delete.assert_awaited_once_with(CORRECTION_COLLECTION, [str(correction_id)])

    @pytest.mark.asyncio
    async def test_not_found_returns_false(self) -> None:
        db = MagicMock()
        db.get = AsyncMock(return_value=None)
        store = _make_vector_store()
        ok = await reject_correction(db, store, uuid.uuid4(), restaurant_id=1)
        assert ok is False
        store.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_wrong_restaurant_returns_false(self) -> None:
        correction_id = uuid.uuid4()
        row = ChatCorrection(
            id=correction_id,
            qdrant_point_id=str(correction_id),
            restaurant_id=2,
            original_query="q",
            original_response="a",
            corrected_response="c",
        )
        db = MagicMock()
        db.get = AsyncMock(return_value=row)
        store = _make_vector_store()

        ok = await reject_correction(db, store, correction_id, restaurant_id=1)

        assert ok is False
        store.delete.assert_not_called()
