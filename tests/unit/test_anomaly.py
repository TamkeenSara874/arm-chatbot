"""Unit tests for src/core/anomaly.py -- proactive rating/sentiment-drop detection.

Compares a trailing recent window against the prior baseline window of the
same length via src.core.review_stats.compute_period_stats (mocked here the
same way tests/unit/test_review_stats.py mocks it), then a thin Redis-cached
wrapper avoids recomputing the two Postgres aggregates on every poll.
"""

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.anomaly import (
    MIN_REVIEWS_FOR_SIGNAL,
    AnomalyResult,
    detect_anomaly,
    get_anomaly_status,
)
from src.core.review_stats import PeriodStats


def _db_returning(recent_count_avg, recent_sentiment, baseline_count_avg, baseline_sentiment):
    """Build a mocked AsyncSession whose .execute() calls satisfy, in order,
    compute_period_stats's two queries for the recent window then the two
    queries for the baseline window (count+avg, then sentiment breakdown, each)."""
    recent_count_result = MagicMock()
    recent_count_result.one.return_value = recent_count_avg
    recent_sentiment_result = MagicMock()
    recent_sentiment_result.all.return_value = recent_sentiment
    baseline_count_result = MagicMock()
    baseline_count_result.one.return_value = baseline_count_avg
    baseline_sentiment_result = MagicMock()
    baseline_sentiment_result.all.return_value = baseline_sentiment

    db = MagicMock()
    db.execute = AsyncMock(
        side_effect=[
            recent_count_result,
            recent_sentiment_result,
            baseline_count_result,
            baseline_sentiment_result,
        ]
    )
    return db


class TestDetectAnomaly:
    @pytest.mark.asyncio
    async def test_not_enough_recent_reviews_does_not_detect(self) -> None:
        db = _db_returning((MIN_REVIEWS_FOR_SIGNAL - 1, 3.0), [], (10, 4.5), [])
        result = await detect_anomaly(db, restaurant_id=1)
        assert result.detected is False

    @pytest.mark.asyncio
    async def test_not_enough_baseline_reviews_does_not_detect(self) -> None:
        db = _db_returning((10, 3.0), [], (MIN_REVIEWS_FOR_SIGNAL - 1, 4.5), [])
        result = await detect_anomaly(db, restaurant_id=1)
        assert result.detected is False

    @pytest.mark.asyncio
    async def test_large_rating_drop_detected(self) -> None:
        db = _db_returning(
            (10, 3.5), [("Negative", 4), ("Positive", 6)], (10, 4.5), [("Positive", 10)]
        )
        result = await detect_anomaly(db, restaurant_id=1)
        assert result.detected is True
        assert "average rating dropped" in result.message

    @pytest.mark.asyncio
    async def test_small_rating_change_not_detected(self) -> None:
        db = _db_returning(
            (10, 4.4), [("Positive", 9), ("Negative", 1)], (10, 4.5), [("Positive", 10)]
        )
        result = await detect_anomaly(db, restaurant_id=1)
        assert result.detected is False
        assert result.message is None

    @pytest.mark.asyncio
    async def test_negative_share_spike_detected(self) -> None:
        db = _db_returning(
            (10, 4.4),
            [("Negative", 6), ("Positive", 4)],
            (10, 4.4),
            [("Negative", 1), ("Positive", 9)],
        )
        result = await detect_anomaly(db, restaurant_id=1)
        assert result.detected is True
        assert "negative reviews rose" in result.message

    @pytest.mark.asyncio
    async def test_result_carries_counts_and_averages(self) -> None:
        db = _db_returning(
            (10, 3.5), [("Negative", 4), ("Positive", 6)], (10, 4.5), [("Positive", 10)]
        )
        result = await detect_anomaly(db, restaurant_id=1)
        assert result.recent_count == 10
        assert result.baseline_count == 10
        assert result.recent_avg_rating == 3.5
        assert result.baseline_avg_rating == 4.5


class TestWindowBoundaries:
    @pytest.mark.asyncio
    async def test_recent_and_baseline_windows_do_not_share_a_boundary_date(self) -> None:
        """Regression test: compute_period_stats' date bounds are both
        inclusive, so if the recent window's start date equalled the
        baseline window's end date, that one day's reviews would be counted
        in both windows at once."""
        calls: list[tuple] = []

        async def fake_compute_period_stats(db, restaurant_id, date_from, date_to):
            calls.append((date_from, date_to))
            return PeriodStats(count=10, avg_rating=4.0, sentiment_counts={})

        with patch("src.core.anomaly.compute_period_stats", side_effect=fake_compute_period_stats):
            await detect_anomaly(MagicMock(), restaurant_id=1)

        assert len(calls) == 2
        (recent_from, _recent_to), (_baseline_from, baseline_to) = calls
        recent_from_date = date.fromisoformat(recent_from)
        baseline_to_date = date.fromisoformat(baseline_to)
        assert baseline_to_date < recent_from_date


class TestGetAnomalyStatus:
    @pytest.mark.asyncio
    async def test_cache_hit_skips_db(self) -> None:
        cached = {
            "detected": True,
            "recent_count": 10,
            "baseline_count": 10,
            "recent_avg_rating": 3.5,
            "baseline_avg_rating": 4.5,
            "recent_negative_share": 0.4,
            "baseline_negative_share": 0.0,
            "message": "Heads up: average rating dropped from 4.5 to 3.5 over the past 7 days.",
        }
        cache = MagicMock()
        cache.get_json = AsyncMock(return_value=cached)
        db = MagicMock()
        db.execute = AsyncMock()

        result = await get_anomaly_status(db, cache, restaurant_id=1)

        assert isinstance(result, AnomalyResult)
        assert result.detected is True
        db.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_miss_computes_and_stores(self) -> None:
        db = _db_returning(
            (10, 3.5), [("Negative", 4), ("Positive", 6)], (10, 4.5), [("Positive", 10)]
        )
        cache = MagicMock()
        cache.get_json = AsyncMock(return_value=None)
        cache.set_json = AsyncMock()

        result = await get_anomaly_status(db, cache, restaurant_id=1)

        assert result.detected is True
        cache.set_json.assert_awaited_once()
        args, _ = cache.set_json.call_args
        assert args[0] == "anomaly:1"
