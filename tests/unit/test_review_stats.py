"""Unit tests for src/core/review_stats.py -- shared exact-SQL period aggregation
used by both trend comparison (test_trend_comparison.py) and anomaly detection
(test_anomaly.py).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.review_stats import compute_period_stats


class TestComputePeriodStats:
    @pytest.mark.asyncio
    async def test_returns_count_avg_rating_and_sentiment_breakdown(self) -> None:
        db = MagicMock()
        count_result = MagicMock()
        count_result.one.return_value = (10, 4.256)
        sentiment_result = MagicMock()
        sentiment_result.all.return_value = [("Positive", 7), ("Negative", 3)]
        db.execute = AsyncMock(side_effect=[count_result, sentiment_result])

        stats = await compute_period_stats(
            db, restaurant_id=1, date_from="2026-06-01", date_to="2026-06-30"
        )

        assert stats.count == 10
        assert stats.avg_rating == 4.26
        assert stats.sentiment_counts == {"Positive": 7, "Negative": 3}
        assert db.execute.await_count == 2

    @pytest.mark.asyncio
    async def test_null_avg_rating_when_no_reviews(self) -> None:
        db = MagicMock()
        count_result = MagicMock()
        count_result.one.return_value = (0, None)
        sentiment_result = MagicMock()
        sentiment_result.all.return_value = []
        db.execute = AsyncMock(side_effect=[count_result, sentiment_result])

        stats = await compute_period_stats(db, restaurant_id=1, date_from=None, date_to=None)

        assert stats.count == 0
        assert stats.avg_rating is None
        assert stats.sentiment_counts == {}

    @pytest.mark.asyncio
    async def test_none_sentiment_label_grouped_as_unknown(self) -> None:
        db = MagicMock()
        count_result = MagicMock()
        count_result.one.return_value = (5, 3.0)
        sentiment_result = MagicMock()
        sentiment_result.all.return_value = [(None, 5)]
        db.execute = AsyncMock(side_effect=[count_result, sentiment_result])

        stats = await compute_period_stats(db, restaurant_id=1, date_from=None, date_to=None)

        assert stats.sentiment_counts == {"Unknown": 5}
