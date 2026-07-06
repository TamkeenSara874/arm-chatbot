"""Unit tests for the trend-comparison fast path (two-period stats via direct SQL).

Mirrors tests/unit/test_count_query.py's style for the count_query fast path --
same mocked-db.execute approach, since these functions are direct Postgres
aggregations, not something that needs a live database to unit-test.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.api.routes.chat import (
    _compute_trend_comparison,
    _format_period_stats,
    _period_label,
)
from src.core.review_stats import PeriodStats
from src.models.schemas import DateFilter


class TestPeriodLabel:
    def test_none_date_filter_falls_back(self) -> None:
        assert _period_label("Current period", None) == "Current period (all time)"

    def test_empty_date_filter_falls_back(self) -> None:
        assert _period_label("Current period", DateFilter()) == "Current period (all time)"

    def test_full_range(self) -> None:
        df = DateFilter(from_date="2026-06-01", to_date="2026-06-30")
        assert _period_label("Current period", df) == "Current period (2026-06-01 to 2026-06-30)"

    def test_missing_from_uses_earliest(self) -> None:
        df = DateFilter(to_date="2026-06-30")
        assert _period_label("Current period", df) == "Current period (earliest to 2026-06-30)"

    def test_missing_to_uses_latest(self) -> None:
        df = DateFilter(from_date="2026-06-01")
        assert _period_label("Current period", df) == "Current period (2026-06-01 to latest)"


class TestFormatPeriodStats:
    def test_includes_count_rating_sentiment(self) -> None:
        stats = PeriodStats(
            count=10, avg_rating=4.2, sentiment_counts={"Positive": 7, "Negative": 3}
        )
        result = _format_period_stats("Current period (all time)", stats)
        assert "10 reviews" in result
        assert "avg rating 4.2/5" in result
        assert "7 Positive" in result
        assert "3 Negative" in result

    def test_singular_review_wording(self) -> None:
        stats = PeriodStats(count=1, avg_rating=5.0, sentiment_counts={})
        result = _format_period_stats("Current period", stats)
        assert "1 review," in result

    def test_no_rating_shows_na(self) -> None:
        stats = PeriodStats(count=0, avg_rating=None, sentiment_counts={})
        result = _format_period_stats("Current period", stats)
        assert "avg rating N/A" in result

    def test_no_sentiment_shows_none(self) -> None:
        stats = PeriodStats(count=0, avg_rating=None, sentiment_counts={})
        result = _format_period_stats("Current period", stats)
        assert "sentiment breakdown: none" in result


class TestComputeTrendComparison:
    @pytest.mark.asyncio
    async def test_returns_none_without_compare_date_filter(self) -> None:
        db = MagicMock()
        decomposed = MagicMock(compare_date_filter=None)
        result = await _compute_trend_comparison(db, restaurant_id=1, decomposed=decomposed)
        assert result is None
        db.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_formats_both_periods_with_labels(self) -> None:
        db = MagicMock()
        current_count = MagicMock()
        current_count.one.return_value = (10, 4.5)
        current_sentiment = MagicMock()
        current_sentiment.all.return_value = [("Positive", 10)]
        previous_count = MagicMock()
        previous_count.one.return_value = (5, 3.0)
        previous_sentiment = MagicMock()
        previous_sentiment.all.return_value = [("Negative", 5)]
        db.execute = AsyncMock(
            side_effect=[current_count, current_sentiment, previous_count, previous_sentiment]
        )

        decomposed = MagicMock(
            date_filter=DateFilter(from_date="2026-06-01", to_date="2026-06-30"),
            compare_date_filter=DateFilter(from_date="2026-05-01", to_date="2026-05-31"),
        )

        result = await _compute_trend_comparison(db, restaurant_id=1, decomposed=decomposed)

        assert result is not None
        assert "Current period (2026-06-01 to 2026-06-30)" in result
        assert "10 reviews" in result
        assert "Comparison period (2026-05-01 to 2026-05-31)" in result
        assert "5 reviews" in result
        assert " | " in result

    @pytest.mark.asyncio
    async def test_missing_current_date_filter_defaults_to_all_time_label(self) -> None:
        db = MagicMock()
        count_result = MagicMock()
        count_result.one.return_value = (0, None)
        sentiment_result = MagicMock()
        sentiment_result.all.return_value = []
        db.execute = AsyncMock(
            side_effect=[count_result, sentiment_result, count_result, sentiment_result]
        )

        decomposed = MagicMock(
            date_filter=None,
            compare_date_filter=DateFilter(from_date="2026-05-01", to_date="2026-05-31"),
        )

        result = await _compute_trend_comparison(db, restaurant_id=1, decomposed=decomposed)

        assert "Current period (all time)" in result
