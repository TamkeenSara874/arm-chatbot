"""Unit tests for the overall-rating fast path (exact avg rating via direct SQL).

Regression coverage for a real bug: a plain "what's my overall rating?"
question was falling through to the normal RAG path and estimating the
average rating from only the top_k retrieved evidence chunks (~3.5/5 from a
6-review sample), materially different from the real computed average
(~3.9-4.0). Mirrors tests/unit/test_trend_comparison.py's style, since
_compute_overall_stats is a thin wrapper around the same compute_period_stats
function trend comparison already uses.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.api.routes.chat import _compute_overall_stats, _format_overall_stats_answer
from src.core.review_stats import PeriodStats
from src.models.schemas import DateFilter


class TestFormatOverallStatsAnswer:
    def test_zero_reviews(self) -> None:
        stats = PeriodStats(count=0, avg_rating=None, sentiment_counts={})
        result = _format_overall_stats_answer(stats, None)
        assert result == "No reviews found (all-time)."

    def test_zero_reviews_with_date_filter_says_selected_period(self) -> None:
        stats = PeriodStats(count=0, avg_rating=None, sentiment_counts={})
        df = DateFilter(from_date="2026-06-01", to_date="2026-06-30")
        result = _format_overall_stats_answer(stats, df)
        assert "selected period" in result
        assert "all-time" not in result

    def test_reviews_present_but_no_ratings_recorded(self) -> None:
        stats = PeriodStats(count=5, avg_rating=None, sentiment_counts={"Neutral": 5})
        result = _format_overall_stats_answer(stats, None)
        assert "5 reviews" in result
        assert "no star ratings are recorded" in result

    def test_normal_case_states_exact_count_and_rating(self) -> None:
        stats = PeriodStats(
            count=2753, avg_rating=3.97, sentiment_counts={"Positive": 1349, "Negative": 500}
        )
        result = _format_overall_stats_answer(stats, None)
        assert "2753 reviews" in result
        assert "3.97/5" in result
        assert "Positive: 1349" in result
        assert "Negative: 500" in result

    def test_sentiment_breakdown_sorted_descending(self) -> None:
        stats = PeriodStats(
            count=10,
            avg_rating=4.0,
            sentiment_counts={"Negative": 1, "Positive": 8, "Mixed": 1},
        )
        result = _format_overall_stats_answer(stats, None)
        pos_idx = result.index("Positive")
        neg_idx = result.index("Negative")
        assert pos_idx < neg_idx  # larger count listed first

    def test_all_time_label_when_no_date_filter(self) -> None:
        stats = PeriodStats(count=1, avg_rating=5.0, sentiment_counts={"Positive": 1})
        result = _format_overall_stats_answer(stats, None)
        assert "(all-time)" in result


class TestComputeOverallStats:
    @pytest.mark.asyncio
    async def test_no_date_filter_queries_all_time(self) -> None:
        db = MagicMock()
        count_result = MagicMock()
        count_result.one.return_value = (2753, 3.97)
        sentiment_result = MagicMock()
        sentiment_result.all.return_value = [("Positive", 1349), ("Negative", 500)]
        db.execute = AsyncMock(side_effect=[count_result, sentiment_result])

        decomposed = MagicMock(date_filter=None)
        result = await _compute_overall_stats(db, restaurant_id=1, decomposed=decomposed)

        assert result.count == 2753
        assert result.avg_rating == 3.97
        assert result.sentiment_counts == {"Positive": 1349, "Negative": 500}

    @pytest.mark.asyncio
    async def test_with_date_filter_scopes_to_period(self) -> None:
        db = MagicMock()
        count_result = MagicMock()
        count_result.one.return_value = (100, 4.2)
        sentiment_result = MagicMock()
        sentiment_result.all.return_value = [("Positive", 90), ("Negative", 10)]
        db.execute = AsyncMock(side_effect=[count_result, sentiment_result])

        decomposed = MagicMock(
            date_filter=DateFilter(from_date="2026-06-01", to_date="2026-06-30")
        )
        result = await _compute_overall_stats(db, restaurant_id=1, decomposed=decomposed)

        assert result.count == 100
        assert result.avg_rating == 4.2
