"""Unit tests for src/core/review_stats.py -- shared exact-SQL period aggregation
used by both trend comparison (test_trend_comparison.py) and anomaly detection
(test_anomaly.py).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.review_stats import (
    compute_period_stats,
    compute_theme_cooccurrence_count,
    compute_theme_count,
)


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


class TestComputeThemeCount:
    """Regression coverage: qualitative-theme count questions ("how many people
    called my staff rude") were previously answered honestly but only from the
    top_k=20 retrieved sample, which could badly undercount a theme that
    actually appears in far more of the full review set. This counts across
    ALL reviews via full_review ILIKE instead.
    """

    @pytest.mark.asyncio
    async def test_counts_reviews_matching_any_keyword(self) -> None:
        db = MagicMock()
        result = MagicMock()
        result.scalar_one.return_value = 38
        db.execute = AsyncMock(return_value=result)

        count = await compute_theme_count(db, restaurant_id=1, keywords=["rude", "unfriendly"])

        assert count == 38
        db.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_empty_keywords_returns_zero_without_querying(self) -> None:
        db = MagicMock()
        db.execute = AsyncMock()

        count = await compute_theme_count(db, restaurant_id=1, keywords=[])

        assert count == 0
        db.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_date_filter_does_not_error(self) -> None:
        db = MagicMock()
        result = MagicMock()
        result.scalar_one.return_value = 5
        db.execute = AsyncMock(return_value=result)

        count = await compute_theme_count(
            db,
            restaurant_id=1,
            keywords=["cold food"],
            date_from="2026-01-01",
            date_to="2026-06-30",
        )

        assert count == 5

    @pytest.mark.asyncio
    async def test_malformed_date_suppressed_to_none(self) -> None:
        db = MagicMock()
        result = MagicMock()
        result.scalar_one.return_value = 0
        db.execute = AsyncMock(return_value=result)

        count = await compute_theme_count(
            db, restaurant_id=1, keywords=["slow service"], date_from="not-a-date"
        )

        assert count == 0


class TestComputeThemeCooccurrenceCount:
    """Regression coverage for a real bug: "which reviews mention both slow
    service and cold food?" had no way to express an AND between two theme
    groups, so decomposition folded both themes into one flat OR-matched
    list -- the exact count reported "any review mentioning either theme"
    while being worded as if it meant "both together," directly
    contradicting the model's own read of the retrieved evidence (which
    found no single review mentioning both). This computes a real
    intersection: theme A ILIKE-matched AND theme B ILIKE-matched.
    """

    @pytest.mark.asyncio
    async def test_counts_reviews_matching_both_groups(self) -> None:
        db = MagicMock()
        result = MagicMock()
        result.scalar_one.return_value = 3
        db.execute = AsyncMock(return_value=result)

        count = await compute_theme_cooccurrence_count(
            db,
            restaurant_id=1,
            keywords_a=["slow service", "slow"],
            keywords_b=["cold food", "cold"],
        )

        assert count == 3
        db.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_empty_first_group_returns_zero_without_querying(self) -> None:
        db = MagicMock()
        db.execute = AsyncMock()

        count = await compute_theme_cooccurrence_count(
            db, restaurant_id=1, keywords_a=[], keywords_b=["cold food"]
        )

        assert count == 0
        db.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_second_group_returns_zero_without_querying(self) -> None:
        db = MagicMock()
        db.execute = AsyncMock()

        count = await compute_theme_cooccurrence_count(
            db, restaurant_id=1, keywords_a=["slow service"], keywords_b=[]
        )

        assert count == 0
        db.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_date_filter_does_not_error(self) -> None:
        db = MagicMock()
        result = MagicMock()
        result.scalar_one.return_value = 1
        db.execute = AsyncMock(return_value=result)

        count = await compute_theme_cooccurrence_count(
            db,
            restaurant_id=1,
            keywords_a=["rude"],
            keywords_b=["cold food"],
            date_from="2026-01-01",
            date_to="2026-06-30",
        )

        assert count == 1
