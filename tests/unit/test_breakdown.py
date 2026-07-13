"""Unit tests for the exact whole-dataset breakdown fast path (GROUP BY via direct SQL).

Mirrors tests/unit/test_trend_comparison.py's style -- direct Postgres
aggregation, mocked db.execute, no live database needed.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.api.routes.chat import _compute_breakdown, _format_breakdown_answer


class TestFormatBreakdownAnswer:
    def test_empty_breakdown(self) -> None:
        result = _format_breakdown_answer({}, "source")
        assert result == "No reviews have a recorded source."

    def test_orders_by_count_descending(self) -> None:
        result = _format_breakdown_answer(
            {"Yelp": (1, 3.2), "Google": (13, 4.0), "Tripadvisor": (5, 3.8)}, "source"
        )
        assert result.index("Google") < result.index("Tripadvisor")
        assert result.index("Tripadvisor") < result.index("Yelp")

    def test_includes_total_and_dimension(self) -> None:
        result = _format_breakdown_answer({"Google": (13, 4.0), "Yelp": (1, 3.2)}, "source")
        assert "across all 14 reviews" in result
        assert "breakdown by source" in result

    def test_single_entry(self) -> None:
        result = _format_breakdown_answer({"Positive": (20, 4.5)}, "sentiment")
        assert "Positive: 20" in result
        assert "across all 20 reviews" in result

    def test_source_dimension_includes_avg_rating(self) -> None:
        # Regression test: "why is my rating on Yelp so much lower than on
        # Google?" had no exact per-platform rating to reason from at all --
        # only counts were computed. A by-source breakdown must state the
        # real avg rating per platform, not just review counts.
        result = _format_breakdown_answer({"Google": (2125, 4.0), "Yelp": (30, 3.2)}, "source")
        assert "avg rating 4.0/5" in result
        assert "avg rating 3.2/5" in result

    def test_non_source_dimension_omits_avg_rating(self) -> None:
        # Avg rating per rating-group or per-sentiment-group isn't a
        # meaningful addition the way it is for source -- only source gets it.
        result = _format_breakdown_answer(
            {"Positive": (20, 4.5), "Negative": (5, 1.5)}, "sentiment"
        )
        assert "avg rating" not in result

    def test_source_with_no_ratings_recorded_omits_avg_rating(self) -> None:
        result = _format_breakdown_answer({"Google": (5, None)}, "source")
        assert "Google: 5 reviews" in result
        assert "avg rating" not in result


class TestComputeBreakdown:
    @pytest.mark.asyncio
    async def test_source_dimension_groups_by_source(self) -> None:
        db = MagicMock()
        result = MagicMock()
        result.all.return_value = [
            ("Google", 13, 4.0),
            ("Tripadvisor", 5, 3.8),
            ("Yelp", 1, 3.2),
            ("OpenTable", 1, 5.0),
        ]
        db.execute = AsyncMock(return_value=result)

        breakdown = await _compute_breakdown(db, restaurant_id=1, dimension="source")

        assert breakdown == {
            "Google": (13, 4.0),
            "Tripadvisor": (5, 3.8),
            "Yelp": (1, 3.2),
            "OpenTable": (1, 5.0),
        }
        db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_rating_dimension_groups_by_rating(self) -> None:
        db = MagicMock()
        result = MagicMock()
        result.all.return_value = [(5.0, 100, 5.0), (1.0, 20, 1.0)]
        db.execute = AsyncMock(return_value=result)

        breakdown = await _compute_breakdown(db, restaurant_id=1, dimension="rating")

        assert breakdown == {"5.0": (100, 5.0), "1.0": (20, 1.0)}

    @pytest.mark.asyncio
    async def test_sentiment_dimension_groups_by_sentiment(self) -> None:
        db = MagicMock()
        result = MagicMock()
        result.all.return_value = [("Positive", 776, 4.3), ("Negative", 315, 1.9)]
        db.execute = AsyncMock(return_value=result)

        breakdown = await _compute_breakdown(db, restaurant_id=1, dimension="sentiment")

        assert breakdown == {"Positive": (776, 4.3), "Negative": (315, 1.9)}

    @pytest.mark.asyncio
    async def test_empty_result_returns_empty_dict(self) -> None:
        db = MagicMock()
        result = MagicMock()
        result.all.return_value = []
        db.execute = AsyncMock(return_value=result)

        breakdown = await _compute_breakdown(db, restaurant_id=999, dimension="source")

        assert breakdown == {}

    @pytest.mark.asyncio
    async def test_null_avg_rating_preserved_as_none(self) -> None:
        db = MagicMock()
        result = MagicMock()
        result.all.return_value = [("Google", 5, None)]
        db.execute = AsyncMock(return_value=result)

        breakdown = await _compute_breakdown(db, restaurant_id=1, dimension="source")

        assert breakdown == {"Google": (5, None)}
