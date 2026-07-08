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
        result = _format_breakdown_answer({"Yelp": 1, "Google": 13, "Tripadvisor": 5}, "source")
        assert result.index("Google: 13") < result.index("Tripadvisor: 5")
        assert result.index("Tripadvisor: 5") < result.index("Yelp: 1")

    def test_includes_total_and_dimension(self) -> None:
        result = _format_breakdown_answer({"Google": 13, "Yelp": 1}, "source")
        assert "across all 14 reviews" in result
        assert "breakdown by source" in result

    def test_single_entry(self) -> None:
        result = _format_breakdown_answer({"Positive": 20}, "sentiment")
        assert "Positive: 20" in result
        assert "across all 20 reviews" in result


class TestComputeBreakdown:
    @pytest.mark.asyncio
    async def test_source_dimension_groups_by_source(self) -> None:
        db = MagicMock()
        result = MagicMock()
        result.all.return_value = [
            ("Google", 13),
            ("Tripadvisor", 5),
            ("Yelp", 1),
            ("OpenTable", 1),
        ]
        db.execute = AsyncMock(return_value=result)

        breakdown = await _compute_breakdown(db, restaurant_id=1, dimension="source")

        assert breakdown == {"Google": 13, "Tripadvisor": 5, "Yelp": 1, "OpenTable": 1}
        db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_rating_dimension_groups_by_rating(self) -> None:
        db = MagicMock()
        result = MagicMock()
        result.all.return_value = [(5.0, 100), (1.0, 20)]
        db.execute = AsyncMock(return_value=result)

        breakdown = await _compute_breakdown(db, restaurant_id=1, dimension="rating")

        assert breakdown == {"5.0": 100, "1.0": 20}

    @pytest.mark.asyncio
    async def test_sentiment_dimension_groups_by_sentiment(self) -> None:
        db = MagicMock()
        result = MagicMock()
        result.all.return_value = [("Positive", 776), ("Negative", 315)]
        db.execute = AsyncMock(return_value=result)

        breakdown = await _compute_breakdown(db, restaurant_id=1, dimension="sentiment")

        assert breakdown == {"Positive": 776, "Negative": 315}

    @pytest.mark.asyncio
    async def test_empty_result_returns_empty_dict(self) -> None:
        db = MagicMock()
        result = MagicMock()
        result.all.return_value = []
        db.execute = AsyncMock(return_value=result)

        breakdown = await _compute_breakdown(db, restaurant_id=999, dimension="source")

        assert breakdown == {}
