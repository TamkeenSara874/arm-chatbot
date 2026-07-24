"""Unit tests for report.py -- pure functions and mocked async helpers."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.report import (
    _aggregate_entities,
    _build_markdown,
    _extract_date_range,
    _generate_summary,
    _load_review_rows,
    _parse_date_arg,
    generate_report,
)


class TestParseDateArg:
    def test_valid_iso_date_returns_date(self) -> None:
        assert _parse_date_arg("2025-06-15") == date(2025, 6, 15)

    def test_none_returns_none(self) -> None:
        assert _parse_date_arg(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert _parse_date_arg("") is None

    def test_null_string_returns_none(self) -> None:
        assert _parse_date_arg("null") is None

    def test_invalid_format_returns_none(self) -> None:
        assert _parse_date_arg("not-a-date") is None


class TestBuildMarkdown:
    def _call(self, **overrides) -> str:
        defaults: dict = {
            "restaurant_id": 1,
            "date_from": None,
            "date_to": None,
            "total_reviews": 50,
            "avg_rating": 4.2,
            "rating_distribution": {"4": 30, "5": 20},
            "sentiment_breakdown": {"Positive": 40, "Negative": 10},
            "source_breakdown": {"Google": 30, "Yelp": 20},
            "top_praised": [("biryani", 15)],
            "top_complained": [("service", 8)],
            "summary": "Overall positive.",
        }
        defaults.update(overrides)
        return _build_markdown(**defaults)

    def test_all_time_period_when_no_dates(self) -> None:
        md = self._call()
        assert "All Time" in md
        assert "50" in md
        assert "4.2/5" in md
        assert "biryani (15 mentions)" in md
        assert "service (8 mentions)" in md

    def test_full_date_range_period_shown(self) -> None:
        md = self._call(date_from=date(2025, 1, 1), date_to=date(2025, 6, 30))
        assert "2025-01-01 to 2025-06-30" in md

    def test_date_from_only(self) -> None:
        md = self._call(date_from=date(2025, 3, 1), date_to=None)
        assert "From 2025-03-01" in md

    def test_date_to_only(self) -> None:
        md = self._call(date_from=None, date_to=date(2025, 12, 31))
        assert "Up to 2025-12-31" in md

    def test_no_rating_shows_na(self) -> None:
        md = self._call(avg_rating=None)
        assert "N/A" in md

    def test_no_praised_section_when_empty(self) -> None:
        md = self._call(top_praised=[], top_complained=[])
        assert "Top Praised" not in md
        assert "Top Complaints" not in md

    def test_praised_section_present_when_non_empty(self) -> None:
        md = self._call(top_praised=[("pasta", 5)], top_complained=[])
        assert "Top Praised" in md
        assert "pasta (5 mentions)" in md

    def test_complained_section_present_when_non_empty(self) -> None:
        md = self._call(top_praised=[], top_complained=[("wait time", 3)])
        assert "Top Complaints" in md
        assert "wait time (3 mentions)" in md

    def test_restaurant_id_in_heading(self) -> None:
        md = self._call(restaurant_id=99)
        assert "Restaurant 99" in md

    def test_overview_section_present(self) -> None:
        md = self._call()
        assert "## Overview" in md
        assert "## Sentiment Breakdown" in md
        assert "## Rating Distribution" in md
        assert "## Source Breakdown" in md

    def test_summary_at_end(self) -> None:
        md = self._call(summary="Great quarter overall.")
        assert "## Summary" in md
        assert "Great quarter overall." in md

    def test_zero_reviews(self) -> None:
        md = self._call(total_reviews=0, avg_rating=None, rating_distribution={})
        assert "0" in md


class TestExtractDateRange:
    @pytest.mark.asyncio
    async def test_returns_extracted_dates_from_tool_call(self) -> None:
        tool_call = MagicMock()
        tool_call.function.arguments = '{"date_from": "2025-01-01", "date_to": "2025-06-30"}'
        mock_response = MagicMock()
        mock_response.choices[0].message.tool_calls = [tool_call]
        mock_openai = MagicMock()
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

        from_d, to_d = await _extract_date_range(
            mock_openai, "gpt-4.1-mini", "Report for Q1 2025", None, None
        )

        assert from_d == date(2025, 1, 1)
        assert to_d == date(2025, 6, 30)

    @pytest.mark.asyncio
    async def test_falls_back_to_provided_dates_on_openai_failure(self) -> None:
        mock_openai = MagicMock()
        mock_openai.chat.completions.create = AsyncMock(side_effect=RuntimeError("API error"))

        fallback = date(2024, 1, 1)
        from_d, to_d = await _extract_date_range(
            mock_openai, "gpt-4.1-mini", "any query", fallback, None
        )

        assert from_d == fallback
        assert to_d is None

    @pytest.mark.asyncio
    async def test_no_tool_calls_uses_fallback(self) -> None:
        mock_response = MagicMock()
        mock_response.choices[0].message.tool_calls = []
        mock_openai = MagicMock()
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

        fallback = date(2025, 3, 1)
        from_d, to_d = await _extract_date_range(
            mock_openai, "gpt-4.1-mini", "all time report", fallback, None
        )

        assert from_d == fallback

    @pytest.mark.asyncio
    async def test_null_dates_in_tool_args_uses_fallback(self) -> None:
        tool_call = MagicMock()
        tool_call.function.arguments = '{"date_from": null, "date_to": null}'
        mock_response = MagicMock()
        mock_response.choices[0].message.tool_calls = [tool_call]
        mock_openai = MagicMock()
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

        fallback_to = date(2025, 6, 30)
        from_d, to_d = await _extract_date_range(
            mock_openai, "gpt-4.1-mini", "all reviews", None, fallback_to
        )

        assert from_d is None
        assert to_d == fallback_to


class TestGenerateSummary:
    @pytest.mark.asyncio
    async def test_returns_llm_content(self) -> None:
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Excellent results."
        mock_openai = MagicMock()
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await _generate_summary(
            openai_client=mock_openai,
            model="gpt-4.1-mini",
            total_reviews=100,
            avg_rating=4.3,
            sentiment_breakdown={"Positive": 80, "Negative": 20},
            top_praised=[("biryani", 30)],
            top_complained=[("service", 10)],
        )

        assert result == "Excellent results."

    @pytest.mark.asyncio
    async def test_returns_fallback_message_on_failure(self) -> None:
        mock_openai = MagicMock()
        mock_openai.chat.completions.create = AsyncMock(side_effect=RuntimeError("API down"))

        result = await _generate_summary(
            openai_client=mock_openai,
            model="gpt-4.1-mini",
            total_reviews=50,
            avg_rating=None,
            sentiment_breakdown={},
            top_praised=[],
            top_complained=[],
        )

        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_none_avg_rating_formats_as_not_available(self) -> None:
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Summary."
        mock_openai = MagicMock()
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

        await _generate_summary(
            openai_client=mock_openai,
            model="gpt-4.1-mini",
            total_reviews=5,
            avg_rating=None,
            sentiment_breakdown={},
            top_praised=[],
            top_complained=[],
        )

        call_args = mock_openai.chat.completions.create.call_args
        prompt_text = call_args[1]["messages"][1]["content"]
        assert "not available" in prompt_text


class TestLoadReviewRows:
    @pytest.mark.asyncio
    async def test_returns_rows_from_db(self) -> None:
        from src.models.db_entities import ReviewChunkMeta

        row = MagicMock(spec=ReviewChunkMeta)
        row.rating = 4.0
        row.sentiment_label = "Positive"
        row.source = "Google"

        scalars = MagicMock()
        scalars.all = MagicMock(return_value=[row])
        execute_result = MagicMock()
        execute_result.scalars = MagicMock(return_value=scalars)
        db = MagicMock()
        db.execute = AsyncMock(return_value=execute_result)

        rows = await _load_review_rows(db, restaurant_id=1, date_from=None, date_to=None)

        assert len(rows) == 1
        assert rows[0] is row

    @pytest.mark.asyncio
    async def test_applies_date_from_filter(self) -> None:
        scalars = MagicMock()
        scalars.all = MagicMock(return_value=[])
        execute_result = MagicMock()
        execute_result.scalars = MagicMock(return_value=scalars)
        db = MagicMock()
        db.execute = AsyncMock(return_value=execute_result)

        await _load_review_rows(db, 1, date_from=date(2025, 1, 1), date_to=None)
        db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_applies_date_to_filter(self) -> None:
        scalars = MagicMock()
        scalars.all = MagicMock(return_value=[])
        execute_result = MagicMock()
        execute_result.scalars = MagicMock(return_value=scalars)
        db = MagicMock()
        db.execute = AsyncMock(return_value=execute_result)

        await _load_review_rows(db, 1, date_from=None, date_to=date(2025, 12, 31))
        db.execute.assert_called_once()


class TestAggregateEntities:
    @pytest.mark.asyncio
    async def test_non_qdrant_store_returns_empty(self) -> None:
        from src.services.vector.base import BaseVectorStore

        store = MagicMock(spec=BaseVectorStore)
        praised, complained = await _aggregate_entities(store, "review_chunks", 1)
        assert praised == []
        assert complained == []

    @pytest.mark.asyncio
    async def test_qdrant_store_aggregates_entity_mentions(self) -> None:
        from src.services.vector.qdrant_store import QdrantStore

        point1 = MagicMock()
        point1.payload = {"food_entities": ["biryani", "naan"], "sentiment_label": "Positive"}
        point2 = MagicMock()
        point2.payload = {"food_entities": ["slow service"], "sentiment_label": "Negative"}

        store = MagicMock(spec=QdrantStore)
        store._build_filter = MagicMock(return_value=None)
        store.client = MagicMock()
        store.client.scroll = AsyncMock(return_value=([point1, point2], None))

        praised, complained = await _aggregate_entities(store, "review_chunks", 1)

        praised_dict = dict(praised)
        complained_dict = dict(complained)
        assert praised_dict.get("biryani") == 1
        assert praised_dict.get("naan") == 1
        assert complained_dict.get("slow service") == 1

    @pytest.mark.asyncio
    async def test_qdrant_scroll_failure_returns_empty(self) -> None:
        from src.services.vector.qdrant_store import QdrantStore

        store = MagicMock(spec=QdrantStore)
        store._build_filter = MagicMock(return_value=None)
        store.client = MagicMock()
        store.client.scroll = AsyncMock(side_effect=RuntimeError("Qdrant down"))

        praised, complained = await _aggregate_entities(store, "review_chunks", 1)
        assert praised == []
        assert complained == []

    @pytest.mark.asyncio
    async def test_point_with_empty_payload_skipped(self) -> None:
        from src.services.vector.qdrant_store import QdrantStore

        point = MagicMock()
        point.payload = None

        store = MagicMock(spec=QdrantStore)
        store._build_filter = MagicMock(return_value=None)
        store.client = MagicMock()
        store.client.scroll = AsyncMock(return_value=([point], None))

        praised, complained = await _aggregate_entities(store, "review_chunks", 1)
        assert praised == []
        assert complained == []

    @pytest.mark.asyncio
    async def test_date_range_forwarded_to_qdrant_filter(self) -> None:
        from src.services.vector.qdrant_store import QdrantStore

        store = MagicMock(spec=QdrantStore)
        store._build_filter = MagicMock(return_value=None)
        store.client = MagicMock()
        store.client.scroll = AsyncMock(return_value=([], None))

        await _aggregate_entities(store, "review_chunks", 1, date(2025, 1, 1), date(2025, 6, 30))

        filters = store._build_filter.call_args[0][0]
        assert filters["restaurant_id"] == 1
        # Epoch bounds are passed so the entity scroll covers the same window as
        # the DB metrics; the end bound is widened to end-of-day, so it sorts after.
        assert filters["date_to"] > filters["date_from"]

    @pytest.mark.asyncio
    async def test_no_date_range_omits_date_filter_keys(self) -> None:
        from src.services.vector.qdrant_store import QdrantStore

        store = MagicMock(spec=QdrantStore)
        store._build_filter = MagicMock(return_value=None)
        store.client = MagicMock()
        store.client.scroll = AsyncMock(return_value=([], None))

        await _aggregate_entities(store, "review_chunks", 1)

        filters = store._build_filter.call_args[0][0]
        assert filters == {"restaurant_id": 1}


class TestGenerateReport:
    def _make_db(self, rows=None):
        scalars = MagicMock()
        scalars.all = MagicMock(return_value=rows or [])
        execute_result = MagicMock()
        execute_result.scalars = MagicMock(return_value=scalars)
        db = MagicMock()
        db.execute = AsyncMock(return_value=execute_result)
        return db

    def _make_openai(self, summary="Good quarter."):
        tool_call = MagicMock()
        tool_call.function.arguments = '{"date_from": null, "date_to": null}'
        tool_response = MagicMock()
        tool_response.choices[0].message.tool_calls = [tool_call]
        summary_response = MagicMock()
        summary_response.choices[0].message.content = summary
        mock_openai = MagicMock()
        mock_openai.chat.completions.create = AsyncMock(
            side_effect=[tool_response, summary_response]
        )
        return mock_openai

    @pytest.mark.asyncio
    async def test_returns_insights_report_with_correct_fields(self) -> None:
        from src.models.db_entities import ReviewChunkMeta
        from src.services.vector.base import BaseVectorStore

        row = MagicMock(spec=ReviewChunkMeta)
        row.rating = 4.0
        row.sentiment_label = "Positive"
        row.source = "Google"

        report = await generate_report(
            user_message="Full report please",
            restaurant_id=7,
            db_session=self._make_db([row]),
            vector_store=MagicMock(spec=BaseVectorStore),
            qdrant_reviews_collection="review_chunks",
            openai_client=self._make_openai("Strong performance."),
            model="gpt-4.1-mini",
        )

        assert report.restaurant_id == 7
        assert report.total_reviews == 1
        assert report.avg_rating == 4.0
        assert report.summary == "Strong performance."
        assert "## Overview" in report.markdown

    @pytest.mark.asyncio
    async def test_avg_rating_none_when_no_ratings(self) -> None:
        from src.models.db_entities import ReviewChunkMeta
        from src.services.vector.base import BaseVectorStore

        row = MagicMock(spec=ReviewChunkMeta)
        row.rating = None
        row.sentiment_label = "Neutral"
        row.source = "Yelp"

        report = await generate_report(
            user_message="Report",
            restaurant_id=1,
            db_session=self._make_db([row]),
            vector_store=MagicMock(spec=BaseVectorStore),
            qdrant_reviews_collection="review_chunks",
            openai_client=self._make_openai("No ratings yet."),
            model="gpt-4.1-mini",
        )

        assert report.avg_rating is None

    @pytest.mark.asyncio
    async def test_empty_reviews_returns_zero_total(self) -> None:
        from src.services.vector.base import BaseVectorStore

        report = await generate_report(
            user_message="Report",
            restaurant_id=2,
            db_session=self._make_db([]),
            vector_store=MagicMock(spec=BaseVectorStore),
            qdrant_reviews_collection="review_chunks",
            openai_client=self._make_openai("No data."),
            model="gpt-4.1-mini",
        )

        assert report.total_reviews == 0
        assert report.avg_rating is None

    @pytest.mark.asyncio
    async def test_sentiment_and_source_breakdowns_populated(self) -> None:
        from src.models.db_entities import ReviewChunkMeta
        from src.services.vector.base import BaseVectorStore

        def make_row(sentiment, source):
            r = MagicMock(spec=ReviewChunkMeta)
            r.rating = 4.0
            r.sentiment_label = sentiment
            r.source = source
            return r

        rows = [
            make_row("Positive", "Google"),
            make_row("Negative", "Yelp"),
            make_row("Positive", "Google"),
        ]

        report = await generate_report(
            user_message="Report",
            restaurant_id=3,
            db_session=self._make_db(rows),
            vector_store=MagicMock(spec=BaseVectorStore),
            qdrant_reviews_collection="review_chunks",
            openai_client=self._make_openai("Mixed results."),
            model="gpt-4.1-mini",
        )

        assert report.sentiment_breakdown.get("Positive") == 2
        assert report.sentiment_breakdown.get("Negative") == 1
        assert report.source_breakdown.get("Google") == 2
        assert report.source_breakdown.get("Yelp") == 1
