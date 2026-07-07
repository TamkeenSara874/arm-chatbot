"""Unit tests for the count_query fast path's sentiment-filter handling.

Regression coverage for a real bug: decomposition (a small, fast free-tier
model) occasionally extracts the wrong sentiment polarity on an otherwise
unambiguous query -- e.g. "total negative reviews" coming back with
sentiment_filter="Positive". A count_query's whole value is an exact,
trustworthy number, so this keyword-based override exists as a deterministic
safety net independent of what the LLM classified.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.routes.chat import (
    _compute_count,
    _format_count_answer,
    _handle_count_query,
    _resolve_sentiment_filter,
)
from src.models.schemas import ChatQueryRequest
from src.utils.tracing import RequestTrace


class TestResolveSentimentFilter:
    def test_negative_keyword_overrides_wrong_llm_classification(self) -> None:
        # The exact observed bug: decomposition said "Positive" for a query
        # that literally asks about negative reviews.
        result = _resolve_sentiment_filter("total negative reviews", "Positive")
        assert result == "Negative"

    def test_positive_keyword_overrides_wrong_llm_classification(self) -> None:
        result = _resolve_sentiment_filter("total positive reviews", "Negative")
        assert result == "Positive"

    def test_keyword_match_is_case_insensitive(self) -> None:
        assert _resolve_sentiment_filter("Total NEGATIVE Reviews", None) == "Negative"

    def test_no_keyword_falls_back_to_decomposed_filter(self) -> None:
        assert _resolve_sentiment_filter("how many reviews mention tacos", "Positive") == "Positive"
        assert _resolve_sentiment_filter("how many reviews mention tacos", None) is None

    def test_ambiguous_both_keywords_falls_back_to_decomposed_filter(self) -> None:
        result = _resolve_sentiment_filter("compare positive and negative reviews", "Positive")
        assert result == "Positive"

    def test_mixed_and_neutral_keywords_recognized(self) -> None:
        assert _resolve_sentiment_filter("how many mixed reviews", "Positive") == "Mixed"
        assert _resolve_sentiment_filter("how many neutral reviews", "Positive") == "Neutral"


class TestFormatCountAnswer:
    def test_no_sentiment_filter(self) -> None:
        assert _format_count_answer(5, None) == "You have 5 reviews in total."

    def test_with_sentiment_filter(self) -> None:
        assert _format_count_answer(5, "Negative") == "You have 5 negative reviews in total."

    def test_zero_count_no_filter(self) -> None:
        assert _format_count_answer(0, None) == "No reviews match that filter."

    def test_zero_count_with_filter(self) -> None:
        assert _format_count_answer(0, "Positive") == "No positive reviews match that filter."

    def test_singular_review_wording(self) -> None:
        assert _format_count_answer(1, None) == "You have 1 review in total."


class TestComputeCount:
    @pytest.mark.asyncio
    async def test_returns_scalar_result(self) -> None:
        db = MagicMock()
        exec_result = MagicMock()
        exec_result.scalar_one.return_value = 42
        db.execute = AsyncMock(return_value=exec_result)
        decomposed = MagicMock(date_filter=None, rating_filter=None)

        count = await _compute_count(
            db, restaurant_id=1, decomposed=decomposed, sentiment_filter="Negative"
        )

        assert count == 42
        db.execute.assert_awaited_once()


class TestHandleCountQuery:
    @pytest.mark.asyncio
    async def test_increments_count_query_total_metric(self) -> None:
        """Regression test: count_query_total was declared in metrics.py but
        never actually incremented anywhere -- a dead metric."""
        db = MagicMock()
        exec_result = MagicMock()
        exec_result.scalar_one.return_value = 7
        db.execute = AsyncMock(return_value=exec_result)
        decomposed = MagicMock(date_filter=None, rating_filter=None, sentiment_filter=None)
        body = ChatQueryRequest(
            session_id=uuid.uuid4(), restaurant_id=1, message="how many reviews?"
        )
        trace = RequestTrace(session_id=str(body.session_id), restaurant_id=1)

        with patch("src.api.routes.chat.count_query_total") as mock_counter:
            answer, msg_id = await _handle_count_query(
                db,
                body,
                restaurant_id=1,
                decomposed=decomposed,
                sanitized="how many reviews?",
                trace=trace,
            )
            mock_counter.inc.assert_called_once()

        assert "7" in answer
