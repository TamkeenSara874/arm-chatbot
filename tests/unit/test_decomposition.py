"""Unit tests for query decomposition with validation retry and safe fallback."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.decomposition import decompose_query


def _make_client(responses: list[str]) -> MagicMock:
    """Mimics a real client's complete_structured(): validates each raw response
    against the given schema in turn, letting ValidationError surface on bad JSON
    exactly like GroqClient/OpenAIClient do, so decompose_query()'s retry logic
    is exercised the same way it is in production.
    """
    client = MagicMock()
    call_index = {"i": 0}

    async def _complete_structured(prompt, system, response_format, **kwargs):
        raw = responses[call_index["i"]]
        call_index["i"] += 1
        return response_format.model_validate_json(raw)

    client.complete_structured = AsyncMock(side_effect=_complete_structured)
    return client


def _valid_json(intent: str = "factual") -> str:
    return json.dumps(
        {
            "intent": intent,
            "complexity": "simple",
            "rephrased_query": "What is the best dish?",
            "aspect_filter": None,
            "sentiment_filter": None,
            "entities": [],
            "needs_aggregation": False,
            "sub_queries": [],
            "source_filter": None,
            "date_filter": None,
            "rating_filter": None,
        }
    )


@pytest.mark.asyncio
async def test_valid_response_returns_decomposed_query() -> None:
    client = _make_client([_valid_json("best_item")])
    result = await decompose_query(client, "What is the best dish?", system="...")
    assert result.intent == "best_item"
    assert result.complexity == "simple"


@pytest.mark.asyncio
async def test_invalid_json_retries_and_succeeds() -> None:
    invalid = "not valid json at all"
    valid = _valid_json("improvement")
    client = _make_client([invalid, valid])
    result = await decompose_query(client, "How can I improve?", system="...")
    assert result.intent == "improvement"
    assert client.complete_structured.call_count == 2


@pytest.mark.asyncio
async def test_both_attempts_fail_returns_safe_fallback() -> None:
    client = _make_client(["not json", "still not json"])
    result = await decompose_query(client, "Some user query", system="...")
    assert result.intent == "factual"
    assert result.complexity == "simple"
    assert "Some user query" in result.rephrased_query


@pytest.mark.asyncio
async def test_unknown_intent_accepted_by_schema() -> None:
    raw = json.dumps(
        {
            "intent": "out_of_scope",
            "complexity": "simple",
            "rephrased_query": "What is the weather?",
        }
    )
    client = _make_client([raw])
    result = await decompose_query(client, "What is the weather?", system="...")
    assert result.intent == "out_of_scope"


@pytest.mark.asyncio
async def test_markdown_fenced_json_fails_and_retries() -> None:
    fenced = "```json\n" + _valid_json("aggregation") + "\n```"
    clean = _valid_json("aggregation")
    client = _make_client([fenced, clean])
    result = await decompose_query(client, "How many reviews?", system="...")
    assert result.intent == "aggregation"


@pytest.mark.asyncio
async def test_decompose_passes_temperature_zero() -> None:
    client = _make_client([_valid_json()])
    await decompose_query(client, "test", system="sys")
    _, kwargs = client.complete_structured.call_args
    assert kwargs.get("temperature") == 0.0


@pytest.mark.asyncio
async def test_null_rephrased_query_does_not_raise() -> None:
    """Regression test: Groq (temperature=0.0) was confirmed live to return
    "rephrased_query": null for one specific prompt -- deterministic, so every
    retry against a different API key reproduced the identical validation
    failure, wastefully cycling through all 8 keys (each several seconds)
    before ever reaching this function's own corrective retry. The schema
    itself now coerces None to "" for this field (see DecomposedQuery in
    schemas.py), so a single valid-shaped response with a null field succeeds
    immediately -- no retry needed at all.
    """
    raw = json.dumps(
        {
            "intent": "factual",
            "complexity": "simple",
            "rephrased_query": None,
            "entities": None,
            "sub_queries": None,
        }
    )
    client = _make_client([raw])
    result = await decompose_query(client, "some query", system="...")
    assert result.rephrased_query == ""
    assert result.entities == []
    assert result.sub_queries == []
    assert client.complete_structured.call_count == 1
