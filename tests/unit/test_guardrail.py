"""Unit tests for the guardrail module."""

from unittest.mock import patch

from src.core.guardrail import (
    GUARDRAIL_INTENTS,
    GUARDRAIL_RESPONSES,
    check_guardrail,
)


def test_out_of_scope_returns_canned_response() -> None:
    response = check_guardrail("out_of_scope")
    assert response is not None
    assert len(response) > 20
    assert "reviews" in response.lower()


def test_ui_question_returns_canned_response() -> None:
    response = check_guardrail("ui_question")
    assert response is not None
    assert "support team" in response.lower() or "app" in response.lower()


def test_manipulation_request_returns_canned_response() -> None:
    response = check_guardrail("manipulation_request")
    assert response is not None
    assert "not able to help" in response.lower()


def test_multi_location_returns_canned_response() -> None:
    response = check_guardrail("multi_location")
    assert response is not None
    assert "one restaurant at a time" in response.lower()


def test_allergen_returns_canned_response() -> None:
    response = check_guardrail("allergen")
    assert response is not None
    assert "kitchen team" in response.lower()


def test_valid_intent_passes_through() -> None:
    for intent in ("best_item", "worst_item", "factual", "sentiment_overview", "improvement"):
        assert check_guardrail(intent) is None, f"Intent '{intent}' should not be guardrailed"


def test_all_guardrail_intents_covered() -> None:
    for intent in GUARDRAIL_INTENTS:
        assert check_guardrail(intent) is not None, f"Intent '{intent}' missing response"


def test_guardrail_increments_counter() -> None:
    with patch("src.core.guardrail.guardrail_triggered_total") as mock_counter:
        mock_counter.labels.return_value.inc = lambda: None
        check_guardrail("out_of_scope")
        mock_counter.labels.assert_called_once_with(type="out_of_scope")


def test_no_em_dashes_in_responses() -> None:
    for response in GUARDRAIL_RESPONSES.values():
        assert "—" not in response, "Em dash found in guardrail response"
        assert "–" not in response, "En dash found in guardrail response"
