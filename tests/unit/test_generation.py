"""Unit tests for src/core/generation.py -- extracted from _pipeline_stream
during the chat.py decomposition refactor. These functions previously had
zero direct test coverage (only reachable through the full HTTP/SSE pipeline).
"""

from unittest.mock import MagicMock

from src.core.generation import (
    NO_EVIDENCE_ANSWER,
    build_generation_prompt,
    build_structured_response,
    check_hallucination_gate,
    clean_answer_text,
    estimate_confidence,
    format_evidence,
    select_generation,
)
from src.core.ranking import RankingResult
from src.models.schemas import DecomposedQuery, EvidenceItem, SubAnswer


def _evidence_item(
    relevance: float = 0.5,
    sentiment_conflict: bool = False,
    rating: float | None = 4.0,
    source: str | None = "Google",
    sentiment: str | None = "Positive",
    date_inferred: bool = False,
    username: str | None = None,
    review_date: str | None = None,
) -> EvidenceItem:
    return EvidenceItem(
        snippet="great food",
        username=username,
        rating=rating,
        source=source,
        sentiment=sentiment,
        sentiment_conflict=sentiment_conflict,
        date_inferred=date_inferred,
        review_date=review_date,
        relevance=relevance,
    )


def _ranking_result(
    evidence: list[EvidenceItem] | None = None,
    low_evidence: bool = False,
    staleness_caveat: str | None = None,
) -> RankingResult:
    return RankingResult(
        evidence=evidence if evidence is not None else [_evidence_item()],
        entity_counts={},
        source_breakdown={},
        recency_spike=False,
        staleness_caveat=staleness_caveat,
        low_evidence=low_evidence,
    )


class TestFormatEvidence:
    def test_includes_rating_source_sentiment(self) -> None:
        result = format_evidence([_evidence_item(rating=5.0, source="Yelp", sentiment="Positive")])
        assert "Rating: 5.0/5" in result
        assert "Source: Yelp" in result
        assert "Sentiment: Positive" in result

    def test_rating_none_shows_na(self) -> None:
        result = format_evidence([_evidence_item(rating=None)])
        assert "Rating: N/A" in result

    def test_includes_reviewer_username_when_present(self) -> None:
        result = format_evidence([_evidence_item(username="Jane Doe")])
        assert "Reviewer: Jane Doe" in result

    def test_omits_reviewer_field_when_username_absent(self) -> None:
        result = format_evidence([_evidence_item(username=None)])
        assert "Reviewer:" not in result

    def test_sentiment_conflict_flag_included(self) -> None:
        result = format_evidence([_evidence_item(sentiment_conflict=True)])
        assert "[sentiment_conflict: true]" in result

    def test_date_inferred_flag_included(self) -> None:
        result = format_evidence([_evidence_item(date_inferred=True)])
        assert "[date_inferred: true]" in result

    def test_includes_review_date_when_present(self) -> None:
        # Regression coverage: naming a specific person (staff or reviewer)
        # based on a review needs its date visible per-item, not just the
        # aggregate staleness_caveat -- a single old review is a very
        # different basis for a real decision (e.g. firing someone) than a
        # recent, repeated pattern, and the model can't say so without this.
        result = format_evidence([_evidence_item(review_date="2024-03-15")])
        assert "Date: 2024-03-15" in result

    def test_omits_date_field_when_absent(self) -> None:
        result = format_evidence([_evidence_item(review_date=None)])
        assert "Date:" not in result

    def test_empty_list_returns_placeholder(self) -> None:
        assert format_evidence([]) == "No review evidence found."

    def test_multiple_items_numbered(self) -> None:
        result = format_evidence([_evidence_item(), _evidence_item()])
        assert "BEGIN REVIEW 1" in result
        assert "BEGIN REVIEW 2" in result


class TestEstimateConfidence:
    def test_low_evidence_uses_04_base(self) -> None:
        ranked = _ranking_result(low_evidence=True)
        assert estimate_confidence(ranked) == 0.4

    def test_staleness_caveat_uses_06_base(self) -> None:
        ranked = _ranking_result(staleness_caveat="stale")
        assert estimate_confidence(ranked) == 0.6

    def test_normal_path_uses_avg_relevance(self) -> None:
        ranked = _ranking_result(evidence=[_evidence_item(relevance=0.8)])
        # base = min(0.95, 0.5 + 0.8*0.5) = min(0.95, 0.9) = 0.9
        assert estimate_confidence(ranked) == 0.9

    def test_no_evidence_no_low_no_stale_uses_05_base(self) -> None:
        ranked = _ranking_result(evidence=[])
        assert estimate_confidence(ranked) == 0.5

    def test_sentiment_conflict_discounts_confidence(self) -> None:
        ranked = _ranking_result(evidence=[_evidence_item(relevance=0.8, sentiment_conflict=True)])
        # base = 0.9, discounted by (1 - 0.4*1.0) = 0.6 -> 0.54
        assert estimate_confidence(ranked) == 0.54

    def test_groundedness_failure_halves_confidence(self) -> None:
        ranked = _ranking_result(evidence=[_evidence_item(relevance=0.8)])
        assert estimate_confidence(ranked, groundedness_ok=False) == 0.45

    def test_result_rounded_to_three_places(self) -> None:
        ranked = _ranking_result(evidence=[_evidence_item(relevance=0.333)])
        result = estimate_confidence(ranked)
        assert result == round(result, 3)


class TestCheckHallucinationGate:
    def test_no_evidence_no_precomputed_count_triggers_gate(self) -> None:
        ranked = _ranking_result(evidence=[])
        assert check_hallucination_gate(ranked, None) == NO_EVIDENCE_ANSWER

    def test_no_evidence_with_precomputed_count_does_not_trigger(self) -> None:
        ranked = _ranking_result(evidence=[])
        assert check_hallucination_gate(ranked, "5") is None

    def test_has_evidence_does_not_trigger(self) -> None:
        ranked = _ranking_result(evidence=[_evidence_item()])
        assert check_hallucination_gate(ranked, None) is None

    def test_no_evidence_with_precomputed_trend_does_not_trigger(self) -> None:
        ranked = _ranking_result(evidence=[])
        assert check_hallucination_gate(ranked, None, "10 reviews | 5 reviews") is None

    def test_no_evidence_no_count_no_trend_triggers_gate(self) -> None:
        ranked = _ranking_result(evidence=[])
        assert check_hallucination_gate(ranked, None, None) == NO_EVIDENCE_ANSWER

    def test_no_evidence_with_precomputed_breakdown_does_not_trigger(self) -> None:
        ranked = _ranking_result(evidence=[])
        assert check_hallucination_gate(ranked, None, None, "Google: 13, Yelp: 1") is None

    def test_no_evidence_no_count_no_trend_no_breakdown_triggers_gate(self) -> None:
        ranked = _ranking_result(evidence=[])
        assert check_hallucination_gate(ranked, None, None, None) == NO_EVIDENCE_ANSWER


class TestSelectGeneration:
    def _settings(self) -> MagicMock:
        settings = MagicMock()
        settings.openai_simple_model = "gpt-4o-mini"
        settings.openai_complex_model = "gpt-4.1"
        return settings

    def test_simple_complexity_selects_simple_model(self) -> None:
        decomposed = DecomposedQuery(intent="factual", complexity="simple")
        selection = select_generation(decomposed, None, self._settings())
        assert selection.is_complex is False
        assert selection.model_used == "gpt-4o-mini"
        assert selection.prompt_name == "chat_response_simple"

    def test_complex_complexity_selects_complex_model(self) -> None:
        decomposed = DecomposedQuery(intent="factual", complexity="complex")
        selection = select_generation(decomposed, None, self._settings())
        assert selection.is_complex is True
        assert selection.model_used == "gpt-4.1"
        assert selection.prompt_name == "chat_response_complex"

    def test_precomputed_count_forces_complex_even_if_simple(self) -> None:
        decomposed = DecomposedQuery(intent="count_query", complexity="simple")
        selection = select_generation(decomposed, "42", self._settings())
        assert selection.is_complex is True
        assert selection.model_used == "gpt-4.1"

    def test_precomputed_trend_forces_complex_even_if_simple(self) -> None:
        decomposed = DecomposedQuery(intent="comparison", complexity="simple")
        selection = select_generation(decomposed, None, self._settings(), "10 reviews | 5 reviews")
        assert selection.is_complex is True
        assert selection.model_used == "gpt-4.1"
        assert selection.prompt_name == "chat_response_complex"

    def test_precomputed_breakdown_forces_complex_even_if_simple(self) -> None:
        decomposed = DecomposedQuery(intent="aggregation", complexity="simple")
        selection = select_generation(
            decomposed, None, self._settings(), None, "Google: 13, Yelp: 1"
        )
        assert selection.is_complex is True
        assert selection.model_used == "gpt-4.1"
        assert selection.prompt_name == "chat_response_complex"


class TestBuildGenerationPrompt:
    def test_simple_prompt_omits_complex_only_kwargs(self) -> None:
        loader = MagicMock()
        loader.format.return_value = ("system", "user")
        build_generation_prompt(
            loader,
            "chat_response_simple",
            False,
            query="q",
            session_context="ctx",
            corrections="None",
            evidence="ev",
        )
        _, kwargs = loader.format.call_args
        assert "sub_queries" not in kwargs
        assert "exact_count" not in kwargs
        assert kwargs["query"] == "q"

    def test_complex_prompt_includes_extra_kwargs(self) -> None:
        loader = MagicMock()
        loader.format.return_value = ("system", "user")
        build_generation_prompt(
            loader,
            "chat_response_complex",
            True,
            query="q",
            session_context="ctx",
            corrections="None",
            evidence="ev",
            sub_queries=["a", "b"],
            entity_counts={"tacos": 2},
            source_breakdown={"Google": 1},
            recency_spike=True,
            exact_count="7",
        )
        _, kwargs = loader.format.call_args
        assert kwargs["sub_queries"] == '["a", "b"]'
        assert kwargs["entity_counts"] == '{"tacos": 2}'
        assert kwargs["recency_spike"] == "true"
        assert kwargs["exact_count"] == "7"

    def test_complex_prompt_defaults_exact_count_to_none_string(self) -> None:
        loader = MagicMock()
        loader.format.return_value = ("system", "user")
        build_generation_prompt(
            loader,
            "chat_response_complex",
            True,
            query="q",
            session_context="ctx",
            corrections="None",
            evidence="ev",
        )
        _, kwargs = loader.format.call_args
        assert kwargs["exact_count"] == "None"

    def test_complex_prompt_defaults_trend_comparison_to_none_string(self) -> None:
        loader = MagicMock()
        loader.format.return_value = ("system", "user")
        build_generation_prompt(
            loader,
            "chat_response_complex",
            True,
            query="q",
            session_context="ctx",
            corrections="None",
            evidence="ev",
        )
        _, kwargs = loader.format.call_args
        assert kwargs["trend_comparison"] == "None"

    def test_complex_prompt_passes_trend_comparison_through(self) -> None:
        loader = MagicMock()
        loader.format.return_value = ("system", "user")
        build_generation_prompt(
            loader,
            "chat_response_complex",
            True,
            query="q",
            session_context="ctx",
            corrections="None",
            evidence="ev",
            trend_comparison="10 reviews | 5 reviews",
        )
        _, kwargs = loader.format.call_args
        assert kwargs["trend_comparison"] == "10 reviews | 5 reviews"

    def test_complex_prompt_defaults_exact_breakdown_to_none_string(self) -> None:
        loader = MagicMock()
        loader.format.return_value = ("system", "user")
        build_generation_prompt(
            loader,
            "chat_response_complex",
            True,
            query="q",
            session_context="ctx",
            corrections="None",
            evidence="ev",
        )
        _, kwargs = loader.format.call_args
        assert kwargs["exact_breakdown"] == "None"

    def test_complex_prompt_passes_exact_breakdown_through(self) -> None:
        loader = MagicMock()
        loader.format.return_value = ("system", "user")
        build_generation_prompt(
            loader,
            "chat_response_complex",
            True,
            query="q",
            session_context="ctx",
            corrections="None",
            evidence="ev",
            exact_breakdown="Google: 13, Yelp: 1",
        )
        _, kwargs = loader.format.call_args
        assert kwargs["exact_breakdown"] == "Google: 13, Yelp: 1"

    def test_unverified_note_defaults_to_none_string(self) -> None:
        loader = MagicMock()
        loader.format.return_value = ("system", "user")
        build_generation_prompt(
            loader,
            "chat_response_simple",
            False,
            query="q",
            session_context="ctx",
            corrections="None",
            evidence="ev",
        )
        _, kwargs = loader.format.call_args
        assert kwargs["unverified_note"] == "None"

    def test_unverified_note_is_passed_through_distinct_from_corrections(self) -> None:
        loader = MagicMock()
        loader.format.return_value = ("system", "user")
        build_generation_prompt(
            loader,
            "chat_response_simple",
            False,
            query="q",
            session_context="ctx",
            corrections="None",
            unverified_note="One user flagged the service as slow.",
            evidence="ev",
        )
        _, kwargs = loader.format.call_args
        assert kwargs["unverified_note"] == "One user flagged the service as slow."
        assert kwargs["corrections"] == "None"


class TestCleanAnswerText:
    def test_plain_text_passthrough(self) -> None:
        assert clean_answer_text("hello world") == "hello world"

    def test_strips_markdown_fence(self) -> None:
        assert clean_answer_text("```\nhello\n```") == "hello"

    def test_strips_language_tagged_fence(self) -> None:
        assert clean_answer_text("```json\nhello\n```") == "hello"

    def test_unwraps_json_envelope_with_answer_key(self) -> None:
        assert clean_answer_text('{"answer": "hi there"}') == "hi there"

    def test_malformed_json_left_unchanged(self) -> None:
        raw = "{not valid json"
        assert clean_answer_text(raw) == raw

    def test_json_without_answer_key_left_unchanged(self) -> None:
        raw = '{"foo": "bar"}'
        assert clean_answer_text(raw) == raw


class TestBuildStructuredResponse:
    def test_wires_answer_evidence_and_confidence(self) -> None:
        ranked = _ranking_result(evidence=[_evidence_item(relevance=0.8)])
        sub_answers: list[SubAnswer] = []
        structured = build_structured_response("the answer", sub_answers, ranked, True)
        assert structured.answer == "the answer"
        assert structured.evidence == ranked.evidence
        assert structured.confidence == estimate_confidence(ranked, True)
        assert structured.caveats == ranked.staleness_caveat
