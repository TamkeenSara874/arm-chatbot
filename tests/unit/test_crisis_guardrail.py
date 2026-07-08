"""Unit tests for the crisis (self-harm language) guardrail."""

from unittest.mock import patch

from src.core.crisis_guardrail import CRISIS_RESPONSE, detect_crisis_language


class TestDetectsCrisisLanguage:
    def test_the_live_repro_case(self) -> None:
        # The exact phrasing that surfaced this gap live: decomposition still
        # classified the whole message as a normal count_query and the answer
        # went straight to review data with no acknowledgment at all.
        assert detect_crisis_language("i want to die, the reviews are so bad") is True

    def test_variants(self) -> None:
        for text in [
            "I want to die",
            "I'm suicidal because of these reviews",
            "having suicidal thoughts lately",
            "thinking about suicide",
            "I want to end my life",
            "ending it all tonight",
            "I don't want to live anymore",
            "there's no reason to live",
            "my life is not worth living",
            "I've been hurting myself",
            "I've been self-harming",
            "self harm is on my mind",
        ]:
            assert detect_crisis_language(text) is True, f"should have triggered on: {text!r}"

    def test_case_insensitive(self) -> None:
        assert detect_crisis_language("I WANT TO DIE") is True

    def test_embedded_in_a_normal_looking_business_question(self) -> None:
        assert detect_crisis_language("how many negative reviews, i want to die over this") is True


class TestDoesNotFalsePositiveOnBusinessLanguage:
    def test_sales_dying_is_not_flagged(self) -> None:
        assert detect_crisis_language("my sales are dying because of bad reviews") is False

    def test_kill_the_competition_is_not_flagged(self) -> None:
        assert detect_crisis_language("how do I kill the competition on ratings") is False

    def test_ordinary_review_question_is_not_flagged(self) -> None:
        assert detect_crisis_language("how many positive reviews do I have?") is False

    def test_empty_string_is_not_flagged(self) -> None:
        assert detect_crisis_language("") is False


class TestCounterAndResponse:
    def test_increments_counter_on_trigger(self) -> None:
        with patch("src.core.crisis_guardrail.guardrail_triggered_total") as mock_counter:
            mock_counter.labels.return_value.inc = lambda: None
            detect_crisis_language("I want to die")
            mock_counter.labels.assert_called_once_with(type="crisis_language")

    def test_does_not_increment_counter_when_not_triggered(self) -> None:
        with patch("src.core.crisis_guardrail.guardrail_triggered_total") as mock_counter:
            detect_crisis_language("how many reviews do I have")
            mock_counter.labels.assert_not_called()

    def test_response_offers_a_concrete_resource(self) -> None:
        assert "988" in CRISIS_RESPONSE
        assert "findahelpline.com" in CRISIS_RESPONSE

    def test_response_leads_with_empathy_not_data(self) -> None:
        # Must not read like a normal count-query answer (e.g. "You have N
        # reviews...") -- that's the exact failure mode this guardrail fixes.
        assert not CRISIS_RESPONSE.lower().startswith("you have")
        assert "sorry" in CRISIS_RESPONSE.lower().split(".")[0]

    def test_no_em_dashes_in_response(self) -> None:
        assert "—" not in CRISIS_RESPONSE
        assert "–" not in CRISIS_RESPONSE
