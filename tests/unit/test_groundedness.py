"""Unit tests for the count-overclaim groundedness heuristic."""

from src.core.groundedness import check_count_groundedness


class TestCheckCountGroundedness:
    def test_count_within_evidence_is_grounded(self) -> None:
        answer = "7 of the 12 retrieved reviews mention slow service."
        assert check_count_groundedness(answer, evidence_count=12) is True

    def test_count_exceeding_evidence_is_flagged(self) -> None:
        answer = "50 reviews mention slow service."
        assert check_count_groundedness(answer, evidence_count=6) is False

    def test_no_numeric_claims_is_grounded(self) -> None:
        answer = "Customers frequently mention slow service and cold food."
        assert check_count_groundedness(answer, evidence_count=6) is True

    def test_star_ratings_are_not_treated_as_counts(self) -> None:
        answer = "Several reviewers gave 5 stars but 20 reviews mention the wait time."
        assert check_count_groundedness(answer, evidence_count=3) is False
        # Sanity: the 5-star mention alone should not trip it.
        answer_ok = "Several reviewers gave 5 stars overall."
        assert check_count_groundedness(answer_ok, evidence_count=3) is True

    def test_years_are_not_treated_as_counts(self) -> None:
        answer = "In 2024 several reviews mention improved service."
        assert check_count_groundedness(answer, evidence_count=3) is True

    def test_precomputed_count_exempts_any_claim(self) -> None:
        answer = "You have 500 positive reviews in total."
        assert (
            check_count_groundedness(answer, evidence_count=6, precomputed_count="500 reviews")
            is True
        )

    def test_empty_answer_is_grounded(self) -> None:
        assert check_count_groundedness("", evidence_count=0) is True
