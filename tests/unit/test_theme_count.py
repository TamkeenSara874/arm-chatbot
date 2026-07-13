"""Unit tests for the theme-count fast path (_format_theme_count_answer).

Regression coverage for a real limitation: qualitative-theme count questions
("how many people called my staff rude") were answered honestly but only
from the top_k=20 retrieved sample ("at least 12 of the 20 retrieved
reviews..."), which could badly undercount a theme actually present in far
more of the full review set. src/core/review_stats.py's compute_theme_count
is covered separately in tests/unit/test_review_stats.py -- this file covers
the answer-formatting half.
"""

from src.api.routes.chat import (
    _format_theme_comparison_answer,
    _format_theme_cooccurrence_answer,
    _format_theme_count_answer,
)


class TestFormatThemeCountAnswer:
    def test_includes_exact_count(self) -> None:
        result = _format_theme_count_answer(38, ["rude", "unfriendly"])
        assert "38" in result

    def test_includes_all_keywords_quoted(self) -> None:
        result = _format_theme_count_answer(12, ["rude", "unfriendly", "hostile"])
        assert '"rude"' in result
        assert '"unfriendly"' in result
        assert '"hostile"' in result

    def test_zero_matches_still_states_zero(self) -> None:
        result = _format_theme_count_answer(0, ["undercooked"])
        assert "0 review" in result

    def test_notes_it_may_miss_differently_worded_reviews(self) -> None:
        result = _format_theme_count_answer(5, ["cold food"])
        assert "different words" in result

    def test_notes_covers_every_review_not_a_sample(self) -> None:
        result = _format_theme_count_answer(5, ["slow service"])
        assert "every review" in result
        assert "not just a sample" in result

    def test_never_exposes_internal_mechanism_language(self) -> None:
        # Regression test: this text is threaded verbatim into the generation
        # prompt as the caveat the model is told to keep, and "keyword
        # search"/"keyword match" leaked through to a live answer shown to a
        # non-technical restaurant owner -- rule 8 (both generation prompts)
        # bans this kind of internal-mechanism language, but the precomputed
        # text itself must not hand the model that phrasing to begin with.
        result = _format_theme_count_answer(5, ["rude"]).lower()
        assert "keyword" not in result
        assert "semantic" not in result


class TestFormatThemeComparisonAnswer:
    """Regression coverage for a real gap: 'which has more complaints: food
    quality or staff behavior, and by how much?' had decomposition try to set
    theme_keywords to the literal phrase "complaints about food quality"
    (matching zero reviews, since nobody writes that exact phrase), so the
    model fell back to eyeballing which theme seemed more common in only the
    20 retrieved reviews -- contradicting a previous, differently-sampled
    answer to a related question. Both themes now get an exact, whole-corpus
    count via two real keyword lists, same as a single theme_count question.
    """

    def test_states_both_exact_counts(self) -> None:
        result = _format_theme_comparison_answer(
            47, ["rude", "unfriendly"], 31, ["bland", "cold food"]
        )
        assert "47" in result
        assert "31" in result

    def test_states_the_real_margin(self) -> None:
        result = _format_theme_comparison_answer(
            47, ["rude", "unfriendly"], 31, ["bland", "cold food"]
        )
        assert "16 more review" in result

    def test_leader_is_whichever_theme_has_higher_count(self) -> None:
        result = _format_theme_comparison_answer(10, ["rude"], 25, ["bland"])
        assert '"bland" appears in 15 more review' in result

    def test_equal_counts_states_tie(self) -> None:
        result = _format_theme_comparison_answer(20, ["rude"], 20, ["bland"])
        assert "same number of reviews (20)" in result

    def test_never_exposes_internal_mechanism_language(self) -> None:
        result = _format_theme_comparison_answer(5, ["rude"], 3, ["bland"]).lower()
        assert "keyword" not in result
        assert "semantic" not in result


class TestFormatThemeCooccurrenceAnswer:
    """Regression coverage: "which reviews mention both slow service and cold
    food?" previously had no way to express an AND between two themes, so it
    got answered with an OR-matched count worded as if it meant "both
    together" -- directly contradicting the model's own read of the
    retrieved evidence. This states one real intersection count instead.
    """

    def test_states_the_exact_count(self) -> None:
        result = _format_theme_cooccurrence_answer(3, ["slow service"], ["cold food"])
        assert "3 review(s)" in result

    def test_states_both_keyword_groups(self) -> None:
        result = _format_theme_cooccurrence_answer(3, ["slow service", "slow"], ["cold food"])
        assert '"slow service"' in result
        assert '"slow"' in result
        assert '"cold food"' in result

    def test_zero_matches_still_states_zero(self) -> None:
        result = _format_theme_cooccurrence_answer(0, ["rude"], ["bland"])
        assert "0 review(s)" in result

    def test_never_exposes_internal_mechanism_language(self) -> None:
        result = _format_theme_cooccurrence_answer(2, ["rude"], ["bland"]).lower()
        assert "keyword" not in result
        assert "semantic" not in result
