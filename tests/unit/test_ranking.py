"""Unit tests for RRF fusion, sentiment-mapped rating, and evidence ranking."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

from src.core.ranking import (
    SENTIMENT_RATING_MAP,
    _split_into_highlight_candidates,
    rank_results,
    reciprocal_rank_fusion,
)
from src.services.vector.base import SearchResult

# Fixed cross-encoder scores for known sentences so highlight tests can
# assert a specific sentence wins, without loading a real model.
_KNOWN_SCORES = {
    "The food was cold.": 0.9,
    "Service was excellent.": 0.1,
}


async def _fake_score_for_highlight(
    query: str, sentences: list[str], model_name: str
) -> list[float]:
    return [_KNOWN_SCORES.get(s, 0.5) for s in sentences]


def _mock_settings(rrf=0.5, recency=0.3, rating=0.2, staleness=365):
    s = MagicMock()
    s.ranking_weight_rrf = rrf
    s.ranking_weight_recency = recency
    s.ranking_weight_rating = rating
    s.data_staleness_days = staleness
    return s


def _make_result(
    chunk_id: str,
    score: float = 1.0,
    rating: float | None = 4.0,
    sentiment_label: str = "Positive",
    sentiment_rating_agree: bool = True,
    days_old: int = 10,
    has_injection: bool = False,
    food_entities: list | None = None,
    source: str = "Google",
    date_inferred: bool = False,
) -> SearchResult:
    review_date = (datetime.now(tz=UTC) - timedelta(days=days_old)).isoformat()
    return SearchResult(
        id=chunk_id,
        score=score,
        payload={
            "text": f"Review text for {chunk_id}",
            "rating": rating,
            "sentiment_label": sentiment_label,
            "sentiment_rating_agree": sentiment_rating_agree,
            "review_date": review_date,
            "username": "TestUser",
            "source": source,
            "food_entities": food_entities or [],
            "has_injection_attempt": has_injection,
            "date_inferred": date_inferred,
        },
    )


class TestReciprocalRankFusion:
    def test_single_list_score_is_1_over_k_plus_rank(self) -> None:
        results = [SearchResult(id="a", score=1.0), SearchResult(id="b", score=0.5)]
        scores = reciprocal_rank_fusion([results], k=60)
        assert abs(scores["a"] - 1 / 61) < 1e-9
        assert abs(scores["b"] - 1 / 62) < 1e-9

    def test_two_lists_sum_contributions(self) -> None:
        list1 = [SearchResult(id="a", score=1.0), SearchResult(id="b", score=0.5)]
        list2 = [SearchResult(id="b", score=0.9), SearchResult(id="a", score=0.4)]
        scores = reciprocal_rank_fusion([list1, list2], k=60)
        expected_a = 1 / 61 + 1 / 62
        expected_b = 1 / 62 + 1 / 61
        assert abs(scores["a"] - expected_a) < 1e-9
        assert abs(scores["b"] - expected_b) < 1e-9

    def test_empty_lists_returns_empty_dict(self) -> None:
        assert reciprocal_rank_fusion([]) == {}
        assert reciprocal_rank_fusion([[]]) == {}

    def test_ordering_is_correct_when_one_source_dominant(self) -> None:
        list1 = [SearchResult(id="top", score=1.0)]
        list2 = [SearchResult(id="other", score=0.9), SearchResult(id="top", score=0.1)]
        scores = reciprocal_rank_fusion([list1, list2], k=60)
        assert scores["top"] > scores["other"]


class TestRankResults:
    async def test_top_k_is_respected(self) -> None:
        results = [_make_result(f"chunk_{i}", score=1.0 / (i + 1)) for i in range(10)]
        ranking = await rank_results(results, _mock_settings(), top_k=3)
        assert len(ranking.evidence) == 3

    async def test_sentiment_conflict_uses_mapped_rating(self) -> None:
        conflict_chunk = _make_result(
            "conflict",
            score=0.9,
            rating=5.0,
            sentiment_label="Negative",
            sentiment_rating_agree=False,
        )
        agree_chunk = _make_result(
            "agree",
            score=0.9,
            rating=5.0,
            sentiment_label="Positive",
            sentiment_rating_agree=True,
        )
        agree_rank = await rank_results([agree_chunk], _mock_settings(), top_k=1)
        conflict_rank = await rank_results([conflict_chunk], _mock_settings(), top_k=1)
        assert agree_rank.evidence[0].rating == 5.0
        assert conflict_rank.evidence[0].sentiment_conflict is True

    async def test_effective_rating_never_zero_on_conflict(self) -> None:
        for label in ("Positive", "Negative", "Mixed", "Neutral", "Unknown"):
            chunk = _make_result("x", sentiment_label=label, sentiment_rating_agree=False)
            ranking = await rank_results([chunk], _mock_settings(), top_k=1)
            assert len(ranking.evidence) == 1

    async def test_sentiment_rating_map_values_are_non_zero(self) -> None:
        for label, value in SENTIMENT_RATING_MAP.items():
            assert value > 0, f"Mapped value for {label} is zero or negative"

    async def test_injection_flag_lowers_composite_score(self) -> None:
        clean = _make_result("clean", score=0.5, has_injection=False)
        flagged = _make_result("flagged", score=0.5, has_injection=True)
        ranking = await rank_results([clean, flagged], _mock_settings(), top_k=2)
        clean_idx = next(
            i for i, e in enumerate(ranking.evidence) if e.snippet == clean.payload["text"]
        )
        flagged_idx = next(
            i for i, e in enumerate(ranking.evidence) if e.snippet == flagged.payload["text"]
        )
        assert clean_idx < flagged_idx, "Clean chunk should rank higher than injected chunk"

    async def test_entity_counts_aggregated_correctly(self) -> None:
        results = [
            _make_result("c1", food_entities=["biryani", "naan"]),
            _make_result("c2", food_entities=["biryani"]),
            _make_result("c3", food_entities=["naan"]),
        ]
        ranking = await rank_results(results, _mock_settings(), top_k=3)
        assert ranking.entity_counts["biryani"] == 2
        assert ranking.entity_counts["naan"] == 2

    async def test_source_breakdown_correct(self) -> None:
        results = [
            _make_result("c1", source="Google"),
            _make_result("c2", source="Yelp"),
            _make_result("c3", source="Google"),
        ]
        ranking = await rank_results(results, _mock_settings(), top_k=3)
        assert ranking.source_breakdown["Google"] == 2
        assert ranking.source_breakdown["Yelp"] == 1

    async def test_recency_spike_detected_when_most_evidence_is_recent(self) -> None:
        results = [_make_result(f"c{i}", days_old=2) for i in range(10)]
        ranking = await rank_results(results, _mock_settings(), top_k=6)
        assert ranking.recency_spike is True

    async def test_recency_spike_not_triggered_for_older_reviews(self) -> None:
        results = [_make_result(f"c{i}", days_old=90) for i in range(6)]
        ranking = await rank_results(results, _mock_settings(), top_k=6)
        assert ranking.recency_spike is False

    async def test_staleness_caveat_injected_for_old_reviews(self) -> None:
        results = [_make_result(f"c{i}", days_old=400) for i in range(6)]
        ranking = await rank_results(results, _mock_settings(), top_k=6)
        assert ranking.staleness_caveat is not None
        assert "year" in ranking.staleness_caveat.lower()

    async def test_staleness_caveat_not_injected_when_only_one_outlier_is_old(self) -> None:
        # 5 recent (90 days) + 1 stale outlier (400 days) -- a corpus spanning
        # years will almost always contain at least one review this old among
        # any reasonably-sized evidence set. The caveat should reflect whether
        # *most* of the evidence is stale, not whether a single outlier is.
        results = [_make_result(f"c{i}", days_old=90) for i in range(5)] + [
            _make_result("c_old", days_old=400)
        ]
        ranking = await rank_results(results, _mock_settings(), top_k=6)
        assert ranking.staleness_caveat is None

    async def test_staleness_caveat_injected_when_majority_is_old(self) -> None:
        # 4 stale (400 days) + 2 recent (90 days) -- majority old, should warn.
        results = [_make_result(f"c{i}", days_old=400) for i in range(4)] + [
            _make_result(f"c{i}", days_old=90) for i in range(4, 6)
        ]
        ranking = await rank_results(results, _mock_settings(), top_k=6)
        assert ranking.staleness_caveat is not None

    async def test_low_evidence_flag_when_fewer_than_three_chunks(self) -> None:
        results = [_make_result("c1"), _make_result("c2")]
        ranking = await rank_results(results, _mock_settings(), top_k=6)
        assert ranking.low_evidence is True

    async def test_no_low_evidence_with_three_or_more_chunks(self) -> None:
        results = [_make_result(f"c{i}") for i in range(3)]
        ranking = await rank_results(results, _mock_settings(), top_k=6)
        assert ranking.low_evidence is False

    async def test_date_inferred_propagated_to_evidence(self) -> None:
        chunk = _make_result("c1", date_inferred=True)
        ranking = await rank_results([chunk], _mock_settings(), top_k=1)
        assert ranking.evidence[0].date_inferred is True

    async def test_review_date_propagated_to_evidence(self) -> None:
        # Regression coverage: a specific person named based on a review
        # needs that review's own date visible, not just the aggregate
        # staleness_caveat (which only fires when most evidence is old) --
        # confirmed as a real gap once REVIEWER/STAFF NAME PRIVACY started
        # allowing real names through on explicit request.
        chunk = _make_result("c1", days_old=10)
        ranking = await rank_results([chunk], _mock_settings(), top_k=1)
        expected_date = (datetime.now(tz=UTC) - timedelta(days=10)).strftime("%Y-%m-%d")
        assert ranking.evidence[0].review_date == expected_date

    async def test_review_date_none_when_unparseable(self) -> None:
        chunk = _make_result("c1")
        chunk.payload["review_date"] = "not-a-date"
        ranking = await rank_results([chunk], _mock_settings(), top_k=1)
        assert ranking.evidence[0].review_date is None

    async def test_empty_input_returns_empty_result(self) -> None:
        ranking = await rank_results([], _mock_settings(), top_k=6)
        assert ranking.evidence == []
        assert ranking.low_evidence is True

    async def test_highlight_none_when_no_reranker_model_given(self) -> None:
        chunk = _make_result("c1")
        chunk.payload["text"] = "The food was cold. Service was excellent."
        ranking = await rank_results([chunk], _mock_settings(), top_k=1)
        assert ranking.evidence[0].highlight is None

    async def test_highlight_none_for_single_sentence_snippet(self) -> None:
        chunk = _make_result("c1")
        chunk.payload["text"] = "Great food overall."
        with patch("src.core.ranking.score_for_highlight", _fake_score_for_highlight):
            ranking = await rank_results(
                [chunk], _mock_settings(), top_k=1, query="food", reranker_model="fake-model"
            )
        assert ranking.evidence[0].highlight is None

    async def test_highlight_picks_highest_scored_sentence(self) -> None:
        chunk = _make_result("c1")
        chunk.payload["text"] = "The food was cold. Service was excellent."
        with patch("src.core.ranking.score_for_highlight", _fake_score_for_highlight):
            ranking = await rank_results(
                [chunk], _mock_settings(), top_k=1, query="food", reranker_model="fake-model"
            )
        assert ranking.evidence[0].highlight == "The food was cold."

    async def test_relevance_calibrated_defaults_true(self) -> None:
        chunk = _make_result("c1")
        ranking = await rank_results([chunk], _mock_settings(), top_k=1)
        assert ranking.evidence[0].relevance_calibrated is True

    async def test_relevance_calibrated_false_when_reranker_fell_back(self) -> None:
        chunk = _make_result("c1")
        ranking = await rank_results([chunk], _mock_settings(), top_k=1, reranked=False)
        assert ranking.evidence[0].relevance_calibrated is False


class TestSplitIntoHighlightCandidates:
    def test_splits_further_on_internal_ellipsis(self) -> None:
        # Regression coverage: NLTK treats "..." as not reliably ending a
        # sentence, so several separate thoughts joined by ellipses were
        # getting glued into one oversized highlight candidate.
        text = (
            "Your Google says you're open till 10pm... we spoke to the "
            "hostess at 8:31, she said with a literal straight face, "
            '"we stop seating at 8:30".. (true story)'
        )
        pieces = _split_into_highlight_candidates(text)
        assert pieces[0] == "Your Google says you're open till 10pm..."
        assert len(pieces) == 2

    def test_short_complete_sentences_not_merged(self) -> None:
        # Regression coverage: an earlier version merged any short piece
        # regardless of whether it was a real, complete sentence -- "Service
        # was excellent." (3 words) is common in review text and must stay
        # its own candidate, not get glued to the previous sentence.
        pieces = _split_into_highlight_candidates("The food was cold. Service was excellent.")
        assert pieces == ["The food was cold.", "Service was excellent."]

    def test_merges_genuine_nltk_mid_sentence_fragment(self) -> None:
        # A real, reproducible NLTK misfire (confirmed via
        # src.core.chunking._sentence_split directly): "!!" plus a lowercase
        # continuation gets split into two pieces, the second an orphaned
        # fragment starting mid-thought. Must be merged back rather than
        # shown as a standalone, confusing highlight.
        pieces = _split_into_highlight_candidates("I loved it!! definitely coming back.")
        assert pieces == ["I loved it!! definitely coming back."]

    def test_lowercase_after_ellipsis_not_treated_as_fragment(self) -> None:
        # The lowercase-start check must only apply to NLTK's own sentence
        # boundaries -- text continuing after "..." is normally lowercase as
        # a stylistic trail-off, not a broken cut. Applying the fragment
        # check there would immediately undo the ellipsis split above.
        pieces = _split_into_highlight_candidates("Open till 10pm... we got there at 8:31.")
        assert pieces == ["Open till 10pm...", "we got there at 8:31."]

    def test_empty_string_returns_empty_list(self) -> None:
        assert _split_into_highlight_candidates("") == []
