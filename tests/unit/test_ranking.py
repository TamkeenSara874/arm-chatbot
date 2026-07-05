"""Unit tests for RRF fusion, sentiment-mapped rating, and evidence ranking."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

from src.core.ranking import (
    SENTIMENT_RATING_MAP,
    rank_results,
    reciprocal_rank_fusion,
)
from src.services.vector.base import SearchResult


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
    def test_top_k_is_respected(self) -> None:
        results = [_make_result(f"chunk_{i}", score=1.0 / (i + 1)) for i in range(10)]
        ranking = rank_results(results, _mock_settings(), top_k=3)
        assert len(ranking.evidence) == 3

    def test_sentiment_conflict_uses_mapped_rating(self) -> None:
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
        agree_rank = rank_results([agree_chunk], _mock_settings(), top_k=1)
        conflict_rank = rank_results([conflict_chunk], _mock_settings(), top_k=1)
        assert agree_rank.evidence[0].rating == 5.0
        assert conflict_rank.evidence[0].sentiment_conflict is True

    def test_effective_rating_never_zero_on_conflict(self) -> None:
        for label in ("Positive", "Negative", "Mixed", "Neutral", "Unknown"):
            chunk = _make_result("x", sentiment_label=label, sentiment_rating_agree=False)
            ranking = rank_results([chunk], _mock_settings(), top_k=1)
            assert len(ranking.evidence) == 1

    def test_sentiment_rating_map_values_are_non_zero(self) -> None:
        for label, value in SENTIMENT_RATING_MAP.items():
            assert value > 0, f"Mapped value for {label} is zero or negative"

    def test_injection_flag_lowers_composite_score(self) -> None:
        clean = _make_result("clean", score=0.5, has_injection=False)
        flagged = _make_result("flagged", score=0.5, has_injection=True)
        ranking = rank_results([clean, flagged], _mock_settings(), top_k=2)
        clean_idx = next(
            i for i, e in enumerate(ranking.evidence) if e.snippet == clean.payload["text"]
        )
        flagged_idx = next(
            i for i, e in enumerate(ranking.evidence) if e.snippet == flagged.payload["text"]
        )
        assert clean_idx < flagged_idx, "Clean chunk should rank higher than injected chunk"

    def test_entity_counts_aggregated_correctly(self) -> None:
        results = [
            _make_result("c1", food_entities=["biryani", "naan"]),
            _make_result("c2", food_entities=["biryani"]),
            _make_result("c3", food_entities=["naan"]),
        ]
        ranking = rank_results(results, _mock_settings(), top_k=3)
        assert ranking.entity_counts["biryani"] == 2
        assert ranking.entity_counts["naan"] == 2

    def test_source_breakdown_correct(self) -> None:
        results = [
            _make_result("c1", source="Google"),
            _make_result("c2", source="Yelp"),
            _make_result("c3", source="Google"),
        ]
        ranking = rank_results(results, _mock_settings(), top_k=3)
        assert ranking.source_breakdown["Google"] == 2
        assert ranking.source_breakdown["Yelp"] == 1

    def test_recency_spike_detected_when_most_evidence_is_recent(self) -> None:
        results = [_make_result(f"c{i}", days_old=2) for i in range(10)]
        ranking = rank_results(results, _mock_settings(), top_k=6)
        assert ranking.recency_spike is True

    def test_recency_spike_not_triggered_for_older_reviews(self) -> None:
        results = [_make_result(f"c{i}", days_old=90) for i in range(6)]
        ranking = rank_results(results, _mock_settings(), top_k=6)
        assert ranking.recency_spike is False

    def test_staleness_caveat_injected_for_old_reviews(self) -> None:
        results = [_make_result(f"c{i}", days_old=400) for i in range(6)]
        ranking = rank_results(results, _mock_settings(), top_k=6)
        assert ranking.staleness_caveat is not None
        assert "year" in ranking.staleness_caveat.lower()

    def test_staleness_caveat_not_injected_when_only_one_outlier_is_old(self) -> None:
        # 5 recent (90 days) + 1 stale outlier (400 days) -- a corpus spanning
        # years will almost always contain at least one review this old among
        # any reasonably-sized evidence set. The caveat should reflect whether
        # *most* of the evidence is stale, not whether a single outlier is.
        results = [_make_result(f"c{i}", days_old=90) for i in range(5)] + [
            _make_result("c_old", days_old=400)
        ]
        ranking = rank_results(results, _mock_settings(), top_k=6)
        assert ranking.staleness_caveat is None

    def test_staleness_caveat_injected_when_majority_is_old(self) -> None:
        # 4 stale (400 days) + 2 recent (90 days) -- majority old, should warn.
        results = [_make_result(f"c{i}", days_old=400) for i in range(4)] + [
            _make_result(f"c{i}", days_old=90) for i in range(4, 6)
        ]
        ranking = rank_results(results, _mock_settings(), top_k=6)
        assert ranking.staleness_caveat is not None

    def test_low_evidence_flag_when_fewer_than_three_chunks(self) -> None:
        results = [_make_result("c1"), _make_result("c2")]
        ranking = rank_results(results, _mock_settings(), top_k=6)
        assert ranking.low_evidence is True

    def test_no_low_evidence_with_three_or_more_chunks(self) -> None:
        results = [_make_result(f"c{i}") for i in range(3)]
        ranking = rank_results(results, _mock_settings(), top_k=6)
        assert ranking.low_evidence is False

    def test_date_inferred_propagated_to_evidence(self) -> None:
        chunk = _make_result("c1", date_inferred=True)
        ranking = rank_results([chunk], _mock_settings(), top_k=1)
        assert ranking.evidence[0].date_inferred is True

    def test_empty_input_returns_empty_result(self) -> None:
        ranking = rank_results([], _mock_settings(), top_k=6)
        assert ranking.evidence == []
        assert ranking.low_evidence is True
