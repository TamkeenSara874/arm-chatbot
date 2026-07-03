from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime

from src.config import Settings
from src.models.schemas import EvidenceItem
from src.services.vector.base import SearchResult

SENTIMENT_RATING_MAP: dict[str, float] = {
    "Positive": 4.5,
    "Negative": 1.5,
    "Mixed": 3.0,
    "Neutral": 3.0,
}

LOW_EVIDENCE_THRESHOLD = 3
RECENCY_SPIKE_DAYS = 7
RECENCY_SPIKE_RATIO = 0.4


@dataclass
class RankingResult:
    evidence: list[EvidenceItem]
    entity_counts: dict[str, int]
    source_breakdown: dict[str, int]
    recency_spike: bool
    staleness_caveat: str | None
    low_evidence: bool


def _effective_rating(payload: dict) -> float:
    """Sentiment-mapped rating used when text sentiment disagrees with the star rating.

    Shared by the composite-score loop and the evidence-building step so the
    number the LLM/frontend sees always matches what ranking actually used --
    previously the raw star rating leaked into EvidenceItem even when
    sentiment_conflict was True.
    """
    rating = payload.get("rating")
    sentiment_rating_agree = payload.get("sentiment_rating_agree", True)
    if rating is not None and sentiment_rating_agree:
        return rating
    return SENTIMENT_RATING_MAP.get(payload.get("sentiment_label") or "", 3.0)


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    k: int = 60,
) -> dict[str, float]:
    """Merge multiple ranked lists via Reciprocal Rank Fusion.

    k=60 is the empirically recommended default from Cormack et al. (2009) for
    balanced fusion across sources with unequal result counts.
    Returns a dict of chunk_id -> fused RRF score.
    """
    scores: dict[str, float] = {}
    for result_list in result_lists:
        for rank, result in enumerate(result_list, start=1):
            scores[result.id] = scores.get(result.id, 0.0) + 1.0 / (k + rank)
    return scores


def rank_results(
    fused_results: list[SearchResult],
    settings: Settings,
    today: datetime | None = None,
    top_k: int = 6,
    staleness_days: int | None = None,
    has_explicit_date_filter: bool = False,
) -> RankingResult:
    """Apply composite scoring and return ranked evidence with diagnostics.

    Each result's .score field must be the RRF score produced by reciprocal_rank_fusion().
    The composite score formula is:
        injection_penalty * (w_rrf*rrf + w_recency*(1/(days+1)) + w_rating*(effective_rating/5))
    where effective_rating is sentiment-mapped when text sentiment and star rating disagree.
    """
    if today is None:
        today = datetime.now(tz=UTC)
    if today.tzinfo is None:
        today = today.replace(tzinfo=UTC)

    if staleness_days is None:
        staleness_days = settings.data_staleness_days

    w_rrf = settings.ranking_weight_rrf
    w_recency = settings.ranking_weight_recency
    w_rating = settings.ranking_weight_rating

    scored: list[tuple[float, SearchResult]] = []

    for result in fused_results:
        p = result.payload
        rrf_score = result.score

        effective_rating = _effective_rating(p)
        rating_contribution = effective_rating / 5.0

        review_date = _parse_date(p.get("review_date"))
        days_old = max(0, (today - review_date).days) if review_date else staleness_days

        recency_score = 1.0 / (days_old + 1)
        injection_penalty = 0.5 if p.get("has_injection_attempt", False) else 1.0

        composite = injection_penalty * (
            w_rrf * rrf_score + w_recency * recency_score + w_rating * rating_contribution
        )
        scored.append((composite, result))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    low_evidence = len(top) < LOW_EVIDENCE_THRESHOLD

    entity_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    dates_in_top: list[datetime] = []
    recent_count = 0

    for _, result in top:
        p = result.payload
        for entity in p.get("food_entities", []):
            entity_counter[entity] += 1
        src = p.get("source")
        if src:
            source_counter[src] += 1
        rdate = _parse_date(p.get("review_date"))
        if rdate:
            dates_in_top.append(rdate)
            if (today - rdate).days <= RECENCY_SPIKE_DAYS:
                recent_count += 1

    recency_spike = bool(top) and (recent_count / len(top)) > RECENCY_SPIKE_RATIO

    staleness_caveat: str | None = None
    if dates_in_top and not recency_spike and not has_explicit_date_filter:
        oldest = min(dates_in_top)
        if (today - oldest).days > staleness_days:
            staleness_caveat = (
                "Note: the most relevant reviews found are from over a year ago. "
                "This information may be outdated -- you may want to look for more recent feedback."
            )

    evidence = [
        EvidenceItem(
            snippet=result.payload.get("text", ""),
            username=result.payload.get("username"),
            rating=result.payload.get("rating"),
            effective_rating=_effective_rating(result.payload),
            source=result.payload.get("source"),
            sentiment=result.payload.get("sentiment_label"),
            sentiment_conflict=not result.payload.get("sentiment_rating_agree", True),
            date_inferred=result.payload.get("date_inferred", False),
            relevance=result.score,
        )
        for _, result in top
    ]

    return RankingResult(
        evidence=evidence,
        entity_counts=dict(entity_counter.most_common(20)),
        source_breakdown=dict(source_counter),
        recency_spike=recency_spike,
        staleness_caveat=staleness_caveat,
        low_evidence=low_evidence,
    )


def _parse_date(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value)
            return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt
        except ValueError:
            return None
    return None
