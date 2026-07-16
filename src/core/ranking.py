from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime

from src.config import Settings
from src.core.chunking import _sentence_split
from src.core.reranker import score_for_highlight
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


async def rank_results(
    fused_results: list[SearchResult],
    settings: Settings,
    today: datetime | None = None,
    top_k: int = 6,
    staleness_days: int | None = None,
    has_explicit_date_filter: bool = False,
    query: str | None = None,
    reranker_model: str | None = None,
    reranked: bool = True,
) -> RankingResult:
    """Apply composite scoring and return ranked evidence with diagnostics.

    Each result's .score field must be the reranker's sigmoid relevance score
    (src/core/reranker.py) -- the real signal for how well a chunk matches the
    query. The composite score formula is:
        injection_penalty * (w_rrf*relevance + w_recency*(1/(days+1)) + w_rating*(effective_rating/5))
    where effective_rating is sentiment-mapped when text sentiment and star rating disagree.
    The w_rrf weight name is historical (it used to weight an RRF score); it
    now weights the reranker's relevance score.

    When query and reranker_model are both given, each evidence item's
    highlight is set to whichever of its sentences the cross-encoder scores
    highest against query (see _add_highlights). Omit either to skip
    highlighting -- e.g. when no reranker_model is configured.
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
        relevance_score = result.score

        effective_rating = _effective_rating(p)
        rating_contribution = effective_rating / 5.0

        review_date = _parse_date(p.get("review_date"))
        days_old = max(0, (today - review_date).days) if review_date else staleness_days

        recency_score = 1.0 / (days_old + 1)
        injection_penalty = 0.5 if p.get("has_injection_attempt", False) else 1.0

        composite = injection_penalty * (
            w_rrf * relevance_score + w_recency * recency_score + w_rating * rating_contribution
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
        # Majority-stale, not single-oldest-stale: a corpus spanning years will
        # almost always contain at least one review older than staleness_days
        # among any reasonably-sized evidence set (e.g. 18 recent + 2 from
        # 2019), and checking only min(dates_in_top) fired this caveat on
        # nearly every aggregate query regardless of how recent most of the
        # evidence actually was. Only warn when most of what's shown is old.
        stale_count = sum(1 for d in dates_in_top if (today - d).days > staleness_days)
        if stale_count > len(dates_in_top) / 2:
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
            review_date=_format_date(_parse_date(result.payload.get("review_date"))),
            relevance=result.score,
            relevance_calibrated=reranked,
        )
        for _, result in top
    ]

    if query is not None and reranker_model is not None:
        await _add_highlights(evidence, query, reranker_model)

    return RankingResult(
        evidence=evidence,
        entity_counts=dict(entity_counter.most_common(20)),
        source_breakdown=dict(source_counter),
        recency_spike=recency_spike,
        staleness_caveat=staleness_caveat,
        low_evidence=low_evidence,
    )


async def _add_highlights(
    evidence: list[EvidenceItem],
    query: str,
    reranker_model: str,
) -> None:
    """Set .highlight on each item to its highest cross-encoder-scored sentence.

    Splits each snippet into sentences (same splitter chunking.py uses) and
    scores all of them against query in one batched call via the already-warm
    local reranker model -- no network I/O, unlike the OpenAI-embedding
    approach this replaced (measured at 2-7s/request on real evidence sets).
    """
    per_item_sentences: list[list[str]] = []
    all_sentences: list[str] = []
    for item in evidence:
        sentences = _split_into_highlight_candidates(item.snippet)
        per_item_sentences.append(sentences)
        if len(sentences) > 1:
            all_sentences.extend(sentences)

    if not all_sentences:
        return

    scores = await score_for_highlight(query, all_sentences, reranker_model)
    if not scores:
        return
    scores_by_sentence = dict(zip(all_sentences, scores, strict=True))

    for item, sentences in zip(evidence, per_item_sentences, strict=True):
        if len(sentences) <= 1:
            continue
        item.highlight = max(sentences, key=lambda s: scores_by_sentence[s])


_ELLIPSIS_RE = re.compile(r"(?<=\.\.\.)\s+|(?<=…)\s+")


def _split_into_highlight_candidates(text: str) -> list[str]:
    """Sentence-split for highlight scoring, then repair the two failure
    modes confirmed live on real, informal review text:

    - An ellipsis ("...") inside a real NLTK sentence boundary doesn't
      reliably end it, so several genuinely separate thoughts (e.g. "...we
      spoke to the hostess... she said... (true story)") get glued into one
      oversized "sentence" -- split further on internal ellipses.
    - A run-on/comma-spliced sentence can get cut in the wrong place by NLTK
      itself, leaving an orphaned fragment that starts mid-thought (a real
      sentence should start capitalized) -- merge it back into the previous
      piece rather than let it be selected and shown incomplete. This check
      only applies to NLTK's own sentence boundaries, not ours: text
      continuing after "..." is normally lowercase as a stylistic
      trail-off, not a broken cut, so applying the same check there would
      immediately undo the ellipsis split above -- confirmed live as a real
      bug in an earlier version of this function. A short-but-complete
      sentence ("Very well.", "Service was excellent.") is common in review
      text and deliberately left alone -- length alone isn't a fragment
      signal, only an unexpected lowercase start is.
    """
    pieces: list[tuple[str, bool]] = []
    for sentence in _sentence_split(text):
        sub_pieces = [p for p in _ELLIPSIS_RE.split(sentence) if p.strip()]
        pieces.extend((p, i > 0) for i, p in enumerate(sub_pieces))

    if not pieces:
        return []

    merged = [pieces[0][0]]
    for piece, from_ellipsis_split in pieces[1:]:
        stripped = piece.strip()
        broken_nltk_cut = (
            not from_ellipsis_split and stripped and stripped[0].islower()
        )
        if broken_nltk_cut:
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    return merged


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


def _format_date(dt: datetime | None) -> str | None:
    """Plain YYYY-MM-DD for EvidenceItem.review_date, or None if unparseable."""
    return dt.strftime("%Y-%m-%d") if dt else None
