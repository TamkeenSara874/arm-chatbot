from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# Canonical casing exactly as stored in each review's Qdrant payload "source"
# field -- Qdrant's MatchAny filter is case-sensitive, so a decomposition
# output like "Opentable" (matching how a user typed it, not how it's stored)
# would silently match zero real reviews while still matching others in the
# same list, e.g. "Google" -- confirmed live as the cause of a source filter
# that appeared to work but returned zero evidence for the misspelled source.
KNOWN_REVIEW_SOURCES = ["Google", "Yelp", "Tripadvisor", "OpenTable", "Nugget"]
_SOURCE_BY_LOWER = {s.lower(): s for s in KNOWN_REVIEW_SOURCES}


class EvidenceItem(BaseModel):
    snippet: str
    username: str | None = None
    rating: float | None = None
    # Sentiment-mapped rating used when sentiment_conflict=True (e.g. a 5-star
    # review with a text complaint scores as if it were ~1.5). Equal to
    # `rating` when the star rating and text sentiment agree.
    effective_rating: float | None = None
    source: str | None = None
    sentiment: str | None = None
    sentiment_conflict: bool = False
    date_inferred: bool = False
    # Per-review date, exposed per evidence item -- not just the aggregate
    # staleness_caveat, which only fires when the MAJORITY of evidence is
    # old. A single old review can still be the one basis for naming a
    # specific person (staff or reviewer) once REVIEWER/STAFF NAME PRIVACY
    # allows real names through on request -- without the individual date,
    # the owner has no way to tell "customers love Sam" apart from "one
    # customer loved Sam, two years ago" before acting on it (e.g. firing
    # someone over a single stale complaint).
    review_date: str | None = None
    relevance: float = 0.0
    # True when `relevance` is a real cross-encoder sigmoid score (a
    # calibrated 0-1 probability, safe to label with fixed thresholds like
    # "Strong match"). False when reranking failed or was skipped as
    # degenerate (couldn't meaningfully tell candidates apart), in which case
    # `relevance` is the retrieval step's own fusion score instead -- a
    # different scale, not comparable to a real reranked score, and not safe
    # to label with the same fixed thresholds.
    relevance_calibrated: bool = True
    # The single sentence within `snippet` most relevant to the query, so the
    # frontend can highlight it -- lets a user spot the relevant part of a
    # long, multi-paragraph review without reading the whole thing. None when
    # snippet is already one sentence (nothing to distinguish) or when no
    # query vector was available to score against (see rank_results).
    highlight: str | None = None


class SubAnswer(BaseModel):
    sub_query: str
    answer: str


class ChatResponseSchema(BaseModel):
    """Structured output schema enforced at the API level via OpenAI beta.parse."""

    answer: str
    sub_answers: list[SubAnswer] = []
    evidence: list[EvidenceItem]
    confidence: float = Field(..., ge=0.0, le=1.0)
    caveats: str | None = None
    entity_counts: dict[str, int] = {}
    source_breakdown: dict[str, int] = {}


class DateFilter(BaseModel):
    from_date: str | None = Field(None, alias="from")
    to_date: str | None = Field(None, alias="to")

    model_config = {"populate_by_name": True}


class RatingFilter(BaseModel):
    min: float | None = None
    max: float | None = None


QueryIntent = Literal[
    "best_item",
    "worst_item",
    "sentiment_overview",
    "specific_aspect",
    "comparison",
    "aggregation",
    "count_query",
    "report",
    "improvement",
    "factual",
    "conversation_recall",
    "out_of_scope",
    "ui_question",
    "report_howto",
    "manipulation_request",
    "multi_location",
    "allergen",
]


class DecomposedQuery(BaseModel):
    # Constrained to the exact intent set the decomposition prompt documents so
    # a malformed/hallucinated intent fails Pydantic validation (triggering
    # decompose_query()'s retry-then-safe-fallback) instead of silently
    # skipping the guardrail check, which only matches against known intents.
    intent: QueryIntent
    aspect_filter: str | None = None
    sentiment_filter: str | None = None
    entities: list[str] = []
    needs_aggregation: bool = False
    complexity: Literal["simple", "complex"] = "simple"
    sub_queries: list[str] = []
    rephrased_query: str = ""
    # A comparison question can name more than one platform ("why is my Yelp
    # rating lower than Google?") -- a single-value filter couldn't express
    # "just these two, not Tripadvisor/OpenTable too", which is exactly why
    # this was silently unused everywhere before: retrieval had no way to
    # apply it. list[str] lets a query restrict evidence to any number of
    # named sources; empty means no restriction.
    source_filter: list[str] = []
    date_filter: DateFilter | None = None
    rating_filter: RatingFilter | None = None
    # Second period for trend-comparison questions ("since last month",
    # "compared to last week") -- date_filter holds the current/more-recent
    # period, this holds the earlier one being compared against. None for
    # every non-comparison query.
    compare_date_filter: DateFilter | None = None
    # Set when the question asks for an exact distribution/ranking across the
    # WHOLE restaurant by one of these dimensions ("which source has the most
    # reviews", "breakdown by rating") -- triggers a direct SQL GROUP BY
    # instead of leaving the model to tally a breakdown from only the
    # retrieved evidence sample and state it as if it were the real total.
    breakdown_dimension: Literal["source", "rating", "sentiment"] | None = None
    # Set when the question asks about reviews WITH written text/comments vs.
    # rating-only reviews with none ("how many reviews have a rating but no
    # written text", "star-rating-only reviews"). Maps directly to
    # ReviewChunkMeta.has_content -- without this, _compute_count() had no way
    # to filter by this condition and would silently fall back to an
    # unfiltered total count whenever a count_query landed on this kind of
    # question, presenting the wrong number as an exact fact.
    content_filter: Literal["has_text", "no_text"] | None = None
    # Set when the question asks about the restaurant's overall rating,
    # overall sentiment, or what percentage/proportion of reviews are
    # positive/negative/neutral/mixed -- triggers a direct SQL average-rating
    # + sentiment-count computation instead of estimating either figure from
    # only the top_k retrieved evidence sample. Deliberately independent of
    # `intent` (like breakdown_dimension/compare_date_filter/content_filter
    # above, and unlike the intent-gated count_query path): a compound
    # question ("what % of my reviews are negative and how worried should I
    # be") must not lose this signal just because the reasoning tail pulls
    # intent classification toward aggregation/reasoning instead of
    # sentiment_overview. Set this whenever the condition applies, regardless
    # of whatever intent/sub_queries also get set for the rest of the question.
    wants_overall_stats: bool = False
    # Set when the question asks how many/what share of reviews mention a
    # QUALITATIVE THEME that has no dedicated database column ("how many
    # people called my staff rude", "how many reviews complain about cold
    # food") -- a short list (2-5) of lowercase keyword/phrase variants
    # covering the theme (e.g. ["rude", "unfriendly", "hostile"]), used to
    # count matching reviews via full_review ILIKE across the WHOLE
    # restaurant, not just the top_k retrieved sample. Unlike count_query
    # (a real stored column, exact), this is a keyword-match approximation --
    # it can undercount differently-worded mentions but scans every review,
    # not just 20. Independent of intent, like wants_overall_stats above:
    # still set this even if the question also asks something else
    # (put that in sub_queries). None when the question isn't asking to
    # count a qualitative theme.
    theme_keywords: list[str] | None = None
    # Set alongside theme_keywords for a question comparing TWO qualitative
    # themes ("which has more complaints: food quality or staff behavior, and
    # by how much?") -- mirrors compare_date_filter sitting alongside
    # date_filter for the same reason: theme_keywords holds the first theme's
    # keywords (e.g. ["rude", "unfriendly", "inattentive"] for staff), this
    # holds the second (e.g. ["bland", "cold food", "undercooked"] for food
    # quality), so both get an exact full-corpus count instead of the model
    # eyeballing which appears more often in only the top_k retrieved sample.
    # None for a single-theme count question.
    compare_theme_keywords: list[str] | None = None
    # Set alongside theme_keywords and compare_theme_keywords when the
    # question asks how many reviews mention BOTH themes TOGETHER in the same
    # review ("which reviews mention both slow service and cold food?", "how
    # many complain about slow service AND cold food at once?") -- as opposed
    # to comparing which theme is more common overall (the default meaning of
    # setting both keyword lists without this flag). Confirmed live as a real
    # bug: with no way to express "both together," decomposition folded both
    # themes into one flat OR-matched list, so the exact count reported "any
    # review mentioning slow service OR cold food" while being worded as if it
    # meant "both at once" -- directly contradicting the model's own read of
    # the actual retrieved evidence, which found no review mentioning both.
    # False (the default) keeps the existing side-by-side comparison meaning.
    theme_require_both: bool = False

    # Groq's JSON-mode output (llama-3.3-70b, temperature=0.0) was confirmed
    # live to occasionally emit `null` for a field whose schema default is ""
    # or [] rather than actually omitting the key -- e.g. "rephrased_query":
    # null -- which fails Pydantic validation on a plain `str`/`list[str]`
    # field. Since temperature=0.0 makes this deterministic for a given
    # prompt, retrying the identical request against a different Groq API
    # key (RotatingGroqClient's rotation-on-failure) reproduced the exact
    # same failure on every single key, wastefully cycling through all of
    # them (each taking several seconds) before decompose_query()'s own
    # corrective retry ever got a chance to run -- confirmed live as a
    # 30-50+ second latency spike. Coercing these three fields' `None` to
    # their normal empty default here means that particular model quirk
    # never reaches Pydantic as a validation failure at all.
    @field_validator("rephrased_query", mode="before")
    @classmethod
    def _none_to_empty_string(cls, v: str | None) -> str:
        return v or ""

    @field_validator("entities", "sub_queries", mode="before")
    @classmethod
    def _none_to_empty_list(cls, v: list[str] | None) -> list[str]:
        return v or []

    @field_validator("source_filter", mode="before")
    @classmethod
    def _normalize_source_filter(cls, v: list[str] | str | None) -> list[str]:
        # Tolerate the model returning a bare string (e.g. "Google") instead
        # of the array the prompt asks for, same defensive spirit as the
        # None-coercion above -- a schema mismatch here shouldn't fail
        # decomposition over a field that's supplementary to begin with.
        if v is None:
            raw = []
        elif isinstance(v, str):
            raw = [v] if v else []
        else:
            raw = v

        # Case-normalize to the canonical stored casing (e.g. "Opentable" or
        # "OPENTABLE" -> "OpenTable") -- Qdrant's filter match is case-exact,
        # so anything else here would silently match zero real reviews.
        # Unrecognized values (a name the model invented) are dropped rather
        # than passed through, since they could never match anything either.
        return [_SOURCE_BY_LOWER[s.lower()] for s in raw if s.lower() in _SOURCE_BY_LOWER]


# Request / response schemas for the API


class SessionCreateRequest(BaseModel):
    restaurant_id: int
    user_identifier: str | None = None


class SessionResponse(BaseModel):
    session_id: uuid.UUID
    restaurant_id: int


class AnomalyAlertResponse(BaseModel):
    detected: bool
    message: str | None = None
    recent_avg_rating: float | None = None
    baseline_avg_rating: float | None = None
    recent_negative_share: float | None = None
    baseline_negative_share: float | None = None
    recent_count: int
    baseline_count: int


class MessageResponse(BaseModel):
    message_id: uuid.UUID
    role: str
    content: str
    confidence: float | None = None
    created_at: str


class ChatQueryRequest(BaseModel):
    session_id: uuid.UUID
    restaurant_id: int
    message: str = Field(..., min_length=1, max_length=2000)
    # Set by the frontend's "Regenerate" action -- skips both cache lookups
    # (exact-text and semantic) so a regenerate actually re-runs the pipeline
    # instead of serving back the same cached answer it's trying to replace.
    # Does not affect writing the fresh response to cache afterward.
    bypass_cache: bool = False


class ChatQueryResponse(BaseModel):
    session_id: uuid.UUID
    message_id: uuid.UUID
    response: ChatResponseSchema
    cached: bool = False
    complexity: str
    model_used: str
    latency_ms: int = 0
    cost_usd: float = 0.0


class CorrectionRequest(BaseModel):
    session_id: uuid.UUID
    message_id: uuid.UUID
    corrected_response: str = Field(..., min_length=1, max_length=4000)


class CorrectionResponse(BaseModel):
    correction_id: uuid.UUID
    is_consensus: bool


class FeedbackRequest(BaseModel):
    message_id: uuid.UUID


class FeedbackResponse(BaseModel):
    ok: bool


class IngestJobResponse(BaseModel):
    job_id: uuid.UUID
    status: str
    progress_pct: int
    total_reviews: int | None = None
    total_chunks: int | None = None
    skipped_empty: int | None = None
    skipped_already_processed: int | None = None
    error_message: str | None = None


class RestaurantListResponse(BaseModel):
    restaurant_ids: list[int]


class ReviewIngestRequest(BaseModel):
    """One review pushed live by a source system, e.g. the moment it's posted
    or edited -- the incremental counterpart to the batch /ingest file upload.

    external_review_id must be a stable identifier from the source system
    (its own review ID) so a repeat call for the same review is a genuine
    update rather than a duplicate: review_id is derived deterministically
    from (restaurant_id, external_review_id).
    """

    restaurant_id: int
    external_review_id: str = Field(..., min_length=1, max_length=255)
    review: str = Field(..., max_length=10000)
    rating: float | None = Field(None, ge=1, le=5)
    username: str | None = Field(None, max_length=255)
    source: str | None = Field(None, max_length=100)
    created_at: str | None = None
    sentiment: str | None = None


class ReviewIngestResponse(BaseModel):
    review_id: str
    status: str
    chunks_written: int


class ReportRequest(BaseModel):
    session_id: uuid.UUID
    restaurant_id: int
    message: str = Field(..., min_length=1, max_length=2000)
    date_from: date | None = None
    date_to: date | None = None


class InsightsReport(BaseModel):
    """Structured output from the export_insights_report tool call."""

    restaurant_id: int
    generated_at: datetime
    date_from: date | None
    date_to: date | None
    total_reviews: int
    avg_rating: float | None
    rating_distribution: dict[str, int]
    sentiment_breakdown: dict[str, int]
    source_breakdown: dict[str, int]
    top_praised: list[tuple[str, int]]
    top_complained: list[tuple[str, int]]
    summary: str
    markdown: str


class ReportResponse(BaseModel):
    restaurant_id: int
    report: InsightsReport
    model_used: str
