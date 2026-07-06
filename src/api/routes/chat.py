"""Chat API routes: sessions, query (SSE), history, corrections, and report."""

# NOTE: deliberately no `from __future__ import annotations` here. Combined
# with slowapi's @limiter.limit() decorator, postponed evaluation breaks
# FastAPI's dependant analysis for every decorated route in this file --
# it can no longer resolve `body: ChatQueryRequest` / Depends() types, so it
# silently treats them as required query parameters instead of a JSON body /
# dependency injection, and every such route 422s on any real request. See
# the identical fix + repro notes in src/api/routes/ingest.py.

import contextlib
import json
import time
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from openai import AsyncOpenAI
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from src.api.dependencies import (
    DbSession,
    RestaurantId,
    get_cache,
    get_complex_client,
    get_decomp_client,
    get_embedder,
    get_openai_client,
    get_simple_client,
    get_summary_client,
    get_vector_store,
)
from src.api.rate_limit import limiter
from src.config import get_settings
from src.core.anomaly import get_anomaly_status
from src.core.correction import find_correction, store_correction
from src.core.decomposition import decompose_query
from src.core.generation import (
    build_generation_prompt,
    build_structured_response,
    check_hallucination_gate,
    clean_answer_text,
    format_evidence,
    select_generation,
)
from src.core.groundedness import check_count_groundedness
from src.core.guardrail import check_guardrail
from src.core.ranking import rank_results
from src.core.report import generate_report
from src.core.retrieval import RetrievalTiming, build_retrieval_params, hybrid_retrieve
from src.core.review_stats import PeriodStats, compute_period_stats
from src.core.semantic_cache import (
    find_cached_response,
    invalidate_cached_response,
    store_cached_response,
)
from src.core.session import (
    build_recent_turns_context,
    build_session_context,
    maybe_trigger_summary,
    store_session_turn,
)
from src.models.db_entities import ChatMessage, ChatSession, ReviewChunkMeta
from src.models.schemas import (
    AnomalyAlertResponse,
    ChatQueryRequest,
    ChatResponseSchema,
    CorrectionRequest,
    CorrectionResponse,
    EvidenceItem,
    FeedbackRequest,
    FeedbackResponse,
    MessageResponse,
    ReportRequest,
    ReportResponse,
    SessionCreateRequest,
    SessionResponse,
    SubAnswer,
)
from src.services.cache import RedisCache
from src.services.embedding.base import BaseEmbedder
from src.services.llm.base import BaseLLMClient
from src.services.prompt_service import get_prompt_loader
from src.services.vector.base import BaseVectorStore
from src.utils.background import fire_and_forget
from src.utils.metrics import active_sessions_gauge
from src.utils.security import sanitize_input, validate_llm_output
from src.utils.tracing import RequestTrace

logger = structlog.get_logger()

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

settings = get_settings()

DecompClient = Annotated[BaseLLMClient, Depends(get_decomp_client)]
SimpleClient = Annotated[BaseLLMClient, Depends(get_simple_client)]
ComplexClient = Annotated[BaseLLMClient, Depends(get_complex_client)]
SummaryClient = Annotated[BaseLLMClient, Depends(get_summary_client)]
Embedder = Annotated[BaseEmbedder, Depends(get_embedder)]
VectorStore = Annotated[BaseVectorStore, Depends(get_vector_store)]
Cache = Annotated[RedisCache, Depends(get_cache)]
OpenAI = Annotated[AsyncOpenAI, Depends(get_openai_client)]


@router.post("/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    body: SessionCreateRequest,
    restaurant_id: RestaurantId,
    db: DbSession,
) -> SessionResponse:
    session = ChatSession(
        restaurant_id=restaurant_id,
        user_identifier=body.user_identifier,
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    active_sessions_gauge.inc()
    logger.info("session_created", session_id=str(session.id), restaurant_id=restaurant_id)
    return SessionResponse(session_id=session.id, restaurant_id=session.restaurant_id)


@router.get("/alerts", response_model=AnomalyAlertResponse)
@limiter.limit(settings.rate_limit_read)
async def get_alerts(
    request: Request,
    restaurant_id: RestaurantId,
    db: DbSession,
    cache: Cache,
) -> AnomalyAlertResponse:
    result = await get_anomaly_status(db, cache, restaurant_id)
    return AnomalyAlertResponse(
        detected=result.detected,
        message=result.message,
        recent_avg_rating=result.recent_avg_rating,
        baseline_avg_rating=result.baseline_avg_rating,
        recent_negative_share=result.recent_negative_share,
        baseline_negative_share=result.baseline_negative_share,
        recent_count=result.recent_count,
        baseline_count=result.baseline_count,
    )


@router.post("/query")
@limiter.limit(settings.rate_limit_chat)
async def chat_query(
    request: Request,
    body: ChatQueryRequest,
    restaurant_id: RestaurantId,
    db: DbSession,
    decomp_client: DecompClient,
    simple_client: SimpleClient,
    complex_client: ComplexClient,
    summary_client: SummaryClient,
    embedder: Embedder,
    vector_store: VectorStore,
    cache: Cache,
) -> EventSourceResponse:
    trace = RequestTrace(
        session_id=str(body.session_id),
        restaurant_id=restaurant_id,
    )

    sanitized = sanitize_input(body.message)

    # Cache check before any LLM work
    cached_data = await cache.get(restaurant_id, sanitized)
    if cached_data:
        trace.cache_hit = True
        trace.emit()
        return EventSourceResponse(_yield_cached(cached_data, body.session_id))

    # Query decomposition. Embedding is deliberately NOT computed here in
    # parallel: retrieval must use decomposed.rephrased_query (pronoun
    # resolution, vague-query expansion), which doesn't exist yet at this
    # point, and guardrailed/count_query intents never need an embedding at
    # all -- computing one here would silently waste an API call on every
    # out-of-scope question. hybrid_retrieve() embeds the right query text
    # itself once decomposition has run.
    recent_turns = await build_recent_turns_context(body.session_id, db)
    loader = get_prompt_loader()
    decomp_system, decomp_user = loader.format(
        "query_decomposition",
        query=sanitized,
        session_context=recent_turns,
        current_date=datetime.now(UTC).date().isoformat(),
    )
    t_decomp = time.perf_counter()
    decomposed = await decompose_query(
        decomp_client,
        decomp_user,
        decomp_system,
        usage_callback=lambda p, c: trace.record_tokens(settings.groq_decomp_model, p, c),
    )
    trace.decomp_ms = (time.perf_counter() - t_decomp) * 1000.0
    trace.intent = decomposed.intent
    trace.complexity = decomposed.complexity
    trace.decomp_model = settings.groq_decomp_model

    # Guardrail — no retrieval or generation for out-of-scope intents
    guardrail_text = check_guardrail(decomposed.intent)
    if guardrail_text:
        trace.emit()
        response = ChatResponseSchema(
            answer=guardrail_text,
            evidence=[],
            confidence=1.0,
        )
        return EventSourceResponse(
            _yield_instant(response, body.session_id, uuid.uuid4(), model_used="guardrail")
        )

    # conversation_recall fast path — this question is about the conversation
    # itself ("what did I ask before?"), not the restaurant's reviews, so it
    # must never touch review retrieval/ranking. Retrieving reviews here
    # would attach fake evidence, a confidence score, and a staleness caveat
    # to an answer that has nothing to do with review content -- exactly the
    # bug this path exists to avoid.
    if decomposed.intent == "conversation_recall":
        session_context = await build_session_context(
            body.session_id, restaurant_id, sanitized, db, vector_store, embedder
        )
        recall_system, recall_user = loader.format(
            "conversation_recall", query=sanitized, session_context=session_context
        )
        t_recall = time.perf_counter()
        recall_answer = await simple_client.complete(
            recall_user,
            recall_system,
            usage_callback=lambda p, c: trace.record_tokens(settings.openai_simple_model, p, c),
        )
        trace.generation_ms = (time.perf_counter() - t_recall) * 1000.0
        trace.generation_model = settings.openai_simple_model
        trace.emit()
        return EventSourceResponse(
            _yield_instant(
                ChatResponseSchema(answer=recall_answer.strip(), evidence=[], confidence=1.0),
                body.session_id,
                uuid.uuid4(),
                model_used=settings.openai_simple_model,
            )
        )

    # count_query fast path — direct Postgres COUNT(*), no LLM generation.
    # A compound question ("how many positive reviews AND how can I
    # improve?") comes back from decomposition as intent=count_query with a
    # non-empty sub_queries list for the other half. In that case the pure
    # fast path would silently drop the second half, so fall through to the
    # full pipeline instead -- but pass the exact DB-computed count along so
    # the LLM states it verbatim rather than trying to (mis)count evidence
    # chunks itself.
    precomputed_count: str | None = None
    if decomposed.intent == "count_query":
        if not decomposed.sub_queries:
            count_answer, count_msg_id = await _handle_count_query(
                db, body, restaurant_id, decomposed, sanitized, trace
            )
            return EventSourceResponse(
                _yield_instant(
                    ChatResponseSchema(answer=count_answer, evidence=[], confidence=1.0),
                    body.session_id,
                    count_msg_id,
                    model_used="direct_query",
                )
            )
        sentiment_filter = _resolve_sentiment_filter(sanitized, decomposed.sentiment_filter)
        count = await _compute_count(db, restaurant_id, decomposed, sentiment_filter)
        precomputed_count = _format_count_answer(count, sentiment_filter)

    # Trend-comparison fast path — decomposition sets compare_date_filter for
    # a time-period comparison question ("since last month", "compared to
    # last week"). Both periods' exact stats are computed via direct SQL
    # here, same rationale as count_query above, and threaded into the
    # complex generation prompt below rather than left for the LLM to
    # estimate from a sample of retrieved evidence.
    precomputed_trend: str | None = None
    if decomposed.compare_date_filter is not None:
        precomputed_trend = await _compute_trend_comparison(db, restaurant_id, decomposed)

    # Semantic cache check. Uses the decomposed/rephrased query (context- and
    # pronoun-resolved, more canonical than the raw message) so a paraphrase
    # of an earlier question can still hit -- exact-text caching alone would
    # miss "what about the pasta?" vs. "how was the pasta?" even though
    # they resolve to the same retrieval query. This only guards the
    # expensive retrieval+generation path below, not the guardrail/count_query
    # fast paths above, which are already cheap.
    retrieval_query = decomposed.rephrased_query.strip() or sanitized
    cached_semantic, cache_query_vector = await find_cached_response(
        retrieval_query,
        restaurant_id,
        embedder,
        vector_store,
        cache,
        settings.qdrant_collection_chat_cache,
        threshold=settings.semantic_cache_similarity_threshold,
    )
    if cached_semantic:
        trace.cache_hit = True
        trace.emit()
        return EventSourceResponse(_yield_cached(cached_semantic, body.session_id))

    # Full pipeline inside the SSE generator
    return EventSourceResponse(
        _pipeline_stream(
            body=body,
            restaurant_id=restaurant_id,
            sanitized=sanitized,
            decomposed=decomposed,
            retrieval_query=retrieval_query,
            precomputed_query_vector=cache_query_vector or None,
            precomputed_count=precomputed_count,
            precomputed_trend=precomputed_trend,
            db=db,
            simple_client=simple_client,
            complex_client=complex_client,
            summary_client=summary_client,
            embedder=embedder,
            vector_store=vector_store,
            cache=cache,
            trace=trace,
        )
    )


@router.post("/report", response_model=ReportResponse)
@limiter.limit(settings.rate_limit_chat)
async def chat_report(
    request: Request,
    body: ReportRequest,
    restaurant_id: RestaurantId,
    db: DbSession,
    vector_store: VectorStore,
    openai_client: OpenAI,
) -> ReportResponse:
    settings_ = get_settings()
    report = await generate_report(
        user_message=body.message,
        restaurant_id=restaurant_id,
        db_session=db,
        vector_store=vector_store,
        qdrant_reviews_collection=settings_.qdrant_collection_reviews,
        openai_client=openai_client,
        model=settings_.openai_simple_model,
        date_from=body.date_from,
        date_to=body.date_to,
    )
    logger.info(
        "report_generated",
        restaurant_id=restaurant_id,
        session_id=str(body.session_id),
    )
    return ReportResponse(
        restaurant_id=restaurant_id, report=report, model_used=settings_.openai_simple_model
    )


@router.get("/sessions/{session_id}/history", response_model=list[MessageResponse])
@limiter.limit(settings.rate_limit_read)
async def get_session_history(
    request: Request,
    session_id: uuid.UUID,
    restaurant_id: RestaurantId,
    db: DbSession,
    limit: int = 20,
    offset: int = 0,
) -> list[MessageResponse]:
    session = await db.get(ChatSession, session_id)
    if session is None or session.restaurant_id != restaurant_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    stmt = (
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
        .limit(limit)
        .offset(offset)
    )
    result = await db.execute(stmt)
    messages = result.scalars().all()
    return [
        MessageResponse(
            message_id=m.id,
            role=m.role,
            content=m.content,
            confidence=m.confidence,
            created_at=m.created_at.isoformat(),
        )
        for m in messages
    ]


@router.post("/{message_id}/feedback", response_model=FeedbackResponse)
@limiter.limit(settings.rate_limit_correct)
async def submit_feedback(
    request: Request,
    message_id: uuid.UUID,
    body: FeedbackRequest,
    restaurant_id: RestaurantId,
    db: DbSession,
) -> FeedbackResponse:
    """Record a thumbs-up on an assistant message. Never overrides evidence --
    purely a positive-feedback signal, unlike /correct which stores a real
    corrected answer.

    message_id here is the *user* message id (same convention /correct uses,
    and the same id the frontend has via response.message_id) -- the
    assistant's own row gets a separate, never-returned id, so the paired
    assistant response is located the same way /correct locates it.
    """
    if message_id != body.message_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="message_id mismatch")

    user_msg = await db.get(ChatMessage, message_id)
    if user_msg is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Message not found")

    session = await db.get(ChatSession, user_msg.session_id)
    if session is None or session.restaurant_id != restaurant_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Message not found")

    stmt = (
        select(ChatMessage)
        .where(ChatMessage.session_id == user_msg.session_id)
        .where(ChatMessage.role == "assistant")
        .where(ChatMessage.created_at >= user_msg.created_at)
        .order_by(ChatMessage.created_at.asc())
        .limit(1)
    )
    result = await db.execute(stmt)
    assistant_msg = result.scalar_one_or_none()
    if assistant_msg is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Message not found")

    assistant_msg.feedback = "up"
    await db.commit()
    return FeedbackResponse(ok=True)


@router.post("/correct", response_model=CorrectionResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit(settings.rate_limit_correct)
async def submit_correction(
    request: Request,
    body: CorrectionRequest,
    restaurant_id: RestaurantId,
    db: DbSession,
    embedder: Embedder,
    vector_store: VectorStore,
    cache: Cache,
    decomp_client: DecompClient,
) -> CorrectionResponse:
    # Retrieve the original message to get the query text and restaurant_id
    user_msg = await db.get(ChatMessage, body.message_id)
    if user_msg is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Message not found")

    session = await db.get(ChatSession, user_msg.session_id)
    if session is None or session.restaurant_id != restaurant_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    # Find the assistant response that follows this specific user message --
    # not just the first assistant message ever sent in the session. The
    # user/assistant pair from one turn are committed in the same transaction
    # in _post_response_tasks(), so Postgres's func.now() gives them the exact
    # same created_at; ">=" + ascending + limit 1 finds that pair rather than
    # an earlier turn's response.
    stmt = (
        select(ChatMessage)
        .where(ChatMessage.session_id == user_msg.session_id)
        .where(ChatMessage.role == "assistant")
        .where(ChatMessage.created_at >= user_msg.created_at)
        .order_by(ChatMessage.created_at.asc())
        .limit(1)
    )
    result = await db.execute(stmt)
    assistant_msg = result.scalar_one_or_none()
    original_response = assistant_msg.content if assistant_msg else ""

    # Classify the original query the same way the live pipeline would, so
    # the stored intent actually matches what find_correction() compares
    # against on a future identical/similar query. A hardcoded "factual" here
    # made the correction nearly unusable: real queries classify into many
    # intents (specific_aspect, sentiment_overview, best_item, ...), so
    # find_correction()'s intent cross-check silently rejected almost every
    # correction -- confirmed live via the eval harness (correction stored
    # and embedded correctly, but never surfaced on re-ask).
    recent_turns = await build_recent_turns_context(user_msg.session_id, db)
    loader = get_prompt_loader()
    decomp_system, decomp_user = loader.format(
        "query_decomposition", query=user_msg.content, session_context=recent_turns
    )
    decomposed = await decompose_query(decomp_client, decomp_user, decomp_system)

    correction_id, is_consensus = await store_correction(
        session_id=body.session_id,
        restaurant_id=session.restaurant_id,
        original_query=user_msg.content,
        original_response=original_response,
        corrected_response=body.corrected_response,
        intent=decomposed.intent,
        embedder=embedder,
        vector_store=vector_store,
        db_session=db,
        sim_threshold=get_settings().correction_sim_threshold,
    )

    # Bust the cache entry for this exact query text -- otherwise a repeat of
    # the same question would keep serving the pre-correction cached answer
    # for the rest of the TTL, silently ignoring the correction just made.
    await cache.invalidate_query(session.restaurant_id, user_msg.content)
    # Also bust the semantic tier: it's keyed on decomposed.rephrased_query,
    # not the raw text above, and separately maintains a Qdrant index point --
    # confirmed live that skipping this left a stale semantic hit still
    # serving the pre-correction answer even after the raw-text key was gone.
    retrieval_query = decomposed.rephrased_query.strip() or user_msg.content
    await invalidate_cached_response(
        retrieval_query,
        session.restaurant_id,
        vector_store,
        cache,
        get_settings().qdrant_collection_chat_cache,
    )

    logger.info(
        "correction_stored",
        correction_id=str(correction_id),
        is_consensus=is_consensus,
        session_id=str(body.session_id),
    )
    return CorrectionResponse(correction_id=correction_id, is_consensus=is_consensus)


_SENTIMENT_KEYWORDS: dict[str, str] = {
    "positive": "Positive",
    "negative": "Negative",
    "mixed": "Mixed",
    "neutral": "Neutral",
}


def _resolve_sentiment_filter(raw_query: str, decomposed_filter: str | None) -> str | None:
    """Deterministic keyword override for decomposition's sentiment_filter.

    Observed live: decomposition (a small, fast free-tier model) can extract
    the wrong polarity on an otherwise-trivial query -- e.g. "total negative
    reviews" coming back with sentiment_filter="Positive". A count_query's
    whole value proposition is an exact, trustworthy number; silently
    reporting the wrong sentiment with 100% confidence is worse than the
    model getting a fuzzier question wrong. When the raw query text names
    exactly one sentiment keyword unambiguously, that keyword wins over
    whatever decomposition extracted -- a substring check is zero-cost and
    more reliable than the LLM for this specific, checkable ambiguity. If
    zero or more than one keyword matches (e.g. "compare positive and
    negative reviews", genuinely ambiguous), decomposition's own extraction
    is trusted instead.
    """
    lowered = raw_query.lower()
    matched = {label for keyword, label in _SENTIMENT_KEYWORDS.items() if keyword in lowered}
    if len(matched) == 1:
        return next(iter(matched))
    return decomposed_filter


async def _compute_count(
    db: AsyncSession,
    restaurant_id: int,
    decomposed,
    sentiment_filter: str | None,
) -> int:
    """Direct Postgres COUNT(*) honoring sentiment/date/rating filters from decomposition."""
    stmt = (
        select(func.count())
        .select_from(ReviewChunkMeta)
        .where(ReviewChunkMeta.chunk_index == 0)
        .where(ReviewChunkMeta.restaurant_id == restaurant_id)
    )

    if sentiment_filter:
        stmt = stmt.where(ReviewChunkMeta.sentiment_label == sentiment_filter)

    if decomposed.date_filter:
        if decomposed.date_filter.from_date:
            with contextlib.suppress(ValueError):
                dt = datetime.fromisoformat(decomposed.date_filter.from_date).replace(tzinfo=UTC)
                stmt = stmt.where(ReviewChunkMeta.review_date >= dt)
        if decomposed.date_filter.to_date:
            with contextlib.suppress(ValueError):
                dt = datetime.fromisoformat(decomposed.date_filter.to_date).replace(tzinfo=UTC)
                stmt = stmt.where(ReviewChunkMeta.review_date <= dt)

    if decomposed.rating_filter:
        if decomposed.rating_filter.min is not None:
            stmt = stmt.where(ReviewChunkMeta.rating >= decomposed.rating_filter.min)
        if decomposed.rating_filter.max is not None:
            stmt = stmt.where(ReviewChunkMeta.rating <= decomposed.rating_filter.max)

    result = await db.execute(stmt)
    return result.scalar_one()


def _period_label(prefix: str, date_filter) -> str:
    if date_filter is None or (not date_filter.from_date and not date_filter.to_date):
        return f"{prefix} (all time)"
    frm = date_filter.from_date or "earliest"
    to = date_filter.to_date or "latest"
    return f"{prefix} ({frm} to {to})"


def _format_period_stats(label: str, stats: PeriodStats) -> str:
    rating_part = f"{stats.avg_rating}/5" if stats.avg_rating is not None else "N/A"
    sentiment_part = (
        ", ".join(f"{count} {label_}" for label_, count in sorted(stats.sentiment_counts.items()))
        or "none"
    )
    return (
        f"{label}: {stats.count} review{'s' if stats.count != 1 else ''}, "
        f"avg rating {rating_part}, sentiment breakdown: {sentiment_part}"
    )


async def _compute_trend_comparison(
    db: AsyncSession,
    restaurant_id: int,
    decomposed,
) -> str | None:
    """Exact dual-period stats (count, avg rating, sentiment breakdown) via direct SQL.

    Called only when decomposition populated compare_date_filter (a
    time-period comparison question). Both periods are computed exactly
    rather than estimated from a sample of retrieved evidence -- the same
    fabricated-precision failure mode the groundedness checks elsewhere in
    this app exist to catch.
    """
    if decomposed.compare_date_filter is None:
        return None

    current = await compute_period_stats(
        db,
        restaurant_id,
        decomposed.date_filter.from_date if decomposed.date_filter else None,
        decomposed.date_filter.to_date if decomposed.date_filter else None,
    )
    previous = await compute_period_stats(
        db,
        restaurant_id,
        decomposed.compare_date_filter.from_date,
        decomposed.compare_date_filter.to_date,
    )

    current_label = _period_label("Current period", decomposed.date_filter)
    previous_label = _period_label("Comparison period", decomposed.compare_date_filter)

    return (
        f"{_format_period_stats(current_label, current)} | "
        f"{_format_period_stats(previous_label, previous)}"
    )


def _format_count_answer(count: int, sentiment_filter: str | None) -> str:
    sentiment_part = f" {sentiment_filter.lower()}" if sentiment_filter else ""
    if count == 0:
        return f"No{sentiment_part} reviews match that filter."
    return f"You have {count}{sentiment_part} review{'s' if count != 1 else ''} in total."


async def _handle_count_query(
    db: AsyncSession,
    body: ChatQueryRequest,
    restaurant_id: int,
    decomposed,
    sanitized: str,
    trace: RequestTrace,
) -> tuple[str, uuid.UUID]:
    t0 = time.perf_counter()
    sentiment_filter = _resolve_sentiment_filter(sanitized, decomposed.sentiment_filter)
    count = await _compute_count(db, restaurant_id, decomposed, sentiment_filter)
    trace.generation_ms = (time.perf_counter() - t0) * 1000.0

    answer = _format_count_answer(count, sentiment_filter)

    msg_id = uuid.uuid4()
    trace.emit()
    return answer, msg_id


async def _pipeline_stream(
    body: ChatQueryRequest,
    restaurant_id: int,
    sanitized: str,
    decomposed,
    retrieval_query: str,
    db: AsyncSession,
    simple_client: BaseLLMClient,
    complex_client: BaseLLMClient,
    summary_client: BaseLLMClient,
    embedder: BaseEmbedder,
    vector_store: BaseVectorStore,
    cache: RedisCache,
    trace: RequestTrace,
    precomputed_count: str | None = None,
    precomputed_trend: str | None = None,
    precomputed_query_vector: list[float] | None = None,
) -> AsyncGenerator[dict, None]:
    settings_ = get_settings()
    message_id = uuid.uuid4()
    full_answer = ""

    try:
        # retrieval_query is passed in from chat_query() -- it's the same
        # text the semantic cache check above already embedded, so
        # precomputed_query_vector (if present) can be reused here instead
        # of embedding it a second time.
        params = build_retrieval_params(decomposed)

        retrieval_timing = RetrievalTiming()
        results = await hybrid_retrieve(
            query=retrieval_query,
            restaurant_id=restaurant_id,
            embedder=embedder,
            vector_store=vector_store,
            collection=settings_.qdrant_collection_reviews,
            top_k=params.top_k,
            date_from=params.date_from,
            date_to=params.date_to,
            rating_min=params.rating_min,
            rating_max=params.rating_max,
            reranker_model=settings_.reranker_model,
            precomputed_dense_vector=precomputed_query_vector,
            timing=retrieval_timing,
        )
        trace.embed_ms = retrieval_timing.embed_ms
        trace.search_ms = retrieval_timing.search_ms
        trace.rerank_ms = retrieval_timing.rerank_ms
        trace.retrieval_ms = retrieval_timing.embed_ms + retrieval_timing.search_ms

        # Ranking. results already carry the reranker's sigmoid relevance
        # score (src/core/reranker.py) -- do NOT re-run reciprocal_rank_fusion
        # here. hybrid_retrieve() already fuses dense+sparse server-side via
        # Qdrant-native RRF; calling reciprocal_rank_fusion([results]) on the
        # single already-reranked list was not a real fusion, it just
        # replaced the reranker's meaningful score with 1/(60+rank) (~0.01-0.03
        # for every result), which is what made EvidencePanel's "match %"
        # badge and _estimate_confidence()'s avg_relevance both useless.
        t1 = time.perf_counter()
        ranked = rank_results(
            results,
            settings_,
            top_k=params.top_k,
            has_explicit_date_filter=bool(decomposed.date_filter),
        )
        trace.ranking_ms = (time.perf_counter() - t1) * 1000.0
        trace.evidence_count = len(ranked.evidence)
        trace.low_evidence = ranked.low_evidence

        gate_answer = check_hallucination_gate(ranked, precomputed_count, precomputed_trend)

        if gate_answer is not None:
            # Hard hallucination gate: with zero retrieved evidence there is
            # nothing grounded to answer from. Prompt rule 1 ("never fabricate")
            # is a soft instruction the model can still ignore under real
            # traffic, so skip the LLM call entirely rather than trust it --
            # this also avoids paying for the correction/session-context
            # embedding calls and the generation call on a query we already
            # know can't be answered.
            model_used = "no_evidence_gate"
            full_answer = gate_answer
            for word in full_answer.split(" "):
                yield {"event": "token", "data": word + " "}
            trace.generation_ms = 0.0
            trace.generation_model = model_used
        else:
            # Correction lookup. A single flag isn't confirmed -- route it into
            # unverified_note (informational only) rather than corrections
            # (treated as ground truth, overriding conflicting evidence) until
            # it reaches CONSENSUS_THRESHOLD distinct flags.
            correction_match = await find_correction(
                query=sanitized,
                restaurant_id=restaurant_id,
                intent=decomposed.intent,
                embedder=embedder,
                vector_store=vector_store,
                threshold=settings_.correction_sim_threshold,
            )
            confirmed_correction = "None"
            unverified_note = "None"
            if correction_match is not None:
                if correction_match.is_consensus:
                    confirmed_correction = correction_match.text
                else:
                    unverified_note = correction_match.text

            # Session context
            session_context = await build_session_context(
                session_id=body.session_id,
                restaurant_id=restaurant_id,
                current_query=sanitized,
                db_session=db,
                vector_store=vector_store,
                embedder=embedder,
                recent_k=settings_.session_recent_messages,
                relevant_k=settings_.session_relevant_k,
                token_budget=settings_.session_context_token_budget,
            )

            selection = select_generation(
                decomposed, precomputed_count, settings_, precomputed_trend
            )
            model_used = selection.model_used
            gen_client = complex_client if selection.is_complex else simple_client
            loader = get_prompt_loader()

            gen_system, gen_user = build_generation_prompt(
                loader,
                selection.prompt_name,
                selection.is_complex,
                query=sanitized,
                session_context=session_context,
                corrections=confirmed_correction,
                unverified_note=unverified_note,
                evidence=format_evidence(ranked.evidence),
                sub_queries=decomposed.sub_queries,
                entity_counts=ranked.entity_counts,
                source_breakdown=ranked.source_breakdown,
                recency_spike=ranked.recency_spike,
                exact_count=precomputed_count,
                trend_comparison=precomputed_trend,
            )

            # LLM streaming
            t_gen = time.perf_counter()
            async for token in gen_client.stream(
                gen_user,
                system=gen_system,
                max_tokens=800 if selection.is_complex else 400,
                temperature=0.3,
                usage_callback=lambda p, c: trace.record_tokens(model_used, p, c),
            ):
                full_answer += token
                yield {"event": "token", "data": token}

            trace.generation_ms = (time.perf_counter() - t_gen) * 1000.0
            trace.generation_model = model_used

        # The simple/complex prompts now instruct the model to respond with
        # plain text directly (previously they asked for a JSON envelope,
        # which both broke token-by-token streaming -- the raw JSON was
        # visible mid-stream -- and was pure overhead, since evidence/
        # confidence/caveats/entity_counts/source_breakdown all come from
        # `ranked`, computed server-side, never from the model's output).
        # clean_answer_text is a defensive cleanup only, for the rare case a
        # model ignores the plain-text instruction and wraps its reply in a
        # fence or JSON.
        answer_text = clean_answer_text(full_answer)
        sub_answers: list[SubAnswer] = []

        # Groundedness heuristic: does the answer state a review/mention count
        # higher than what was actually retrieved? Cheap code-only check (no
        # extra LLM call) used as the accuracy signal alongside confidence.
        # A trend comparison exempts the same way a precomputed count does --
        # both periods' numbers come from direct SQL, not the retrieved
        # evidence sample, so they can legitimately exceed evidence_count.
        trace.groundedness_ok = check_count_groundedness(
            answer_text, len(ranked.evidence), precomputed_count or precomputed_trend
        )

        # Build structured response for the final event
        structured = build_structured_response(
            answer_text, sub_answers, ranked, trace.groundedness_ok
        )
        structured = validate_llm_output(structured)
        trace.confidence = structured.confidence
        # trace.cost_usd is already accumulated from real provider-reported
        # token usage via trace.record_tokens() (decomposition + generation
        # usage_callback hooks above) -- no separate estimate needed here.

        final_payload = {
            "message_id": str(message_id),
            "session_id": str(body.session_id),
            "response": structured.model_dump(),
            "cached": False,
            "complexity": decomposed.complexity,
            "model_used": model_used,
            "latency_ms": round(trace.total_ms),
            "cost_usd": round(trace.cost_usd, 6),
        }
        yield {"event": "done", "data": json.dumps(final_payload)}

        # Post-response: persist, session memory, cache — fire-and-forget.
        # Persists/caches structured.answer (the parsed text), not the raw
        # full_answer JSON blob -- otherwise reloaded history, session
        # context, and cache hits would all still show the unparsed JSON.
        fire_and_forget(
            _post_response_tasks(
                session_id=body.session_id,
                message_id=message_id,
                sanitized=sanitized,
                full_answer=structured.answer,
                structured=structured,
                model_used=model_used,
                complexity=decomposed.complexity,
                chunk_ids=[r.id for r in results[: len(ranked.evidence)]],
                embedder=embedder,
                vector_store=vector_store,
                cache=cache,
                restaurant_id=restaurant_id,
                summary_client=summary_client,
                summary_trigger=settings_.session_summary_trigger,
                retrieval_query=retrieval_query,
                precomputed_query_vector=precomputed_query_vector,
            ),
            name=f"post-response-{message_id}",
        )

    except Exception as exc:
        logger.error(
            "chat_pipeline_failed",
            session_id=str(body.session_id),
            error=str(exc),
            exc_info=True,
        )
        yield {
            "event": "error",
            "data": json.dumps(
                {
                    "error": "service_unavailable",
                    "message": ("I am temporarily unable to answer. Please try again in a moment."),
                }
            ),
        }
    finally:
        trace.emit()


async def _post_response_tasks(
    session_id: uuid.UUID,
    message_id: uuid.UUID,
    sanitized: str,
    full_answer: str,
    structured: ChatResponseSchema,
    model_used: str,
    complexity: str,
    chunk_ids: list[str],
    embedder: BaseEmbedder,
    vector_store: BaseVectorStore,
    cache: RedisCache,
    restaurant_id: int,
    summary_client: BaseLLMClient,
    summary_trigger: int,
    retrieval_query: str,
    precomputed_query_vector: list[float] | None = None,
) -> None:
    """Persist messages, update session memory, write cache.

    Runs as a fire-and-forget task (fire_and_forget(), not awaited) so it
    never delays the SSE response -- which means it must NOT reuse the
    request-scoped `db` session from Depends(get_db): FastAPI tears that
    session down as soon as the route handler returns, and this task keeps
    running after that, racing the teardown (confirmed live: intermittent
    "This transaction is closed" / IllegalStateChangeError, which silently
    dropped message persistence, session memory, and cache writes whenever
    it lost the race). Opens its own independent session instead, same
    pattern src/workers/ingest_worker.py already uses for the same reason.
    """
    from src.services.database import get_session_factory

    session_factory = get_session_factory()
    try:
        async with session_factory() as db:
            # Save user message
            user_msg = ChatMessage(
                id=message_id,
                session_id=session_id,
                role="user",
                content=sanitized,
            )
            db.add(user_msg)

            # Save assistant message
            asst_msg = ChatMessage(
                session_id=session_id,
                role="assistant",
                content=full_answer,
                retrieved_chunk_ids=chunk_ids,
                confidence=structured.confidence,
                model_used=model_used,
            )
            db.add(asst_msg)
            await db.commit()

            # Update session last_activity
            session_row = await db.get(ChatSession, session_id)
            if session_row:
                from datetime import UTC, datetime

                session_row.last_activity_at = datetime.now(tz=UTC)
                await db.commit()

            # Store user turn in Qdrant session memory
            await store_session_turn(
                session_id=session_id,
                restaurant_id=restaurant_id,
                role="user",
                content=sanitized,
                embedder=embedder,
                vector_store=vector_store,
            )

            # Maybe trigger rolling summary
            await maybe_trigger_summary(
                session_id=session_id,
                db_session=db,
                llm_client=summary_client,
                summary_trigger=summary_trigger,
            )

            # Cache write. Includes complexity/model_used so a future cache
            # hit can report what actually generated this answer instead of
            # a hardcoded guess -- previously every cache hit claimed
            # complexity="simple" even if a "complex" gpt-4.1 call produced
            # it, which is misleading for cost/observability.
            cache_value = {
                "answer": full_answer,
                "evidence": [e.model_dump() for e in structured.evidence],
                "confidence": structured.confidence,
                "caveats": structured.caveats,
                "entity_counts": structured.entity_counts,
                "source_breakdown": structured.source_breakdown,
                "complexity": complexity,
                "model_used": model_used,
            }
            # Exact-text tier: cheap, catches a literal repeat of this message
            # without needing an embedding or a Qdrant round-trip.
            await cache.set(restaurant_id, sanitized, cache_value)
            # Semantic tier: keyed on the decomposed/rephrased query so a
            # differently-worded paraphrase can hit too. Reuses the embedding
            # already computed for the semantic-cache lookup on this request
            # (precomputed_query_vector) instead of embedding again.
            await store_cached_response(
                retrieval_query,
                restaurant_id,
                cache_value,
                embedder,
                vector_store,
                cache,
                get_settings().qdrant_collection_chat_cache,
                precomputed_vector=precomputed_query_vector,
            )
    except Exception as exc:
        logger.warning("post_response_tasks_failed", error=str(exc))


async def _yield_cached(data: dict, session_id: uuid.UUID) -> AsyncGenerator[dict, None]:
    """Emit a cached response as a single 'done' SSE event."""
    response = ChatResponseSchema(
        answer=data.get("answer", ""),
        evidence=[EvidenceItem(**e) for e in data.get("evidence", [])],
        confidence=data.get("confidence", 0.0),
        caveats=data.get("caveats"),
        entity_counts=data.get("entity_counts", {}),
        source_breakdown=data.get("source_breakdown", {}),
    )
    payload = {
        "message_id": str(uuid.uuid4()),
        "session_id": str(session_id),
        "response": response.model_dump(),
        "cached": True,
        # Reflects whatever actually generated this answer (stored alongside
        # it in cache.set()) rather than a hardcoded guess -- a cache hit of
        # a "complex" gpt-4.1 call previously always misreported "simple".
        "complexity": data.get("complexity", "simple"),
        "model_used": "cache",
    }
    yield {"event": "done", "data": json.dumps(payload)}


async def _yield_instant(
    response: ChatResponseSchema,
    session_id: uuid.UUID,
    message_id: uuid.UUID,
    model_used: str = "none",
) -> AsyncGenerator[dict, None]:
    """Emit a non-streamed response (guardrail, count_query) as a single 'done' event."""
    payload = {
        "message_id": str(message_id),
        "session_id": str(session_id),
        "response": response.model_dump(),
        "cached": False,
        "complexity": "simple",
        "model_used": model_used,
    }
    yield {"event": "done", "data": json.dumps(payload)}
