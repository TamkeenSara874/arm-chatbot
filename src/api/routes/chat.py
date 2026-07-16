"""Chat API routes: sessions, query (SSE), history, corrections, and report."""

# NOTE: deliberately no `from __future__ import annotations` here. Combined
# with slowapi's @limiter.limit() decorator, postponed evaluation breaks
# FastAPI's dependant analysis for every decorated route in this file --
# it can no longer resolve `body: ChatQueryRequest` / Depends() types, so it
# silently treats them as required query parameters instead of a JSON body /
# dependency injection, and every such route 422s on any real request. See
# the identical fix + repro notes in src/api/routes/ingest.py.

import asyncio
import contextlib
import json
import re
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
from src.core.crisis_guardrail import CRISIS_RESPONSE, detect_crisis_language
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
from src.core.review_stats import (
    PeriodStats,
    compute_period_stats,
    compute_theme_cooccurrence_count,
    compute_theme_count,
)
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
from src.utils.metrics import active_sessions_gauge, count_query_total
from src.utils.security import redact_reviewer_names, sanitize_input, validate_llm_output
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

    # Crisis check -- ahead of the cache, decomposition, and every other
    # guardrail. Deterministic and code-only on purpose: a distress signal
    # can be buried inside an otherwise ordinary-looking business question
    # (confirmed live: "I want to die, the reviews are so bad" still
    # decomposed as a normal count_query and got a data-only answer with no
    # acknowledgment at all) -- this must never depend on decomposition, a
    # cache hit, or the generation prompt's general tone rule noticing it.
    if detect_crisis_language(sanitized):
        trace.emit()
        crisis_msg_id = uuid.uuid4()
        fire_and_forget(
            _persist_instant_exchange(
                session_id=body.session_id,
                message_id=crisis_msg_id,
                sanitized=sanitized,
                answer=CRISIS_RESPONSE,
                model_used="crisis_response",
                restaurant_id=restaurant_id,
                embedder=embedder,
                vector_store=vector_store,
            ),
            name=f"persist-crisis-{crisis_msg_id}",
        )
        return EventSourceResponse(
            _yield_instant(
                ChatResponseSchema(answer=CRISIS_RESPONSE, evidence=[], confidence=1.0),
                body.session_id,
                crisis_msg_id,
                model_used="crisis_response",
            )
        )

    # Cache check before any LLM work -- skipped on a regenerate request
    # (body.bypass_cache), which exists specifically so "Regenerate" produces
    # an actually-fresh answer instead of the same cached one it's replacing.
    cached_data = None if body.bypass_cache else await cache.get(restaurant_id, sanitized)
    if cached_data:
        trace.cache_hit = True
        trace.emit()
        cache_msg_id = uuid.uuid4()
        fire_and_forget(
            _persist_instant_exchange(
                session_id=body.session_id,
                message_id=cache_msg_id,
                sanitized=sanitized,
                answer=cached_data.get("answer", ""),
                model_used=cached_data.get("model_used", "cache"),
                restaurant_id=restaurant_id,
                embedder=embedder,
                vector_store=vector_store,
            ),
            name=f"persist-cache-{cache_msg_id}",
        )
        return EventSourceResponse(_yield_cached(cached_data, body.session_id, cache_msg_id))

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
        usage_callback=lambda p, c, ca: trace.record_tokens(settings.groq_decomp_model, p, c, ca),
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
        guardrail_msg_id = uuid.uuid4()
        fire_and_forget(
            _persist_instant_exchange(
                session_id=body.session_id,
                message_id=guardrail_msg_id,
                sanitized=sanitized,
                answer=guardrail_text,
                model_used="guardrail",
                restaurant_id=restaurant_id,
                embedder=embedder,
                vector_store=vector_store,
            ),
            name=f"persist-guardrail-{guardrail_msg_id}",
        )
        return EventSourceResponse(
            _yield_instant(response, body.session_id, guardrail_msg_id, model_used="guardrail")
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
            usage_callback=lambda p, c, ca: trace.record_tokens(
                settings.openai_simple_model, p, c, ca
            ),
        )
        trace.generation_ms = (time.perf_counter() - t_recall) * 1000.0
        trace.generation_model = settings.openai_simple_model
        trace.emit()
        recall_msg_id = uuid.uuid4()
        fire_and_forget(
            _persist_instant_exchange(
                session_id=body.session_id,
                message_id=recall_msg_id,
                sanitized=sanitized,
                answer=recall_answer.strip(),
                model_used=settings.openai_simple_model,
                restaurant_id=restaurant_id,
                embedder=embedder,
                vector_store=vector_store,
            ),
            name=f"persist-recall-{recall_msg_id}",
        )
        return EventSourceResponse(
            _yield_instant(
                ChatResponseSchema(answer=recall_answer.strip(), evidence=[], confidence=1.0),
                body.session_id,
                recall_msg_id,
                model_used=settings.openai_simple_model,
            )
        )

    # Reply-status fast path — checked ahead of count_query since "how many
    # reviews haven't I replied to" would otherwise be misclassified as an
    # exact-countable question and answered with an unrelated number (see
    # _is_reply_status_question's docstring). No DB column backs this at all,
    # so this is an honest "I don't have that" rather than a computed answer.
    if _is_reply_status_question(sanitized):
        trace.emit()
        reply_status_msg_id = uuid.uuid4()
        fire_and_forget(
            _persist_instant_exchange(
                session_id=body.session_id,
                message_id=reply_status_msg_id,
                sanitized=sanitized,
                answer=REPLY_STATUS_ANSWER,
                model_used="reply_status_gate",
                restaurant_id=restaurant_id,
                embedder=embedder,
                vector_store=vector_store,
            ),
            name=f"persist-reply-status-{reply_status_msg_id}",
        )
        return EventSourceResponse(
            _yield_instant(
                ChatResponseSchema(answer=REPLY_STATUS_ANSWER, evidence=[], confidence=1.0),
                body.session_id,
                reply_status_msg_id,
                model_used="reply_status_gate",
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
            fire_and_forget(
                _persist_instant_exchange(
                    session_id=body.session_id,
                    message_id=count_msg_id,
                    sanitized=sanitized,
                    answer=count_answer,
                    model_used="direct_query",
                    restaurant_id=restaurant_id,
                    embedder=embedder,
                    vector_store=vector_store,
                ),
                name=f"persist-count-{count_msg_id}",
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
        precomputed_count = _format_count_answer(count, sentiment_filter, decomposed.content_filter)
    elif decomposed.content_filter is not None:
        # Same rationale as compare_date_filter/breakdown_dimension below: content_filter
        # is only ever set by decomposition when the question has a real, exact
        # has-text/no-text count component, independent of what intent label the
        # classifier assigned -- e.g. "roughly how many reviews have no written text,
        # and what does that imply for X" can come back as intent=aggregation rather
        # than count_query. Trust the structured field, not the intent guess, so the
        # exact count still gets computed and stated verbatim instead of the LLM
        # estimating it from only the top_k retrieved evidence chunks.
        sentiment_filter = _resolve_sentiment_filter(sanitized, decomposed.sentiment_filter)
        count = await _compute_count(db, restaurant_id, decomposed, sentiment_filter)
        precomputed_count = _format_count_answer(count, sentiment_filter, decomposed.content_filter)

    # Trend-comparison fast path — decomposition sets compare_date_filter for
    # a time-period comparison question ("since last month", "compared to
    # last week"). Both periods' exact stats are computed via direct SQL
    # here, same rationale as count_query above, and threaded into the
    # complex generation prompt below rather than left for the LLM to
    # estimate from a sample of retrieved evidence.
    precomputed_trend: str | None = None
    if decomposed.compare_date_filter is not None:
        precomputed_trend = await _compute_trend_comparison(db, restaurant_id, decomposed)

    # Breakdown fast path — decomposition sets breakdown_dimension for a
    # whole-dataset distribution/ranking question ("which source has the
    # most reviews"). Computed via direct SQL GROUP BY, same rationale as
    # count_query/trend above, and threaded into the complex generation
    # prompt so the model states real totals instead of tallying whatever
    # top_k evidence happened to be retrieved and presenting that as if it
    # were the full restaurant's numbers.
    precomputed_breakdown: str | None = None
    if decomposed.breakdown_dimension is not None:
        breakdown = await _compute_breakdown(db, restaurant_id, decomposed.breakdown_dimension)
        precomputed_breakdown = _format_breakdown_answer(breakdown, decomposed.breakdown_dimension)

    # Overall-rating fast path — decomposition sets wants_overall_stats for a
    # plain "what's my overall rating" / "how are my reviews doing" / "what
    # percentage of my reviews are negative" question. Left to the normal RAG
    # path, this was estimating the average rating (or sentiment percentage)
    # from only the top_k retrieved evidence chunks (confirmed live: ~3.5/5
    # from a 6-review sample vs. the real ~3.9-4.0 computed average; and
    # separately, a "what % negative" question that retrieves review text
    # containing "negative" skews its own sample negative, estimating 75%
    # against a real ~19%) -- the same fabricated-precision failure mode as
    # count_query/trend/breakdown above, just for a question shape those
    # three don't cover. Computed via the same compute_period_stats already
    # used by trend comparison and anomaly detection.
    #
    # Deliberately gated on the independent wants_overall_stats field, not on
    # decomposed.intent == "sentiment_overview" -- a compound question ("what
    # % of my reviews are negative and how worried should I be") must not
    # lose this signal just because the reasoning tail pulls intent
    # classification toward aggregation instead. See EXACT-STAT FIELDS ARE
    # INDEPENDENT OF INTENT in query_decomposition.yaml.
    precomputed_overall_stats: str | None = None
    if decomposed.wants_overall_stats:
        overall_stats = await _compute_overall_stats(db, restaurant_id, decomposed)
        precomputed_overall_stats = _format_overall_stats_answer(
            overall_stats, decomposed.date_filter
        )

    # Theme-count fast path — decomposition sets theme_keywords for a
    # qualitative-theme count question ("how many people called my staff
    # rude", "how many reviews complain about cold food"). These have no
    # dedicated database column (unlike count_query's rating/sentiment/
    # source/content_filter), so before this they were answered honestly but
    # only from the top_k=20 retrieved sample ("at least 12 of the 20
    # retrieved reviews..."), which could badly undercount a theme that
    # actually appears in far more of the 2,753 reviews. Counted via
    # full_review ILIKE across the WHOLE restaurant instead -- still a
    # keyword-match approximation (misses differently-worded mentions), but
    # scans every review, not just whatever retrieval happened to surface.
    precomputed_theme_count: str | None = None
    if decomposed.theme_keywords:
        date_from = decomposed.date_filter.from_date if decomposed.date_filter else None
        date_to = decomposed.date_filter.to_date if decomposed.date_filter else None
        if decomposed.compare_theme_keywords and decomposed.theme_require_both:
            # Co-occurrence ("which reviews mention both slow service and
            # cold food?") -- confirmed live as a real bug: with no way to
            # express "both together," decomposition folded both themes into
            # one flat OR-matched list, so the exact count reported "any
            # review mentioning slow service OR cold food" while worded as
            # if it meant "both at once" -- directly contradicting the
            # model's own read of the retrieved evidence, which found no
            # review actually mentioning both. This is a real AND across the
            # two keyword groups, computed via a dedicated SQL query.
            both_count = await compute_theme_cooccurrence_count(
                db,
                restaurant_id,
                decomposed.theme_keywords,
                decomposed.compare_theme_keywords,
                date_from,
                date_to,
            )
            precomputed_theme_count = _format_theme_cooccurrence_answer(
                both_count, decomposed.theme_keywords, decomposed.compare_theme_keywords
            )
        elif decomposed.compare_theme_keywords:
            # Two-theme comparison ("which has more complaints: food quality
            # or staff behavior, and by how much?") -- confirmed live as a
            # real gap: decomposition tried to set theme_keywords to the
            # literal phrase "complaints about food quality" (matching zero
            # reviews, since nobody writes that exact phrase), so the model
            # fell back to eyeballing which theme seemed more common in only
            # the 20 retrieved reviews. Both themes now get their own exact,
            # whole-corpus count the same way a single theme does.
            theme_count = await compute_theme_count(
                db, restaurant_id, decomposed.theme_keywords, date_from, date_to
            )
            compare_count = await compute_theme_count(
                db, restaurant_id, decomposed.compare_theme_keywords, date_from, date_to
            )
            precomputed_theme_count = _format_theme_comparison_answer(
                theme_count,
                decomposed.theme_keywords,
                compare_count,
                decomposed.compare_theme_keywords,
            )
        else:
            theme_count = await compute_theme_count(
                db, restaurant_id, decomposed.theme_keywords, date_from, date_to
            )
            precomputed_theme_count = _format_theme_count_answer(
                theme_count, decomposed.theme_keywords
            )

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
    # bypass_cache still reuses cache_query_vector for retrieval below (no
    # point embedding the same text twice), it just refuses to treat a
    # semantic match as a hit -- same reasoning as the exact-text skip above.
    if body.bypass_cache:
        cached_semantic = None
    if cached_semantic:
        trace.cache_hit = True
        trace.emit()
        semantic_msg_id = uuid.uuid4()
        fire_and_forget(
            _persist_instant_exchange(
                session_id=body.session_id,
                message_id=semantic_msg_id,
                sanitized=sanitized,
                answer=cached_semantic.get("answer", ""),
                model_used=cached_semantic.get("model_used", "cache"),
                restaurant_id=restaurant_id,
                embedder=embedder,
                vector_store=vector_store,
            ),
            name=f"persist-semantic-cache-{semantic_msg_id}",
        )
        return EventSourceResponse(_yield_cached(cached_semantic, body.session_id, semantic_msg_id))

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
            precomputed_breakdown=precomputed_breakdown,
            precomputed_overall_stats=precomputed_overall_stats,
            precomputed_theme_count=precomputed_theme_count,
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

# Deterministic keyword check for "reply status" questions ("how many reviews
# haven't I replied to?", "show me my unanswered reviews"). ReviewChunkMeta
# has no reply/response-status column at all -- that's tracked only in the
# AIO dashboard's own review-management UI, never ingested into this system.
# Confirmed live: left unguarded, count_query's fast path silently answers
# with an unrelated exact number (total review count, or a sentiment-filtered
# count) stated at confidence 1.0, since it has no "unreplied" column to
# filter by and just drops that part of the question instead of flagging it
# as unanswerable. A multi-word phrase list (not single words like "reply")
# keeps this from firing on unrelated questions that happen to contain
# "answer"/"response" in another sense.
# Regex, not rigid substrings -- confirmed live that a naive phrase list like
# "haven't replied" misses natural question-word-order phrasing such as
# "reviews haven't I replied to yet?", where a pronoun sits between the
# auxiliary and the verb. `.{0,15}` tolerates that gap (a short pronoun/adverb)
# without matching across unrelated clauses.
_REPLY_STATUS_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"have\s*n[o']?t\b.{0,15}\b(repl(y|ied|ies)|respond(ed)?|answer(ed)?)",
        r"\bun(replied|answered|responded)\b",
        r"\bnot\s+(yet\s+)?(repl(y|ied)|respond(ed)?|answer(ed)?)\b",
        r"\breply\s+status\b",
        r"\bresponse\s+status\b",
    )
)

REPLY_STATUS_ANSWER = (
    "I don't have visibility into which reviews you've replied to -- reply/response status is "
    "tracked in your AIO dashboard's review management screen, not in the review content I have "
    "access to. I can help with what the reviews themselves say, though -- want to ask about that?"
)


def _is_reply_status_question(raw_query: str) -> bool:
    return any(p.search(raw_query) for p in _REPLY_STATUS_PATTERNS)


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

    # Without this, a count_query landing on a "how many reviews have a
    # rating but no written text" style question had no way to express that
    # condition -- it silently fell back to an unfiltered total and stated
    # that as if it answered the actual question asked.
    if decomposed.content_filter == "no_text":
        stmt = stmt.where(ReviewChunkMeta.has_content.is_(False))
    elif decomposed.content_filter == "has_text":
        stmt = stmt.where(ReviewChunkMeta.has_content.is_(True))

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


_BREAKDOWN_COLUMNS = {
    "source": ReviewChunkMeta.source,
    "rating": ReviewChunkMeta.rating,
    "sentiment": ReviewChunkMeta.sentiment_label,
}


async def _compute_breakdown(
    db: AsyncSession,
    restaurant_id: int,
    dimension: str,
) -> dict[str, tuple[int, float | None]]:
    """Exact GROUP BY count + avg rating across the WHOLE restaurant's
    reviews by one dimension (source/rating/sentiment) via direct SQL.

    Mirrors _compute_count/_compute_trend_comparison's rationale: a "which
    source has the most reviews" question needs the real total, not a tally
    over whatever top_k evidence happened to be retrieved for this query.
    `dimension` is only ever one of the three keys in _BREAKDOWN_COLUMNS
    (constrained by DecomposedQuery's Literal type), so this never builds a
    column reference from raw user/model text.

    Also computes avg rating per group -- added after a real gap surfaced
    live: "why is my rating on Yelp so much lower than on Google?" had no
    exact per-platform rating to reason from at all (only per-platform
    review *counts* were computed), so the model could only speculate about
    generic reasons rather than starting from the real numbers the dashboard
    itself shows (e.g. "Google 4.0 vs Yelp 3.2").
    """
    column = _BREAKDOWN_COLUMNS[dimension]
    stmt = (
        select(column, func.count(), func.avg(ReviewChunkMeta.rating))
        .select_from(ReviewChunkMeta)
        .where(ReviewChunkMeta.chunk_index == 0)
        .where(ReviewChunkMeta.restaurant_id == restaurant_id)
        .where(column.is_not(None))
        .group_by(column)
    )
    result = await db.execute(stmt)
    return {
        str(key): (count, round(avg_rating, 2) if avg_rating is not None else None)
        for key, count, avg_rating in result.all()
    }


async def _compute_overall_stats(
    db: AsyncSession,
    restaurant_id: int,
    decomposed,
) -> PeriodStats:
    """Exact review count, average rating, and sentiment breakdown via direct
    SQL, for whatever date range decomposition extracted (or all-time if
    none). Reuses compute_period_stats (already shared by trend comparison
    and anomaly detection) rather than estimating an average rating from a
    handful of retrieved evidence chunks, which can differ meaningfully from
    the true average (confirmed live: a 6-review sample estimated ~3.5/5
    while the real computed average was 3.9-4.0).
    """
    date_from = decomposed.date_filter.from_date if decomposed.date_filter else None
    date_to = decomposed.date_filter.to_date if decomposed.date_filter else None
    return await compute_period_stats(db, restaurant_id, date_from, date_to)


def _format_overall_stats_answer(stats: PeriodStats, date_filter) -> str:
    period = " in the selected period" if date_filter else " (all-time)"
    if stats.count == 0:
        return f"No reviews found{period}."
    if stats.avg_rating is None:
        return f"You have {stats.count} reviews{period}, but no star ratings are recorded for them."
    # Percentages are precomputed here (not left for the LLM to divide count/total
    # itself) so a "what percentage of my reviews are negative" question gets the
    # exact figure verbatim -- the same rationale as stating count/avg_rating verbatim.
    sentiment_parts = ", ".join(
        f"{label}: {count} ({count / stats.count:.0%})"
        for label, count in sorted(stats.sentiment_counts.items(), key=lambda x: -x[1])
    )
    return (
        f"Exact stats{period}: {stats.count} reviews, average rating {stats.avg_rating}/5. "
        f"Sentiment breakdown -- {sentiment_parts}."
    )


def _format_theme_count_answer(count: int, keywords: list[str]) -> str:
    # Phrased for a non-technical restaurant owner, not as an internal
    # mechanism description -- "keyword search"/"keyword match" leaked
    # through to a live answer once already (confirmed live) because this
    # text is threaded verbatim into the generation prompt as the caveat the
    # model is told to keep. Say what the number covers and its real
    # limitation, without naming the technique used to compute it.
    keyword_list = ", ".join(f'"{kw}"' for kw in keywords)
    return (
        f"Exact count across all your reviews: {count} review(s) mention {keyword_list} "
        "(or similar wording). This covers every review, not just a sample, but it may "
        "still miss reviews that describe the same issue using different words."
    )


def _format_theme_comparison_answer(
    count_a: int, keywords_a: list[str], count_b: int, keywords_b: list[str]
) -> str:
    words_a = ", ".join(f'"{kw}"' for kw in keywords_a)
    words_b = ", ".join(f'"{kw}"' for kw in keywords_b)
    diff = abs(count_a - count_b)
    if count_a == count_b:
        comparison = f"Both appear in the same number of reviews ({count_a})."
    else:
        leader_words = words_a if count_a > count_b else words_b
        comparison = f"{leader_words} appears in {diff} more review(s)."
    return (
        f"Exact counts across all your reviews, not just a sample: {count_a} review(s) mention "
        f"{words_a} (or similar wording); {count_b} review(s) mention {words_b} (or similar "
        f"wording). {comparison} Both counts may still miss reviews phrased differently."
    )


def _format_theme_cooccurrence_answer(
    count: int, keywords_a: list[str], keywords_b: list[str]
) -> str:
    words_a = ", ".join(f'"{kw}"' for kw in keywords_a)
    words_b = ", ".join(f'"{kw}"' for kw in keywords_b)
    return (
        f"Exact count across all your reviews, not just a sample: {count} review(s) mention "
        f"{words_a} AND {words_b} together in the same review. This may still miss reviews "
        "phrased differently, but it is not limited to whatever reviews happened to be retrieved."
    )


def _format_breakdown_answer(breakdown: dict[str, tuple[int, float | None]], dimension: str) -> str:
    if not breakdown:
        return f"No reviews have a recorded {dimension}."
    ordered = sorted(breakdown.items(), key=lambda x: -x[1][0])
    if dimension == "source":
        # Avg rating per source is the actual number a "why is my rating on
        # Yelp lower than Google" question needs -- count alone can't answer it.
        parts = ", ".join(
            f"{label}: {count} reviews, avg rating {avg_rating}/5"
            if avg_rating is not None
            else f"{label}: {count} reviews"
            for label, (count, avg_rating) in ordered
        )
    else:
        parts = ", ".join(f"{label}: {count}" for label, (count, _) in ordered)
    total = sum(count for count, _ in breakdown.values())
    return f"Exact breakdown by {dimension} across all {total} reviews -- {parts}."


def _format_count_answer(
    count: int, sentiment_filter: str | None, content_filter: str | None = None
) -> str:
    sentiment_part = f" {sentiment_filter.lower()}" if sentiment_filter else ""
    content_part = ""
    if content_filter == "no_text":
        content_part = " with a rating but no written text"
    elif content_filter == "has_text":
        content_part = " with written text"
    if count == 0:
        return f"No{sentiment_part} reviews{content_part} match that filter."
    return f"You have {count}{sentiment_part} review{'s' if count != 1 else ''}{content_part} in total."


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

    answer = _format_count_answer(count, sentiment_filter, decomposed.content_filter)

    msg_id = uuid.uuid4()
    trace.emit()
    count_query_total.inc()
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
    precomputed_breakdown: str | None = None,
    precomputed_overall_stats: str | None = None,
    precomputed_theme_count: str | None = None,
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
            source_filter=params.source_filter,
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
        ranked = await rank_results(
            results,
            settings_,
            top_k=params.top_k,
            has_explicit_date_filter=bool(decomposed.date_filter),
            query=retrieval_query,
            reranker_model=settings_.reranker_model,
            reranked=retrieval_timing.reranked,
        )
        trace.ranking_ms = (time.perf_counter() - t1) * 1000.0
        trace.evidence_count = len(ranked.evidence)
        trace.low_evidence = ranked.low_evidence

        gate_answer = check_hallucination_gate(
            ranked,
            precomputed_count,
            precomputed_trend,
            precomputed_breakdown,
            precomputed_overall_stats,
            precomputed_theme_count,
        )

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
            # Correction lookup and session context have no data dependency on
            # each other, and both independently embed this same query text --
            # embed it once here and run both concurrently instead of two
            # sequential calls each re-embedding the same string. On an embed
            # failure here, query_vector stays None and each function falls
            # back to embedding it itself (their existing per-function
            # try/except already tolerates that), so this is a pure latency
            # optimization with no change to failure-mode behavior.
            try:
                query_vector = await embedder.embed_one(sanitized)
            except Exception as exc:
                logger.warning("shared_query_embed_failed", error=str(exc))
                query_vector = None

            # Correction lookup. A single flag isn't confirmed -- route it into
            # unverified_note (informational only) rather than corrections
            # (treated as ground truth, overriding conflicting evidence) until
            # it reaches CONSENSUS_THRESHOLD distinct flags.
            correction_match, session_context = await asyncio.gather(
                find_correction(
                    query=sanitized,
                    restaurant_id=restaurant_id,
                    intent=decomposed.intent,
                    embedder=embedder,
                    vector_store=vector_store,
                    threshold=settings_.correction_sim_threshold,
                    precomputed_vector=query_vector,
                ),
                build_session_context(
                    session_id=body.session_id,
                    restaurant_id=restaurant_id,
                    current_query=sanitized,
                    db_session=db,
                    vector_store=vector_store,
                    embedder=embedder,
                    recent_k=settings_.session_recent_messages,
                    relevant_k=settings_.session_relevant_k,
                    token_budget=settings_.session_context_token_budget,
                    precomputed_query_vector=query_vector,
                ),
            )
            confirmed_correction = "None"
            unverified_note = "None"
            if correction_match is not None:
                if correction_match.is_consensus:
                    confirmed_correction = correction_match.text
                else:
                    unverified_note = correction_match.text

            selection = select_generation(
                decomposed,
                precomputed_count,
                settings_,
                precomputed_trend,
                precomputed_breakdown,
                precomputed_overall_stats,
                precomputed_theme_count,
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
                exact_breakdown=precomputed_breakdown,
                overall_stats=precomputed_overall_stats,
                theme_count=precomputed_theme_count,
            )

            # LLM streaming
            t_gen = time.perf_counter()
            async for token in gen_client.stream(
                gen_user,
                system=gen_system,
                max_tokens=800 if selection.is_complex else 400,
                temperature=0.3,
                usage_callback=lambda p, c, ca: trace.record_tokens(model_used, p, c, ca),
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
        # Deterministic backstop for the REVIEWER NAME PRIVACY prompt rule --
        # confirmed live that the rule alone isn't a guarantee (a reviewer's
        # real username showed up attached to an example despite never being
        # asked for). Runs unconditionally, regardless of whether the model
        # complied with the prompt rule.
        answer_text = redact_reviewer_names(answer_text, ranked.evidence, sanitized)
        sub_answers: list[SubAnswer] = []

        # Groundedness heuristic: does the answer state a review/mention count
        # higher than what was actually retrieved? Cheap code-only check (no
        # extra LLM call) used as the accuracy signal alongside confidence.
        # A trend comparison exempts the same way a precomputed count does --
        # both periods' numbers come from direct SQL, not the retrieved
        # evidence sample, so they can legitimately exceed evidence_count.
        trace.groundedness_ok = check_count_groundedness(
            answer_text,
            len(ranked.evidence),
            precomputed_count
            or precomputed_trend
            or precomputed_breakdown
            or precomputed_overall_stats
            or precomputed_theme_count,
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


async def _persist_instant_exchange(
    session_id: uuid.UUID,
    message_id: uuid.UUID,
    sanitized: str,
    answer: str,
    model_used: str,
    restaurant_id: int,
    embedder: BaseEmbedder,
    vector_store: BaseVectorStore,
) -> None:
    """Persist the user+assistant turn for fast paths that skip the full
    pipeline (guardrail, conversation_recall, count_query).

    Without this, these turns never reach Postgres or Qdrant session_memory --
    both build_recent_turns_context (Postgres) and build_session_context's
    cross-session ANN search (Qdrant) only ever see persisted rows, so a
    guardrail decline or count-query answer would silently vanish from chat
    history and from every future session-context/conversation_recall lookup,
    including its own. Confirmed live: a corrections lookup against a
    count-query answer found no original_response for the same reason.
    """
    from src.services.database import get_session_factory

    session_factory = get_session_factory()
    try:
        async with session_factory() as db:
            user_msg = ChatMessage(
                id=message_id, session_id=session_id, role="user", content=sanitized
            )
            db.add(user_msg)
            asst_msg = ChatMessage(
                session_id=session_id,
                role="assistant",
                content=answer,
                model_used=model_used,
            )
            db.add(asst_msg)
            await db.commit()

            session_row = await db.get(ChatSession, session_id)
            if session_row:
                session_row.last_activity_at = datetime.now(tz=UTC)
                await db.commit()

            await store_session_turn(
                session_id=session_id,
                restaurant_id=restaurant_id,
                role="user",
                content=sanitized,
                embedder=embedder,
                vector_store=vector_store,
            )
    except Exception as exc:
        logger.warning("instant_persist_failed", error=str(exc))


async def _yield_cached(
    data: dict, session_id: uuid.UUID, message_id: uuid.UUID
) -> AsyncGenerator[dict, None]:
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
        "message_id": str(message_id),
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
