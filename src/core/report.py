from __future__ import annotations

import json
from datetime import UTC, date, datetime
from typing import Any

import structlog
from openai import AsyncOpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.db_entities import ReviewChunkMeta
from src.models.schemas import InsightsReport
from src.services.vector.base import BaseVectorStore
from src.utils.metrics import report_generated_total

logger = structlog.get_logger()

EXPORT_INSIGHTS_REPORT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "export_insights_report",
        "description": (
            "Aggregate review data and generate a structured insights report for a restaurant. "
            "Call this with the date range and aspects the user mentioned."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "date_from": {
                    "type": "string",
                    "description": "Start date ISO YYYY-MM-DD or null for all time",
                },
                "date_to": {
                    "type": "string",
                    "description": "End date ISO YYYY-MM-DD or null for all time",
                },
                "aspects": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Aspects to focus on (food, service, ambiance). Empty = all.",
                },
            },
            "required": [],
        },
    },
}


async def generate_report(
    user_message: str,
    restaurant_id: int,
    db_session: AsyncSession,
    vector_store: BaseVectorStore,
    qdrant_reviews_collection: str,
    openai_client: AsyncOpenAI,
    model: str,
    date_from: date | None = None,
    date_to: date | None = None,
) -> InsightsReport:
    """Generate an insights report via OpenAI tool calling + DB aggregations.

    Flow:
    1. OpenAI extracts date range parameters from the user's natural language
    2. Backend runs Postgres aggregations (totals, averages, distributions)
    3. Qdrant scroll aggregates entity mentions by sentiment polarity
    4. OpenAI generates a plain-English summary paragraph
    5. Markdown is assembled and returned in InsightsReport
    """
    extracted_from, extracted_to = await _extract_date_range(
        openai_client, model, user_message, date_from, date_to
    )

    rows = await _load_review_rows(db_session, restaurant_id, extracted_from, extracted_to)

    total_reviews = len(rows)
    ratings = [r.rating for r in rows if r.rating is not None]
    avg_rating = round(sum(ratings) / len(ratings), 2) if ratings else None

    rating_distribution: dict[str, int] = {}
    for r in rows:
        if r.rating is not None:
            key = str(int(r.rating))
            rating_distribution[key] = rating_distribution.get(key, 0) + 1

    sentiment_breakdown: dict[str, int] = {}
    for r in rows:
        label = r.sentiment_label or "Unknown"
        sentiment_breakdown[label] = sentiment_breakdown.get(label, 0) + 1

    source_breakdown: dict[str, int] = {}
    for r in rows:
        src = r.source or "Unknown"
        source_breakdown[src] = source_breakdown.get(src, 0) + 1

    top_praised, top_complained = await _aggregate_entities(
        vector_store, qdrant_reviews_collection, restaurant_id, extracted_from, extracted_to
    )

    summary = await _generate_summary(
        openai_client=openai_client,
        model=model,
        total_reviews=total_reviews,
        avg_rating=avg_rating,
        sentiment_breakdown=sentiment_breakdown,
        top_praised=top_praised,
        top_complained=top_complained,
    )

    markdown = _build_markdown(
        restaurant_id=restaurant_id,
        date_from=extracted_from,
        date_to=extracted_to,
        total_reviews=total_reviews,
        avg_rating=avg_rating,
        rating_distribution=rating_distribution,
        sentiment_breakdown=sentiment_breakdown,
        source_breakdown=source_breakdown,
        top_praised=top_praised,
        top_complained=top_complained,
        summary=summary,
    )

    report_generated_total.labels(restaurant_id=str(restaurant_id)).inc()
    logger.info("report_generated", restaurant_id=restaurant_id, total_reviews=total_reviews)

    return InsightsReport(
        restaurant_id=restaurant_id,
        generated_at=datetime.now(tz=UTC),
        date_from=extracted_from,
        date_to=extracted_to,
        total_reviews=total_reviews,
        avg_rating=avg_rating,
        rating_distribution=rating_distribution,
        sentiment_breakdown=sentiment_breakdown,
        source_breakdown=source_breakdown,
        top_praised=top_praised,
        top_complained=top_complained,
        summary=summary,
        markdown=markdown,
    )


async def _extract_date_range(
    openai_client: AsyncOpenAI,
    model: str,
    user_message: str,
    fallback_from: date | None,
    fallback_to: date | None,
) -> tuple[date | None, date | None]:
    """Use OpenAI tool calling to extract a date range from natural language."""
    try:
        response = await openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant that extracts date parameters. "
                        "When the user asks for a report, call export_insights_report "
                        "with the date range they mentioned. If no dates are mentioned, "
                        "pass null for both."
                    ),
                },
                {"role": "user", "content": user_message},
            ],
            tools=[EXPORT_INSIGHTS_REPORT_TOOL],
            tool_choice="required",
            max_tokens=256,
            temperature=0.0,
        )
        tool_calls = response.choices[0].message.tool_calls
        if tool_calls:
            args = json.loads(tool_calls[0].function.arguments)
            date_from = _parse_date_arg(args.get("date_from")) or fallback_from
            date_to = _parse_date_arg(args.get("date_to")) or fallback_to
            return date_from, date_to
    except Exception as exc:
        logger.warning("report_date_extraction_failed", error=str(exc))

    return fallback_from, fallback_to


def _parse_date_arg(value: str | None) -> date | None:
    if not value or value == "null":
        return None
    try:
        return date.fromisoformat(value)
    except (ValueError, TypeError):
        return None


async def _load_review_rows(
    db_session: AsyncSession,
    restaurant_id: int,
    date_from: date | None,
    date_to: date | None,
) -> list[ReviewChunkMeta]:
    stmt = select(ReviewChunkMeta).where(
        ReviewChunkMeta.restaurant_id == restaurant_id,
        ReviewChunkMeta.chunk_index == 0,
    )
    if date_from:
        dt_from = datetime(date_from.year, date_from.month, date_from.day, tzinfo=UTC)
        stmt = stmt.where(ReviewChunkMeta.review_date >= dt_from)
    if date_to:
        dt_to = datetime(date_to.year, date_to.month, date_to.day, 23, 59, 59, tzinfo=UTC)
        stmt = stmt.where(ReviewChunkMeta.review_date <= dt_to)
    result = await db_session.execute(stmt)
    return list(result.scalars().all())


async def _aggregate_entities(
    vector_store: BaseVectorStore,
    collection: str,
    restaurant_id: int,
    date_from: date | None = None,
    date_to: date | None = None,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Scroll Qdrant payloads and tally entity mentions by sentiment polarity.

    The date range is applied here too, so the top praised/complained aspects
    cover exactly the same reviews as the Postgres metrics above. Without it the
    entity tallies would silently span all time while the headline numbers were
    scoped to the selected period, and the two halves of the report would
    disagree. The Qdrant range is on the review's own timestamp (review_date_ts),
    matching the review_date filter in _load_review_rows; bounds are widened to
    full days the same way so the two stores select the same rows.
    """
    praised: dict[str, int] = {}
    complained: dict[str, int] = {}

    try:
        from src.services.vector.qdrant_store import QdrantStore

        if not isinstance(vector_store, QdrantStore):
            return [], []

        filters: dict[str, Any] = {"restaurant_id": restaurant_id}
        if date_from:
            filters["date_from"] = datetime(
                date_from.year, date_from.month, date_from.day, tzinfo=UTC
            ).timestamp()
        if date_to:
            filters["date_to"] = datetime(
                date_to.year, date_to.month, date_to.day, 23, 59, 59, tzinfo=UTC
            ).timestamp()
        qdrant_filter = vector_store._build_filter(filters)
        offset = None

        while True:
            results, next_offset = await vector_store.client.scroll(
                collection_name=collection,
                scroll_filter=qdrant_filter,
                limit=200,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in results:
                if not point.payload:
                    continue
                entities: list[str] = point.payload.get("food_entities", [])
                sentiment = point.payload.get("sentiment_label", "Neutral")
                for entity in entities:
                    if sentiment == "Positive":
                        praised[entity] = praised.get(entity, 0) + 1
                    elif sentiment == "Negative":
                        complained[entity] = complained.get(entity, 0) + 1
            if next_offset is None:
                break
            offset = next_offset
    except Exception as exc:
        logger.warning("entity_aggregation_failed", error=str(exc))

    top_praised = sorted(praised.items(), key=lambda x: x[1], reverse=True)[:10]
    top_complained = sorted(complained.items(), key=lambda x: x[1], reverse=True)[:10]
    return top_praised, top_complained


async def _generate_summary(
    openai_client: AsyncOpenAI,
    model: str,
    total_reviews: int,
    avg_rating: float | None,
    sentiment_breakdown: dict[str, int],
    top_praised: list[tuple[str, int]],
    top_complained: list[tuple[str, int]],
) -> str:
    praised_str = ", ".join(f"{e} ({n})" for e, n in top_praised[:5]) or "none identified"
    complained_str = ", ".join(f"{e} ({n})" for e, n in top_complained[:5]) or "none identified"
    rating_str = f"{avg_rating}/5" if avg_rating is not None else "not available"

    prompt = (
        f"Total reviews: {total_reviews}. Average rating: {rating_str}. "
        f"Sentiment: {sentiment_breakdown}. "
        f"Most praised: {praised_str}. Most complained about: {complained_str}. "
        "Write a 2-paragraph plain-English executive summary for a restaurant owner. "
        "Be direct and actionable. No bullet points. No technical jargon."
    )
    try:
        response = await openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You write clear, concise executive summaries for restaurant owners.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=400,
            temperature=0.3,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        logger.warning("report_summary_generation_failed", error=str(exc))
        return "Summary generation failed. Please review the data tables above."


def _build_markdown(
    restaurant_id: int,
    date_from: date | None,
    date_to: date | None,
    total_reviews: int,
    avg_rating: float | None,
    rating_distribution: dict[str, int],
    sentiment_breakdown: dict[str, int],
    source_breakdown: dict[str, int],
    top_praised: list[tuple[str, int]],
    top_complained: list[tuple[str, int]],
    summary: str,
) -> str:
    period = "All Time"
    if date_from and date_to:
        period = f"{date_from} to {date_to}"
    elif date_from:
        period = f"From {date_from}"
    elif date_to:
        period = f"Up to {date_to}"

    generated = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
    rating_str = f"{avg_rating}/5" if avg_rating is not None else "N/A"

    lines: list[str] = [
        f"# Insights Report (Restaurant {restaurant_id}) - {period}",
        f"Generated: {generated}",
        "",
        "## Overview",
        f"- **Total reviews:** {total_reviews}",
        f"- **Average rating:** {rating_str}",
        f"- **Period:** {period}",
        "",
        "## Sentiment Breakdown",
        "| Sentiment | Count |",
        "|-----------|-------|",
    ]
    for label, count in sorted(sentiment_breakdown.items()):
        lines.append(f"| {label} | {count} |")

    lines += [
        "",
        "## Rating Distribution",
        "| Stars | Count |",
        "|-------|-------|",
    ]
    for stars in sorted(rating_distribution.keys()):
        lines.append(f"| {stars} | {rating_distribution[stars]} |")

    lines += [
        "",
        "## Source Breakdown",
        "| Platform | Count |",
        "|----------|-------|",
    ]
    for src, count in sorted(source_breakdown.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"| {src} | {count} |")

    if top_praised:
        lines += ["", "## Top Praised Aspects"]
        for i, (entity, count) in enumerate(top_praised, start=1):
            lines.append(f"{i}. {entity} ({count} mentions)")

    if top_complained:
        lines += ["", "## Top Complaints"]
        for i, (entity, count) in enumerate(top_complained, start=1):
            lines.append(f"{i}. {entity} ({count} mentions)")

    lines += ["", "## Summary", summary]

    return "\n".join(lines)
