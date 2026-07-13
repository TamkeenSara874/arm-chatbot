from __future__ import annotations

import contextlib
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.db_entities import ReviewChunkMeta


@dataclass
class PeriodStats:
    count: int
    avg_rating: float | None
    sentiment_counts: dict[str, int]


async def compute_period_stats(
    db: AsyncSession,
    restaurant_id: int,
    date_from: str | None,
    date_to: str | None,
) -> PeriodStats:
    """Direct Postgres aggregation (count, avg rating, sentiment breakdown) for one date range.

    Shared by trend comparison (src/api/routes/chat.py) and anomaly detection
    (src/core/anomaly.py) -- exact numbers via SQL, not an LLM estimate over a
    sample of retrieved evidence.
    """
    base_filters = [
        ReviewChunkMeta.chunk_index == 0,
        ReviewChunkMeta.restaurant_id == restaurant_id,
    ]

    if date_from:
        with contextlib.suppress(ValueError):
            dt = datetime.fromisoformat(date_from).replace(tzinfo=UTC)
            base_filters.append(ReviewChunkMeta.review_date >= dt)
    if date_to:
        with contextlib.suppress(ValueError):
            dt = datetime.fromisoformat(date_to).replace(tzinfo=UTC)
            base_filters.append(ReviewChunkMeta.review_date <= dt)

    count_stmt = (
        select(func.count(), func.avg(ReviewChunkMeta.rating))
        .select_from(ReviewChunkMeta)
        .where(*base_filters)
    )
    count, avg_rating = (await db.execute(count_stmt)).one()

    sentiment_stmt = (
        select(ReviewChunkMeta.sentiment_label, func.count())
        .select_from(ReviewChunkMeta)
        .where(*base_filters)
        .group_by(ReviewChunkMeta.sentiment_label)
    )
    sentiment_counts = {
        (label or "Unknown"): cnt for label, cnt in (await db.execute(sentiment_stmt)).all()
    }

    return PeriodStats(
        count=count,
        avg_rating=round(avg_rating, 2) if avg_rating is not None else None,
        sentiment_counts=sentiment_counts,
    )


async def compute_theme_count(
    db: AsyncSession,
    restaurant_id: int,
    keywords: list[str],
    date_from: str | None = None,
    date_to: str | None = None,
) -> int:
    """Exact count of reviews whose full text contains any of the given keywords.

    For qualitative themes ("rude staff", "cold food") that have no dedicated
    database column, this counts across ALL reviews via full_review ILIKE --
    not just the top_k retrieved sample an aggregation question would
    otherwise be limited to. Still an approximation of the underlying theme,
    not a semantic match: it only catches reviews using one of these literal
    words/phrases, so it can undercount (e.g. a review saying "hostile" for a
    "rude" search won't match) -- but it never overstates precision beyond
    what was actually matched, and covers the full corpus rather than a
    20-review sample skewed by retrieval relevance.
    """
    if not keywords:
        return 0

    base_filters = [
        ReviewChunkMeta.chunk_index == 0,
        ReviewChunkMeta.restaurant_id == restaurant_id,
        ReviewChunkMeta.full_review.is_not(None),
    ]
    if date_from:
        with contextlib.suppress(ValueError):
            dt = datetime.fromisoformat(date_from).replace(tzinfo=UTC)
            base_filters.append(ReviewChunkMeta.review_date >= dt)
    if date_to:
        with contextlib.suppress(ValueError):
            dt = datetime.fromisoformat(date_to).replace(tzinfo=UTC)
            base_filters.append(ReviewChunkMeta.review_date <= dt)

    keyword_filter = or_(*(ReviewChunkMeta.full_review.ilike(f"%{kw}%") for kw in keywords))
    stmt = select(func.count()).select_from(ReviewChunkMeta).where(*base_filters, keyword_filter)
    return (await db.execute(stmt)).scalar_one()


async def compute_theme_cooccurrence_count(
    db: AsyncSession,
    restaurant_id: int,
    keywords_a: list[str],
    keywords_b: list[str],
    date_from: str | None = None,
    date_to: str | None = None,
) -> int:
    """Exact count of reviews whose full text contains BOTH theme A and theme
    B keywords together (an AND across the two groups, an OR within each).

    Distinct from compute_theme_count (a single theme, OR-matched) and a
    two-theme comparison (two independent counts) -- this is for "which
    reviews mention both slow service and cold food?" style questions, which
    need a real intersection, not a flat union of every keyword from both
    themes. Confirmed live as a real gap: with no AND-combination available,
    a "both X and Y" question got answered with an OR-matched count instead,
    directly contradicting the model's own qualitative read of the evidence
    (the exact number said 8, but no single retrieved review actually
    mentioned both themes at once).
    """
    if not keywords_a or not keywords_b:
        return 0

    base_filters = [
        ReviewChunkMeta.chunk_index == 0,
        ReviewChunkMeta.restaurant_id == restaurant_id,
        ReviewChunkMeta.full_review.is_not(None),
    ]
    if date_from:
        with contextlib.suppress(ValueError):
            dt = datetime.fromisoformat(date_from).replace(tzinfo=UTC)
            base_filters.append(ReviewChunkMeta.review_date >= dt)
    if date_to:
        with contextlib.suppress(ValueError):
            dt = datetime.fromisoformat(date_to).replace(tzinfo=UTC)
            base_filters.append(ReviewChunkMeta.review_date <= dt)

    filter_a = or_(*(ReviewChunkMeta.full_review.ilike(f"%{kw}%") for kw in keywords_a))
    filter_b = or_(*(ReviewChunkMeta.full_review.ilike(f"%{kw}%") for kw in keywords_b))
    stmt = (
        select(func.count()).select_from(ReviewChunkMeta).where(*base_filters, filter_a, filter_b)
    )
    return (await db.execute(stmt)).scalar_one()
