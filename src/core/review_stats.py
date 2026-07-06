from __future__ import annotations

import contextlib
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import func, select
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
