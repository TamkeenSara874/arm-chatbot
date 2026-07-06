from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.review_stats import compute_period_stats
from src.services.cache import RedisCache

RECENT_WINDOW_DAYS = 7
MIN_REVIEWS_FOR_SIGNAL = 3
RATING_DROP_THRESHOLD = 0.5
NEGATIVE_SHARE_INCREASE_THRESHOLD = 0.25
ANOMALY_CACHE_TTL_SECONDS = 12 * 3600


@dataclass
class AnomalyResult:
    detected: bool
    recent_count: int
    baseline_count: int
    recent_avg_rating: float | None = None
    baseline_avg_rating: float | None = None
    recent_negative_share: float | None = None
    baseline_negative_share: float | None = None
    message: str | None = None


def _negative_share(count: int, sentiment_counts: dict[str, int]) -> float | None:
    if count == 0:
        return None
    return sentiment_counts.get("Negative", 0) / count


async def detect_anomaly(db: AsyncSession, restaurant_id: int) -> AnomalyResult:
    """Compare a trailing recent window against the prior baseline window of the
    same length, using exact Postgres aggregates (src.core.review_stats) --
    flags a rating drop or a rise in negative-review share large enough to be
    worth surfacing unprompted, not just ordinary evidence-level fluctuation.
    """
    # compute_period_stats' date bounds are both inclusive (>=/<=), so the two
    # windows must not share a boundary date -- otherwise reviews on that one
    # day get counted in both the recent and baseline windows at once.
    now = datetime.now(UTC)
    recent_from_date = (now - timedelta(days=RECENT_WINDOW_DAYS)).date()
    baseline_from_date = (now - timedelta(days=RECENT_WINDOW_DAYS * 2)).date()
    baseline_to_date = recent_from_date - timedelta(days=1)

    recent = await compute_period_stats(
        db, restaurant_id, recent_from_date.isoformat(), now.date().isoformat()
    )
    baseline = await compute_period_stats(
        db, restaurant_id, baseline_from_date.isoformat(), baseline_to_date.isoformat()
    )

    # Too little data on either side to trust a comparison -- a 1-review
    # baseline or recent window would make the threshold checks below
    # meaningless noise, not a real signal.
    if recent.count < MIN_REVIEWS_FOR_SIGNAL or baseline.count < MIN_REVIEWS_FOR_SIGNAL:
        return AnomalyResult(
            detected=False, recent_count=recent.count, baseline_count=baseline.count
        )

    rating_drop = None
    if recent.avg_rating is not None and baseline.avg_rating is not None:
        rating_drop = round(baseline.avg_rating - recent.avg_rating, 2)

    recent_neg = _negative_share(recent.count, recent.sentiment_counts)
    baseline_neg = _negative_share(baseline.count, baseline.sentiment_counts)
    neg_increase = None
    if recent_neg is not None and baseline_neg is not None:
        neg_increase = round(recent_neg - baseline_neg, 4)

    rating_flagged = rating_drop is not None and rating_drop >= RATING_DROP_THRESHOLD
    sentiment_flagged = (
        neg_increase is not None and neg_increase >= NEGATIVE_SHARE_INCREASE_THRESHOLD
    )
    detected = rating_flagged or sentiment_flagged

    message = None
    if detected:
        parts = []
        if rating_flagged:
            parts.append(
                f"average rating dropped from {baseline.avg_rating} to {recent.avg_rating} "
                f"over the past {RECENT_WINDOW_DAYS} days"
            )
        if sentiment_flagged:
            parts.append(
                f"negative reviews rose from {round(baseline_neg * 100)}% to "
                f"{round(recent_neg * 100)}% of total over the past {RECENT_WINDOW_DAYS} days"
            )
        message = "Heads up: " + " and ".join(parts) + "."

    return AnomalyResult(
        detected=detected,
        recent_count=recent.count,
        baseline_count=baseline.count,
        recent_avg_rating=recent.avg_rating,
        baseline_avg_rating=baseline.avg_rating,
        recent_negative_share=recent_neg,
        baseline_negative_share=baseline_neg,
        message=message,
    )


def _cache_key(restaurant_id: int) -> str:
    return f"anomaly:{restaurant_id}"


async def get_anomaly_status(
    db: AsyncSession, cache: RedisCache, restaurant_id: int
) -> AnomalyResult:
    """Cached wrapper around detect_anomaly.

    Review stats don't shift fast enough to warrant recomputing two Postgres
    aggregates on every poll of the alerts endpoint, so the result is cached
    with a TTL and recomputed lazily on expiry -- no separate scheduled job
    needed, matching this app's existing request/response-only architecture.
    """
    key = _cache_key(restaurant_id)
    cached = await cache.get_json(key)
    if cached is not None:
        return AnomalyResult(**cached)

    result = await detect_anomaly(db, restaurant_id)
    await cache.set_json(key, asdict(result), ANOMALY_CACHE_TTL_SECONDS)
    return result
