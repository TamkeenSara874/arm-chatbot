from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.review_stats import compute_period_stats
from src.models.db_entities import ChatCorrection, ChatCorrectionVote
from src.services.embedding.base import BaseEmbedder
from src.services.vector.base import BaseVectorStore

logger = structlog.get_logger()

CORRECTION_COLLECTION = "correction_embeddings"
CONSENSUS_THRESHOLD = 3
# Distinct-session votes must also span real time, not just real sessions --
# otherwise an attacker who can freely create sessions (no signup exists)
# still only needs to script 3 of them back to back. An hour is short enough
# for a genuine, actively-used correction to still clear it within a normal
# day of traffic, long enough that scripting around it is conspicuous rather
# than instant.
CONSENSUS_MIN_SPAN_SECONDS = 3600.0
# A session that has submitted *any* correction recently is asked to wait --
# cheap, real friction against rapid-fire scripted submission that a purely
# distinct-session count doesn't address on its own (nothing stops one
# attacker from holding several sessions open at once).
SUBMISSION_COOLDOWN_SECONDS = 60.0
# How far a claimed rating/review-count in a correction's own text may
# diverge from the real, exactly-computed value before it's rejected outright
# rather than trusted. Loose enough to tolerate a stale-by-a-few-reviews
# claim, tight enough to catch "we have a perfect 5-star rating" against a
# real 3.97.
RATING_CONTRADICTION_TOLERANCE = 0.5
COUNT_CONTRADICTION_RELATIVE_TOLERANCE = 0.15

_RATING_CLAIM_RE = re.compile(
    r"(\d(?:\.\d)?)\s*(?:-|\s)?\s*(?:star|stars|/\s*5|out of 5)", re.IGNORECASE
)
_COUNT_CLAIM_RE = re.compile(r"([\d,]+)\s*reviews?\b", re.IGNORECASE)


@dataclass
class CorrectionMatch:
    text: str
    is_consensus: bool


async def find_correction(
    query: str,
    restaurant_id: int,
    intent: str,
    embedder: BaseEmbedder,
    vector_store: BaseVectorStore,
    threshold: float = 0.85,
    precomputed_vector: list[float] | None = None,
) -> CorrectionMatch | None:
    """Search for a stored correction that matches the current query and context.

    Returns a CorrectionMatch (text + whether correction_count has reached
    CONSENSUS_THRESHOLD distinct sessions, spaced out over real time) if a
    match is found above `threshold` with a compatible intent and matching
    restaurant. Returns None otherwise.

    is_consensus is the caller's signal for how much weight to give the
    correction: a single flag isn't yet confirmed and shouldn't override real
    review evidence the same way a multi-session consensus should -- see how
    chat.py routes this into either the `corrections` (ground truth) or
    `unverified_note` (informational only) generation-prompt field.

    Intent cross-check prevents a correction for one query type from being
    incorrectly applied to a different query type even if the text is similar.

    precomputed_vector lets a caller that already embedded this exact query
    text (e.g. to also call build_session_context() on the same text) skip a
    second, redundant embedding call.

    A rejected correction never surfaces here -- the reject endpoint deletes
    its Qdrant point, so this search simply can't find it anymore.
    """
    try:
        vector = (
            precomputed_vector
            if precomputed_vector is not None
            else await embedder.embed_one(query)
        )
        results = await vector_store.search(
            CORRECTION_COLLECTION,
            vector,
            limit=5,
            score_threshold=threshold,
            filters={"restaurant_id": restaurant_id},
        )
    except Exception as exc:
        logger.warning("correction_lookup_failed", error=str(exc))
        return None

    for result in results:
        p = result.payload
        stored_intent = p.get("intent")
        if stored_intent and stored_intent != intent:
            logger.debug(
                "correction_intent_mismatch",
                stored_intent=stored_intent,
                current_intent=intent,
                score=result.score,
            )
            continue
        corrected_response = p.get("corrected_response")
        if not corrected_response:
            continue
        return CorrectionMatch(
            text=corrected_response, is_consensus=bool(p.get("is_consensus", False))
        )

    return None


def scan_for_stat_contradiction(
    text: str, real_avg_rating: float | None, real_count: int
) -> str | None:
    """Return a rejection reason if text claims a rating/review-count that
    contradicts the real, exactly-computed value -- or None if the text
    makes no such claim, or the claim is close enough to be plausible.

    Deliberately narrow: only the two claim types this codebase can verify
    exactly (overall rating, total review count) are checked. A qualitative
    correction ("the wait-staff issue was fixed last month") isn't
    fact-checkable this way and is left to the other guardrails instead of
    risking false positives here.
    """
    rating_match = _RATING_CLAIM_RE.search(text)
    if rating_match and real_avg_rating is not None:
        claimed = float(rating_match.group(1))
        if abs(claimed - real_avg_rating) > RATING_CONTRADICTION_TOLERANCE:
            return (
                f"Claims a rating of {claimed}, but the real average is "
                f"{real_avg_rating:.2f} -- correction rejected."
            )

    count_match = _COUNT_CLAIM_RE.search(text)
    if count_match:
        claimed_count = int(count_match.group(1).replace(",", ""))
        tolerance = max(1, real_count * COUNT_CONTRADICTION_RELATIVE_TOLERANCE)
        if abs(claimed_count - real_count) > tolerance:
            return (
                f"Claims {claimed_count} reviews, but the real count is "
                f"{real_count} -- correction rejected."
            )

    return None


async def check_stat_contradiction(text: str, db: AsyncSession, restaurant_id: int) -> str | None:
    """DB-backed wrapper around scan_for_stat_contradiction -- computes the
    real stats to check text's claims against."""
    stats = await compute_period_stats(db, restaurant_id, date_from=None, date_to=None)
    return scan_for_stat_contradiction(text, stats.avg_rating, stats.count)


async def _record_vote(
    db_session: AsyncSession,
    correction_id: uuid.UUID,
    session_id: uuid.UUID,
    ip_address: str | None,
) -> bool:
    """Record this session's vote for correction_id. Returns True if this
    session hadn't voted for it before (a genuinely new, distinct
    corroboration), False if it had (the unique constraint on
    (correction_id, session_id) is what's actually enforcing this, not this
    function's own logic -- a race between two requests from the same
    session still can't double-count)."""
    vote = ChatCorrectionVote(
        correction_id=correction_id, session_id=session_id, ip_address=ip_address
    )
    db_session.add(vote)
    try:
        await db_session.commit()
        return True
    except IntegrityError:
        await db_session.rollback()
        return False


async def _vote_stats(db_session: AsyncSession, correction_id: uuid.UUID) -> tuple[int, float]:
    """(distinct session count, seconds between earliest and latest vote)."""
    stmt = select(
        func.count(func.distinct(ChatCorrectionVote.session_id)),
        func.min(ChatCorrectionVote.created_at),
        func.max(ChatCorrectionVote.created_at),
    ).where(ChatCorrectionVote.correction_id == correction_id)
    count, earliest, latest = (await db_session.execute(stmt)).one()
    if count == 0 or earliest is None or latest is None:
        return 0, 0.0
    return count, (latest - earliest).total_seconds()


async def session_in_cooldown(db_session: AsyncSession, session_id: uuid.UUID) -> bool:
    """True if this session submitted any correction within the last
    SUBMISSION_COOLDOWN_SECONDS -- cheap friction against rapid-fire
    submission that a distinct-session count alone doesn't stop (nothing
    prevents one attacker from holding several sessions open at once, but
    it does stop the trivial single-session and tight-loop-of-sessions cases)."""
    cutoff = datetime.now(tz=UTC) - timedelta(seconds=SUBMISSION_COOLDOWN_SECONDS)
    stmt = (
        select(ChatCorrectionVote.id)
        .where(ChatCorrectionVote.session_id == session_id)
        .where(ChatCorrectionVote.created_at >= cutoff)
        .limit(1)
    )
    result = await db_session.execute(stmt)
    return result.scalar_one_or_none() is not None


async def store_correction(
    session_id: uuid.UUID,
    restaurant_id: int,
    original_query: str,
    original_response: str,
    corrected_response: str,
    intent: str,
    embedder: BaseEmbedder,
    vector_store: BaseVectorStore,
    db_session: AsyncSession,
    ip_address: str | None = None,
    sim_threshold: float = 0.85,
) -> tuple[uuid.UUID, bool]:
    """Embed and persist a user correction.

    If a similar correction already exists (score >= sim_threshold), this
    session's vote is recorded against it (see _record_vote) instead of
    creating a duplicate cluster. correction_count/is_consensus are derived
    from COUNT(DISTINCT session_id) in chat_correction_vote, spaced over
    CONSENSUS_MIN_SPAN_SECONDS -- not incremented on trust. Only a genuinely
    new distinct-session vote updates the stored corrected_response text, so
    a session that already voted can't quietly swap in different wording
    while riding on other sessions' vote count.

    Returns (correction_id, is_consensus).
    """
    vector = await embedder.embed_one(original_query)

    existing = await vector_store.search(
        CORRECTION_COLLECTION,
        vector,
        limit=3,
        score_threshold=sim_threshold,
        filters={"restaurant_id": restaurant_id},
    )

    if existing:
        correction_id = uuid.UUID(existing[0].id)
    else:
        correction_id = uuid.uuid4()
        await vector_store.upsert(
            CORRECTION_COLLECTION,
            [
                {
                    "id": str(correction_id),
                    "vector": vector,
                    "payload": {
                        "restaurant_id": restaurant_id,
                        "original_query": original_query,
                        "corrected_response": corrected_response,
                        "intent": intent,
                        "correction_count": 0,
                        "is_consensus": False,
                    },
                }
            ],
        )
        db_session.add(
            ChatCorrection(
                id=correction_id,
                qdrant_point_id=str(correction_id),
                session_id=session_id,
                restaurant_id=restaurant_id,
                original_query=original_query,
                original_response=original_response,
                corrected_response=corrected_response,
                correction_count=0,
                is_consensus=False,
            )
        )
        await db_session.commit()

    vote_is_new = await _record_vote(db_session, correction_id, session_id, ip_address)

    if vote_is_new:
        await vector_store.update_payload(
            CORRECTION_COLLECTION,
            str(correction_id),
            {"corrected_response": corrected_response, "intent": intent},
        )
        row = await db_session.get(ChatCorrection, correction_id)
        if row:
            row.corrected_response = corrected_response

    distinct_sessions, span_seconds = await _vote_stats(db_session, correction_id)
    is_consensus = (
        distinct_sessions >= CONSENSUS_THRESHOLD and span_seconds >= CONSENSUS_MIN_SPAN_SECONDS
    )

    await vector_store.update_payload(
        CORRECTION_COLLECTION,
        str(correction_id),
        {"correction_count": distinct_sessions, "is_consensus": is_consensus},
    )
    await _sync_meta(db_session, correction_id, distinct_sessions, is_consensus)

    return correction_id, is_consensus


async def reject_correction(
    db_session: AsyncSession,
    vector_store: BaseVectorStore,
    correction_id: uuid.UUID,
    restaurant_id: int,
) -> bool:
    """Admin undo: deletes the Qdrant point (so find_correction() can never
    surface it again) and marks the Postgres row + vote history rejected
    (kept, not deleted, for audit). Returns False if no such correction
    exists for this restaurant."""
    row = await db_session.get(ChatCorrection, correction_id)
    if row is None or row.restaurant_id != restaurant_id:
        return False

    await vector_store.delete(CORRECTION_COLLECTION, [str(correction_id)])
    row.is_rejected = True
    row.is_consensus = False
    row.updated_at = datetime.now(tz=UTC)
    await db_session.commit()
    return True


async def _sync_meta(
    db_session: AsyncSession,
    correction_id: uuid.UUID,
    count: int,
    is_consensus: bool,
) -> None:
    row = await db_session.get(ChatCorrection, correction_id)
    if row:
        row.correction_count = count
        row.is_consensus = is_consensus
        row.updated_at = datetime.now(tz=UTC)
        await db_session.commit()
