from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.db_entities import ChatCorrection
from src.services.embedding.base import BaseEmbedder
from src.services.vector.base import BaseVectorStore

logger = structlog.get_logger()

CORRECTION_COLLECTION = "correction_embeddings"
CONSENSUS_THRESHOLD = 3


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
) -> CorrectionMatch | None:
    """Search for a stored correction that matches the current query and context.

    Returns a CorrectionMatch (text + whether correction_count has reached
    CONSENSUS_THRESHOLD distinct flags) if a match is found above `threshold`
    with a compatible intent and matching restaurant. Returns None otherwise.

    is_consensus is the caller's signal for how much weight to give the
    correction: a single flag isn't yet confirmed and shouldn't override real
    review evidence the same way a multi-session consensus should -- see how
    chat.py routes this into either the `corrections` (ground truth) or
    `unverified_note` (informational only) generation-prompt field.

    Intent cross-check prevents a correction for one query type from being
    incorrectly applied to a different query type even if the text is similar.
    """
    try:
        vector = await embedder.embed_one(query)
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


async def store_correction(
    session_id: uuid.UUID | None,
    restaurant_id: int,
    original_query: str,
    original_response: str,
    corrected_response: str,
    intent: str,
    embedder: BaseEmbedder,
    vector_store: BaseVectorStore,
    db_session: AsyncSession,
    sim_threshold: float = 0.85,
) -> tuple[uuid.UUID, bool]:
    """Embed and persist a user correction.

    If a similar correction already exists (score >= sim_threshold), its count is
    incremented instead of creating a duplicate. Returns (correction_id, is_consensus)
    where is_consensus is True once correction_count reaches CONSENSUS_THRESHOLD.
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
        best = existing[0]
        new_count = best.payload.get("correction_count", 1) + 1
        is_consensus = new_count >= CONSENSUS_THRESHOLD
        await vector_store.update_payload(
            CORRECTION_COLLECTION,
            best.id,
            {
                "corrected_response": corrected_response,
                "correction_count": new_count,
                "is_consensus": is_consensus,
                # Refresh intent too -- otherwise a point created before the
                # caller started passing the real classified intent (e.g. an
                # old hardcoded placeholder) would keep failing
                # find_correction()'s intent cross-check forever.
                "intent": intent,
            },
        )
        correction_id = uuid.UUID(best.id)
        await _sync_meta(db_session, correction_id, new_count, is_consensus)
        return correction_id, is_consensus

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
                    "correction_count": 1,
                    "is_consensus": False,
                },
            }
        ],
    )

    row = ChatCorrection(
        id=correction_id,
        qdrant_point_id=str(correction_id),
        session_id=session_id,
        restaurant_id=restaurant_id,
        original_query=original_query,
        original_response=original_response,
        corrected_response=corrected_response,
        correction_count=1,
        is_consensus=False,
    )
    db_session.add(row)
    await db_session.commit()

    return correction_id, False


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
