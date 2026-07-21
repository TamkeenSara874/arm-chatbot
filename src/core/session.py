from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.db_entities import ChatMessage, ChatSession
from src.services.embedding.base import BaseEmbedder
from src.services.llm.base import BaseLLMClient
from src.services.vector.base import BaseVectorStore
from src.utils.background import fire_and_forget
from src.utils.token_budget import enforce_token_budget

logger = structlog.get_logger()

SESSION_MEMORY_COLLECTION = "session_memory"

# build_session_context's relevant-turn search now spans this restaurant's
# entire session_memory history, not just the current session -- unlike the
# old session-scoped search (naturally bounded by how long one conversation
# runs), an unbounded restaurant-wide search could surface something many
# months old that's no longer representative. Cap how far back a
# cross-session match can come from.
MAX_CROSS_SESSION_AGE_DAYS = 90

# Below this, a recent-turn is still part of the active back-and-forth --
# no point annotating something from 5 minutes ago as "old."
_RECENT_TURN_STALE_THRESHOLD_MINUTES = 30


def _elapsed_note(created_at: datetime | None, now: datetime) -> str:
    """Inline " (N minutes/hours ago)" note for a same-session recent turn.

    The cross-session relevant-turn path above already labels a turn from a
    *different* session with how long ago it happened, specifically so the
    model can judge whether it's still current. The same-session "[Recent
    messages]" block had no equivalent -- confirmed live as a real bug: a
    session left open for over an hour had an old, unrelated exchange
    ("...how worried should I be?") blended into the answer to a brand new,
    unrelated question ("what the overall rating of my restaurant"), because
    nothing in the prompt signaled that turn was over an hour stale rather
    than the message right before it. Below the threshold, no note is added
    -- an actively continuing conversation doesn't need one.
    """
    if created_at is None:
        return ""
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    minutes_ago = (now - created_at).total_seconds() / 60
    if minutes_ago < _RECENT_TURN_STALE_THRESHOLD_MINUTES:
        return ""
    if minutes_ago < 120:
        return f" ({int(minutes_ago)} minutes ago)"
    hours_ago = minutes_ago / 60
    if hours_ago < 48:
        return f" ({int(hours_ago)} hours ago)"
    return f" ({int(hours_ago / 24)} days ago)"


async def store_session_turn(
    session_id: uuid.UUID,
    restaurant_id: int,
    role: str,
    content: str,
    embedder: BaseEmbedder,
    vector_store: BaseVectorStore,
    answer: str | None = None,
) -> None:
    """Embed and upsert one message into the session_memory Qdrant collection.

    Called after each user message so future semantic lookups can surface it --
    both within this session and, via build_session_context's restaurant_id
    filter, across this restaurant's other sessions too (cross-session memory).
    Failures are swallowed so a Qdrant outage never breaks the chat response.

    `answer` carries the assistant's reply to `content` into the payload. Only
    the user's question is embedded -- the reply just rides along. Storing the
    reply as its own point instead would have cost a second embedding call per
    turn, doubled the collection, and let long multi-topic replies outscore
    short intent-shaped questions for the fixed relevant_k slots. Pairing gets
    the same recall for free: without it, a fact the assistant stated was
    unreachable by semantic search at any distance once it fell out of the
    recent-messages window, because only questions were ever indexed.
    """
    point_id = str(uuid.uuid4())
    try:
        vector = await embedder.embed_one(content)
        payload: dict[str, object] = {
            "session_id": str(session_id),
            "restaurant_id": restaurant_id,
            "role": role,
            "content": content,
            "created_at_ts": int(datetime.now(tz=UTC).timestamp()),
        }
        if answer:
            payload["answer"] = answer
        await vector_store.upsert(
            SESSION_MEMORY_COLLECTION,
            [{"id": point_id, "vector": vector, "payload": payload}],
        )
    except Exception as exc:
        logger.warning(
            "session_memory_store_failed",
            session_id=str(session_id),
            role=role,
            error=str(exc),
        )


async def build_recent_turns_context(
    session_id: uuid.UUID,
    db_session: AsyncSession,
    recent_k: int = 2,
    token_budget: int = 800,
) -> str:
    """Build a cheap "last N turns" string for pronoun resolution at decomposition time.

    Unlike build_session_context, this skips the Qdrant ANN lookup -- decomposition
    only needs immediate continuity (e.g. resolving "that" or "it"), not the full
    context window used for answer generation. It still needs a token cap though:
    a long complex-tier answer (routinely several hundred tokens) sitting in the
    last couple of turns can otherwise blow up the decomposition prompt to tens of
    thousands of tokens -- confirmed live, where two long recent turns pushed a
    single decomposition call's prompt to 22k+ tokens, driving huge latency and,
    on that occasion, misclassifying a clearly out-of-scope question.
    """
    stmt = (
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(recent_k * 2)
    )
    result = await db_session.execute(stmt)
    recent_messages = list(reversed(result.scalars().all()))

    if not recent_messages:
        return ""

    lines = [f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}" for m in recent_messages]
    return enforce_token_budget("\n".join(lines), max_tokens=token_budget)


async def build_session_context(
    session_id: uuid.UUID,
    restaurant_id: int,
    current_query: str,
    db_session: AsyncSession,
    vector_store: BaseVectorStore,
    embedder: BaseEmbedder,
    recent_k: int = 5,
    relevant_k: int = 3,
    token_budget: int = 6000,
    precomputed_query_vector: list[float] | None = None,
) -> str:
    """Build a combined context string for injection into the chat prompt.

    Combines three layers (in order of priority):
    1. A rolling LLM summary if one exists on the session row (covers distant history)
    2. The top relevant_k past turns semantically similar to the current query --
       searched across this restaurant's ENTIRE session_memory history (filtered
       by restaurant_id, not session_id), so a relevant exchange from a past,
       separate conversation surfaces here too, not just turns from the current
       session. A turn from a different session is labeled with how long ago it
       happened so the model can judge whether it's likely still current, and
       anything older than MAX_CROSS_SESSION_AGE_DAYS is excluded outright.
    3. The last recent_k message pairs verbatim for immediate continuity

    The combined block is trimmed to token_budget before returning.

    precomputed_query_vector lets a caller that already embedded current_query
    (e.g. to also call find_correction() on the same text) skip a second,
    redundant embedding call.
    """
    session_row = await db_session.get(ChatSession, session_id)
    summary = session_row.summary if session_row else None

    stmt = (
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(recent_k * 2)
    )
    result = await db_session.execute(stmt)
    recent_messages = list(reversed(result.scalars().all()))

    relevant_turns: list[str] = []
    recent_contents: set[str] = {m.content for m in recent_messages}

    try:
        query_vector = (
            precomputed_query_vector
            if precomputed_query_vector is not None
            else await embedder.embed_one(current_query)
        )
        ann_results = await vector_store.search(
            SESSION_MEMORY_COLLECTION,
            query_vector,
            limit=relevant_k + recent_k,
            filters={"restaurant_id": restaurant_id},
        )
        now_ts = datetime.now(tz=UTC).timestamp()
        for ann_result in ann_results:
            content = ann_result.payload.get("content", "")
            role = ann_result.payload.get("role", "user")
            if not content or content in recent_contents:
                continue

            age_note = ""
            if ann_result.payload.get("session_id") != str(session_id):
                created_at_ts = ann_result.payload.get("created_at_ts")
                if not created_at_ts:
                    continue
                days_ago = (now_ts - created_at_ts) / 86400
                if days_ago > MAX_CROSS_SESSION_AGE_DAYS:
                    continue
                if days_ago >= 1:
                    age_note = f" (from a past conversation, {int(days_ago)} day(s) ago)"

            label = "User" if role == "user" else "Assistant"
            turn = f"{label}: {content}{age_note}"

            # The paired reply is rendered as something the assistant said
            # *previously*, not as evidence. Without that framing the model
            # treats its own prior answer as an established fact and will
            # restate a figure that new ingestion has since changed.
            answer = ann_result.payload.get("answer")
            if answer:
                turn += f"\nAssistant previously answered: {answer}"
            relevant_turns.append(turn)
            if len(relevant_turns) >= relevant_k:
                break
    except Exception as exc:
        logger.warning(
            "session_memory_ann_failed",
            session_id=str(session_id),
            error=str(exc),
        )

    parts: list[str] = []

    if summary:
        parts.append(f"[Summary of earlier conversation]\n{summary}")

    if relevant_turns:
        parts.append("[Relevant past exchanges]\n" + "\n".join(relevant_turns))

    if recent_messages:
        now = datetime.now(tz=UTC)
        lines = [
            f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}"
            f"{_elapsed_note(m.created_at, now)}"
            for m in recent_messages
        ]
        parts.append("[Recent messages]\n" + "\n".join(lines))

    combined = "\n\n".join(parts)
    return enforce_token_budget(combined, max_tokens=token_budget)


async def purge_expired_sessions(
    db_session: AsyncSession,
    vector_store: BaseVectorStore,
    ttl_days: int,
) -> int:
    """Delete sessions idle for longer than ttl_days, from Postgres and Qdrant.

    SESSION_TTL_DAYS has been a setting since the first migration but nothing
    ever enforced it, so chat_session, chat_message and the session_memory
    collection all grew without bound. session_memory is the expensive one:
    every point is a 3072-dim vector Qdrant holds in RAM.

    Postgres first, then Qdrant. If the process dies between the two, the
    leftover Qdrant points are unreferenced but harmless, and the next sweep
    removes them anyway because the cutoff is absolute rather than relative to
    what Postgres still holds. Doing it the other way round would leave
    sessions whose memory had been deleted underneath them.

    Returns the number of sessions deleted.
    """
    cutoff = datetime.now(tz=UTC) - timedelta(days=ttl_days)

    # chat_message rows go with them via ON DELETE CASCADE on the FK.
    result = await db_session.execute(
        delete(ChatSession).where(ChatSession.last_activity_at < cutoff)
    )
    deleted = result.rowcount or 0
    await db_session.commit()

    try:
        await vector_store.delete_by_filter(
            SESSION_MEMORY_COLLECTION,
            {"created_before": int(cutoff.timestamp())},
        )
    except Exception as exc:
        # Postgres is already committed. Log and let the next sweep retry
        # rather than failing the whole purge and losing that progress.
        logger.warning("session_memory_purge_failed", error=str(exc))

    if deleted:
        logger.info("expired_sessions_purged", count=deleted, ttl_days=ttl_days)
    return deleted


async def maybe_trigger_summary(
    session_id: uuid.UUID,
    db_session: AsyncSession,
    llm_client: BaseLLMClient,
    summary_trigger: int = 20,
    refresh_every: int = 20,
) -> None:
    """Fire a background summary task if the session is due for one.

    Due means either "never summarized and past the trigger", or "summarized,
    but refresh_every more messages have accumulated since". The previous
    version returned early whenever a summary already existed, so it ran
    exactly once per session and then froze -- a conversation that reached 200
    messages still carried a summary of its first 50, and nothing covered the
    other 150.

    The task runs as a fire-and-forget asyncio task so it never delays the response.
    """
    count_result = await db_session.execute(
        select(func.count()).where(ChatMessage.session_id == session_id)
    )
    message_count = count_result.scalar_one()

    if message_count < summary_trigger:
        return

    session_row = await db_session.get(ChatSession, session_id)
    covered = session_row.summary_message_count if session_row else None

    # covered is NULL for a session written by the old one-shot path, which
    # never recorded its coverage. Treating that as 0 re-summarizes from the
    # start once, after which the count is accurate.
    if covered is not None and message_count - covered < refresh_every:
        return

    fire_and_forget(
        _generate_and_save_summary(session_id, llm_client),
        name=f"session-summary-{session_id}",
    )


async def _generate_and_save_summary(
    session_id: uuid.UUID,
    llm_client: BaseLLMClient,
) -> None:
    # Fire-and-forget (fire_and_forget(), not awaited): must not reuse the
    # caller's db_session. That session belongs to a request that may finish
    # and get torn down by FastAPI's Depends(get_db) cleanup before this task
    # completes -- same "This transaction is closed" race already found and
    # fixed in chat.py's _post_response_tasks. Opens its own session instead.
    from src.services.database import get_session_factory

    try:
        async with get_session_factory()() as db_session:
            session_row = await db_session.get(ChatSession, session_id)
            if session_row is None:
                return

            previous_summary = session_row.summary
            covered = session_row.summary_message_count or 0

            # Only the messages the existing summary does not already cover.
            # Re-reading the whole conversation each refresh would make every
            # summary call more expensive than the last -- at 200 messages that
            # is a ~40k-token prompt, several times an hour, forever.
            stmt = (
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at)
                .offset(covered)
            )
            result = await db_session.execute(stmt)
            new_messages = result.scalars().all()
            if not new_messages:
                return

            conversation = "\n".join(
                f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}" for m in new_messages
            )

            if previous_summary and covered:
                prompt = (
                    f"Summary so far:\n{previous_summary}\n\n"
                    f"New messages since that summary:\n{conversation}\n\n"
                    "Output a single 2-3 sentence summary covering both. No preamble."
                )
            else:
                prompt = (
                    f"Conversation to summarize:\n{conversation}\n\n"
                    "Output a 2-3 sentence summary only. No preamble."
                )

            summary = await llm_client.complete(
                prompt=prompt,
                system=(
                    "You are a conversation summarizer. Produce a dense, factual summary "
                    "in 2-3 sentences. Preserve all specific facts mentioned. Never editorialize."
                ),
                max_tokens=200,
                temperature=0.2,
            )

            # Re-read rather than reusing the row fetched above: this task is
            # fire-and-forget and the conversation may have advanced while the
            # LLM call was in flight. Recording covered + len(new_messages)
            # (not the live count) keeps the marker honest about what the
            # summary text actually covers.
            session_row = await db_session.get(ChatSession, session_id)
            if session_row:
                session_row.summary = summary
                session_row.summary_message_count = covered + len(new_messages)
                await db_session.commit()
                logger.info(
                    "session_summary_saved",
                    session_id=str(session_id),
                    covers_messages=covered + len(new_messages),
                    incremental=bool(previous_summary and covered),
                )
    except Exception as exc:
        logger.warning(
            "session_summary_failed",
            session_id=str(session_id),
            error=str(exc),
        )
