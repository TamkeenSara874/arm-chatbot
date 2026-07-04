from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.db_entities import ChatMessage, ChatSession
from src.services.embedding.base import BaseEmbedder
from src.services.llm.base import BaseLLMClient
from src.services.vector.base import BaseVectorStore
from src.utils.token_budget import enforce_token_budget

logger = structlog.get_logger()

SESSION_MEMORY_COLLECTION = "session_memory"


async def store_session_turn(
    session_id: uuid.UUID,
    role: str,
    content: str,
    embedder: BaseEmbedder,
    vector_store: BaseVectorStore,
) -> None:
    """Embed and upsert one message into the session_memory Qdrant collection.

    Called after each user message so future semantic lookups can surface it.
    Failures are swallowed so a Qdrant outage never breaks the chat response.
    """
    point_id = str(uuid.uuid4())
    try:
        vector = await embedder.embed_one(content)
        await vector_store.upsert(
            SESSION_MEMORY_COLLECTION,
            [
                {
                    "id": point_id,
                    "vector": vector,
                    "payload": {
                        "session_id": str(session_id),
                        "role": role,
                        "content": content,
                        "created_at_ts": int(datetime.now(tz=UTC).timestamp()),
                    },
                }
            ],
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
) -> str:
    """Build a cheap "last N turns" string for pronoun resolution at decomposition time.

    Unlike build_session_context, this skips the Qdrant ANN lookup and token-budget
    trimming -- decomposition only needs immediate continuity (e.g. resolving "that"
    or "it"), not the full context window used for answer generation.
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
    return "\n".join(lines)


async def build_session_context(
    session_id: uuid.UUID,
    current_query: str,
    db_session: AsyncSession,
    vector_store: BaseVectorStore,
    embedder: BaseEmbedder,
    recent_k: int = 5,
    relevant_k: int = 3,
    token_budget: int = 6000,
) -> str:
    """Build a combined context string for injection into the chat prompt.

    Combines three layers (in order of priority):
    1. A rolling LLM summary if one exists on the session row (covers distant history)
    2. The top relevant_k past turns semantically similar to the current query
    3. The last recent_k message pairs verbatim for immediate continuity

    The combined block is trimmed to token_budget before returning.
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
        query_vector = await embedder.embed_one(current_query)
        ann_results = await vector_store.search(
            SESSION_MEMORY_COLLECTION,
            query_vector,
            limit=relevant_k + recent_k,
            filters={"session_id": str(session_id)},
        )
        for ann_result in ann_results:
            content = ann_result.payload.get("content", "")
            role = ann_result.payload.get("role", "user")
            if content and content not in recent_contents:
                label = "User" if role == "user" else "Assistant"
                relevant_turns.append(f"{label}: {content}")
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
        lines = [
            f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}" for m in recent_messages
        ]
        parts.append("[Recent messages]\n" + "\n".join(lines))

    combined = "\n\n".join(parts)
    return enforce_token_budget(combined, max_tokens=token_budget)


async def maybe_trigger_summary(
    session_id: uuid.UUID,
    db_session: AsyncSession,
    llm_client: BaseLLMClient,
    summary_trigger: int = 50,
) -> None:
    """Check message count and fire a background summary task if the trigger is reached.

    The task runs as a fire-and-forget asyncio task so it never delays the response.
    """
    count_result = await db_session.execute(
        select(func.count()).where(ChatMessage.session_id == session_id)
    )
    message_count = count_result.scalar_one()

    if message_count < summary_trigger:
        return

    session_row = await db_session.get(ChatSession, session_id)
    if session_row and session_row.summary is not None:
        return

    asyncio.create_task(
        _generate_and_save_summary(session_id, llm_client),
        name=f"session-summary-{session_id}",
    )


async def _generate_and_save_summary(
    session_id: uuid.UUID,
    llm_client: BaseLLMClient,
) -> None:
    # Fire-and-forget (asyncio.create_task, not awaited): must not reuse the
    # caller's db_session. That session belongs to a request that may finish
    # and get torn down by FastAPI's Depends(get_db) cleanup before this task
    # completes -- same "This transaction is closed" race already found and
    # fixed in chat.py's _post_response_tasks. Opens its own session instead.
    from src.services.database import get_session_factory

    try:
        async with get_session_factory()() as db_session:
            stmt = (
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at)
            )
            result = await db_session.execute(stmt)
            messages = result.scalars().all()

            conversation = "\n".join(
                f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}" for m in messages
            )

            summary = await llm_client.complete(
                prompt=(
                    f"Conversation to summarize:\n{conversation}\n\n"
                    "Output a 2-3 sentence summary only. No preamble."
                ),
                system=(
                    "You are a conversation summarizer. Produce a dense, factual summary "
                    "in 2-3 sentences. Preserve all specific facts mentioned. Never editorialize."
                ),
                max_tokens=200,
                temperature=0.2,
            )

            session_row = await db_session.get(ChatSession, session_id)
            if session_row:
                session_row.summary = summary
                await db_session.commit()
                logger.info("session_summary_saved", session_id=str(session_id))
    except Exception as exc:
        logger.warning(
            "session_summary_failed",
            session_id=str(session_id),
            error=str(exc),
        )
