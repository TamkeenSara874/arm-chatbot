"""Unit tests for src/api/routes/chat.py's _persist_instant_exchange and the
message_id threading in _yield_cached.

Regression coverage for a real bug: the guardrail, conversation_recall, and
count_query fast paths all emit their response via _yield_instant() without
ever persisting the user/assistant turn to Postgres or Qdrant session_memory.
Confirmed live: a count-query answer and a guardrail decline both vanished
from chat history, and a later conversation_recall question had no awareness
either turn had happened -- both build_recent_turns_context (Postgres) and
build_session_context's cross-session search (Qdrant) only ever see
persisted rows. The same gap existed for both cache-hit tiers (_yield_cached),
fixed the same way.
"""

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.routes.chat import _persist_instant_exchange, _yield_cached


def _make_db() -> MagicMock:
    session_row = MagicMock()
    db = MagicMock()
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.get = AsyncMock(return_value=session_row)
    db.__aenter__ = AsyncMock(return_value=db)
    db.__aexit__ = AsyncMock(return_value=False)
    return db


def _patch_session_factory(db: MagicMock):
    factory = MagicMock(return_value=db)
    return patch("src.services.database.get_session_factory", return_value=factory)


class TestPersistInstantExchange:
    @pytest.mark.asyncio
    async def test_saves_user_and_assistant_messages(self) -> None:
        db = _make_db()
        embedder = MagicMock()
        vector_store = MagicMock()

        with (
            _patch_session_factory(db),
            patch("src.api.routes.chat.store_session_turn", new=AsyncMock()) as store_turn,
        ):
            await _persist_instant_exchange(
                session_id=uuid.uuid4(),
                message_id=uuid.uuid4(),
                sanitized="how many negative reviews do I have?",
                answer="You have 12 negative reviews in total.",
                model_used="direct_query",
                restaurant_id=1,
                embedder=embedder,
                vector_store=vector_store,
            )

        assert db.add.call_count == 2
        db.commit.assert_awaited()
        store_turn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stores_user_turn_in_session_memory(self) -> None:
        db = _make_db()
        embedder = MagicMock()
        vector_store = MagicMock()

        with (
            _patch_session_factory(db),
            patch("src.api.routes.chat.store_session_turn", new=AsyncMock()) as store_turn,
        ):
            await _persist_instant_exchange(
                session_id=uuid.uuid4(),
                message_id=uuid.uuid4(),
                sanitized="what did I ask before?",
                answer="You asked about negative reviews.",
                model_used="gpt-4o-mini",
                restaurant_id=1,
                embedder=embedder,
                vector_store=vector_store,
            )

        _, kwargs = store_turn.call_args
        assert kwargs["role"] == "user"
        assert kwargs["content"] == "what did I ask before?"

    @pytest.mark.asyncio
    async def test_swallows_errors_without_raising(self) -> None:
        db = _make_db()
        db.commit = AsyncMock(side_effect=RuntimeError("db down"))
        embedder = MagicMock()
        vector_store = MagicMock()

        with _patch_session_factory(db):
            # Must not raise -- this runs as a fire-and-forget background task.
            await _persist_instant_exchange(
                session_id=uuid.uuid4(),
                message_id=uuid.uuid4(),
                sanitized="how is the weather today",
                answer="I can only answer questions based on your restaurant's reviews.",
                model_used="guardrail",
                restaurant_id=1,
                embedder=embedder,
                vector_store=vector_store,
            )


class TestYieldCached:
    @pytest.mark.asyncio
    async def test_uses_the_passed_in_message_id(self) -> None:
        """Regression test: previously generated its own random message_id
        internally, decoupled from whatever id the caller persisted the turn
        under -- the SSE payload's message_id must match the persisted row."""
        message_id = uuid.uuid4()
        session_id = uuid.uuid4()
        data = {"answer": "You have 12 reviews.", "confidence": 1.0}

        events = [event async for event in _yield_cached(data, session_id, message_id)]

        assert len(events) == 1
        payload = json.loads(events[0]["data"])
        assert payload["message_id"] == str(message_id)
        assert payload["session_id"] == str(session_id)
