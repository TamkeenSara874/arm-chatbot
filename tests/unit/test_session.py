"""Unit tests for session context building."""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.session import build_session_context, store_session_turn
from src.utils.token_budget import estimate_tokens


def _make_message(role: str, content: str) -> MagicMock:
    msg = MagicMock()
    msg.role = role
    msg.content = content
    msg.session_id = uuid.uuid4()
    msg.created_at = datetime.now(tz=UTC)
    return msg


def _make_db_session(
    messages: list[MagicMock] | None = None,
    summary: str | None = None,
) -> MagicMock:
    session_row = MagicMock()
    session_row.summary = summary

    db = MagicMock()
    db.get = AsyncMock(return_value=session_row)

    scalars = MagicMock()
    scalars.all = MagicMock(return_value=messages or [])
    execute_result = MagicMock()
    execute_result.scalars = MagicMock(return_value=scalars)
    db.execute = AsyncMock(return_value=execute_result)
    return db


def _make_embedder(vector: list[float] | None = None) -> MagicMock:
    embedder = MagicMock()
    embedder.embed_one = AsyncMock(return_value=vector or [0.0] * 3072)
    return embedder


def _make_vector_store(ann_results=None) -> MagicMock:

    store = MagicMock()
    store.search = AsyncMock(return_value=ann_results or [])
    store.upsert = AsyncMock()
    return store


class TestBuildSessionContext:
    @pytest.mark.asyncio
    async def test_empty_history_returns_empty_string(self) -> None:
        db = _make_db_session(messages=[])
        embedder = _make_embedder()
        store = _make_vector_store()
        result = await build_session_context(
            session_id=uuid.uuid4(),
            current_query="What is best?",
            db_session=db,
            vector_store=store,
            embedder=embedder,
        )
        assert result == ""

    @pytest.mark.asyncio
    async def test_recent_messages_included(self) -> None:
        messages = [
            _make_message("user", "What is your best dish?"),
            _make_message("assistant", "The biryani is highly praised."),
        ]
        db = _make_db_session(messages=messages)
        embedder = _make_embedder()
        store = _make_vector_store()
        result = await build_session_context(
            session_id=uuid.uuid4(),
            current_query="Tell me more",
            db_session=db,
            vector_store=store,
            embedder=embedder,
        )
        assert "biryani is highly praised" in result

    @pytest.mark.asyncio
    async def test_summary_prepended_when_present(self) -> None:
        db = _make_db_session(summary="Earlier we discussed biryani and service issues.")
        embedder = _make_embedder()
        store = _make_vector_store()
        result = await build_session_context(
            session_id=uuid.uuid4(),
            current_query="Tell me more",
            db_session=db,
            vector_store=store,
            embedder=embedder,
        )
        assert "Earlier we discussed biryani" in result

    @pytest.mark.asyncio
    async def test_ann_results_deduplicated_with_recent_messages(self) -> None:
        from src.services.vector.base import SearchResult

        content = "The biryani was excellent."
        messages = [_make_message("user", content)]
        db = _make_db_session(messages=messages)
        embedder = _make_embedder()
        ann = [
            SearchResult(
                id="m1",
                score=0.95,
                payload={"role": "user", "content": content},
            )
        ]
        store = _make_vector_store(ann_results=ann)
        result = await build_session_context(
            session_id=uuid.uuid4(),
            current_query="more questions",
            db_session=db,
            vector_store=store,
            embedder=embedder,
        )
        count = result.count(content)
        assert count == 1, "Duplicate content from ANN should be deduplicated"

    @pytest.mark.asyncio
    async def test_token_budget_enforced(self) -> None:
        long_content = "very long content " * 500
        messages = [_make_message("user", long_content)]
        db = _make_db_session(messages=messages)
        embedder = _make_embedder()
        store = _make_vector_store()
        result = await build_session_context(
            session_id=uuid.uuid4(),
            current_query="query",
            db_session=db,
            vector_store=store,
            embedder=embedder,
            token_budget=200,
        )
        assert estimate_tokens(result) <= 200

    @pytest.mark.asyncio
    async def test_ann_failure_degrades_gracefully(self) -> None:
        messages = [_make_message("user", "What is the best food?")]
        db = _make_db_session(messages=messages)
        embedder = _make_embedder()
        store = MagicMock()
        store.search = AsyncMock(side_effect=RuntimeError("Qdrant down"))
        store.upsert = AsyncMock()

        result = await build_session_context(
            session_id=uuid.uuid4(),
            current_query="more?",
            db_session=db,
            vector_store=store,
            embedder=embedder,
        )
        assert "What is the best food?" in result


class TestStoreSessionTurn:
    @pytest.mark.asyncio
    async def test_successful_upsert(self) -> None:
        embedder = _make_embedder()
        store = _make_vector_store()
        await store_session_turn(
            session_id=uuid.uuid4(),
            role="user",
            content="Hello!",
            embedder=embedder,
            vector_store=store,
        )
        store.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_failure_does_not_raise(self) -> None:
        embedder = _make_embedder()
        store = MagicMock()
        store.upsert = AsyncMock(side_effect=RuntimeError("Qdrant unavailable"))
        await store_session_turn(
            session_id=uuid.uuid4(),
            role="user",
            content="Hello!",
            embedder=embedder,
            vector_store=store,
        )
