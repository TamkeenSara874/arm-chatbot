"""Unit tests for session context building."""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.session import (
    _generate_and_save_summary,
    build_recent_turns_context,
    build_session_context,
    maybe_trigger_summary,
    store_session_turn,
)
from src.utils.token_budget import estimate_tokens


def _make_message(role: str, content: str, created_at: datetime | None = None) -> MagicMock:
    msg = MagicMock()
    msg.role = role
    msg.content = content
    msg.session_id = uuid.uuid4()
    msg.created_at = created_at or datetime.now(tz=UTC)
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


class TestBuildRecentTurnsContext:
    """Regression coverage for a real bug: this context previously had no
    token cap, so a couple of long complex-tier answers in the last few
    turns could balloon a single decomposition call to tens of thousands of
    tokens -- confirmed live at 22k+ tokens, which drove huge latency and
    misclassified a clearly out-of-scope question."""

    @pytest.mark.asyncio
    async def test_empty_history_returns_empty_string(self) -> None:
        db = _make_db_session(messages=[])
        result = await build_recent_turns_context(session_id=uuid.uuid4(), db_session=db)
        assert result == ""

    @pytest.mark.asyncio
    async def test_includes_recent_messages(self) -> None:
        messages = [
            _make_message("user", "What is your best dish?"),
            _make_message("assistant", "The biryani is highly praised."),
        ]
        db = _make_db_session(messages=messages)
        result = await build_recent_turns_context(session_id=uuid.uuid4(), db_session=db)
        assert "What is your best dish?" in result
        assert "The biryani is highly praised." in result

    @pytest.mark.asyncio
    async def test_long_turns_are_capped_to_token_budget(self) -> None:
        long_answer = "word " * 5000  # far more than any reasonable token budget
        messages = [
            _make_message("user", "How can I improve?"),
            _make_message("assistant", long_answer),
        ]
        db = _make_db_session(messages=messages)
        result = await build_recent_turns_context(
            session_id=uuid.uuid4(), db_session=db, token_budget=100
        )
        assert estimate_tokens(result) <= 100


class TestBuildSessionContext:
    @pytest.mark.asyncio
    async def test_empty_history_returns_empty_string(self) -> None:
        db = _make_db_session(messages=[])
        embedder = _make_embedder()
        store = _make_vector_store()
        result = await build_session_context(
            session_id=uuid.uuid4(),
            restaurant_id=1,
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
            restaurant_id=1,
            current_query="Tell me more",
            db_session=db,
            vector_store=store,
            embedder=embedder,
        )
        assert "biryani is highly praised" in result

    @pytest.mark.asyncio
    async def test_fresh_recent_message_has_no_elapsed_note(self) -> None:
        messages = [_make_message("user", "What is your best dish?")]
        db = _make_db_session(messages=messages)
        embedder = _make_embedder()
        store = _make_vector_store()
        result = await build_session_context(
            session_id=uuid.uuid4(),
            restaurant_id=1,
            current_query="Tell me more",
            db_session=db,
            vector_store=store,
            embedder=embedder,
        )
        assert "ago)" not in result

    @pytest.mark.asyncio
    async def test_stale_same_session_message_gets_elapsed_note(self) -> None:
        # Regression test: a real live bug had a same-session turn from over
        # an hour earlier ("...how worried should I be?") blended into the
        # answer to a brand new, unrelated question ("what the overall rating
        # of my restaurant"), because nothing signaled that turn was stale --
        # only cross-session turns got an age label. An hour-old turn in the
        # SAME session now gets one too.
        from datetime import timedelta

        old_time = datetime.now(tz=UTC) - timedelta(hours=2)
        messages = [_make_message("user", "How worried should I be?", created_at=old_time)]
        db = _make_db_session(messages=messages)
        embedder = _make_embedder()
        store = _make_vector_store()
        result = await build_session_context(
            session_id=uuid.uuid4(),
            restaurant_id=1,
            current_query="What is my overall rating?",
            db_session=db,
            vector_store=store,
            embedder=embedder,
        )
        assert "hours ago)" in result

    @pytest.mark.asyncio
    async def test_summary_prepended_when_present(self) -> None:
        db = _make_db_session(summary="Earlier we discussed biryani and service issues.")
        embedder = _make_embedder()
        store = _make_vector_store()
        result = await build_session_context(
            session_id=uuid.uuid4(),
            restaurant_id=1,
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
            restaurant_id=1,
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
            restaurant_id=1,
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
            restaurant_id=1,
            current_query="more?",
            db_session=db,
            vector_store=store,
            embedder=embedder,
        )
        assert "What is the best food?" in result


class TestAnnNotInRecent:
    @pytest.mark.asyncio
    async def test_ann_result_not_in_recent_appears_in_context(self) -> None:
        """ANN results that are NOT in the recent window should appear as relevant turns."""
        from src.services.vector.base import SearchResult

        old_content = "An old question about pasta from weeks ago."
        recent_content = "What is the best dish?"
        session_id = uuid.uuid4()

        messages = [_make_message("user", recent_content)]
        db = _make_db_session(messages=messages)
        embedder = _make_embedder()
        ann = [
            SearchResult(
                id="old1",
                score=0.91,
                # Same session_id as the call below -- this test is about
                # same-session ANN surfacing, not the cross-session age
                # filtering build_session_context also does now.
                payload={"role": "user", "content": old_content, "session_id": str(session_id)},
            )
        ]
        store = _make_vector_store(ann_results=ann)

        result = await build_session_context(
            session_id=session_id,
            restaurant_id=1,
            current_query="Tell me about pasta",
            db_session=db,
            vector_store=store,
            embedder=embedder,
        )

        assert old_content in result, "Old ANN content not in recent window should be included"

    @pytest.mark.asyncio
    async def test_ann_relevant_section_label_present(self) -> None:
        """When ANN surfaces non-recent content, the relevant section header appears."""
        from src.services.vector.base import SearchResult

        old_content = "I asked about the service last week."
        session_id = uuid.uuid4()
        messages = [_make_message("user", "current question")]
        db = _make_db_session(messages=messages)
        embedder = _make_embedder()
        ann = [
            SearchResult(
                id="old2",
                score=0.88,
                payload={"role": "user", "content": old_content, "session_id": str(session_id)},
            )
        ]
        store = _make_vector_store(ann_results=ann)

        result = await build_session_context(
            session_id=session_id,
            restaurant_id=1,
            current_query="service question",
            db_session=db,
            vector_store=store,
            embedder=embedder,
        )

        assert "Relevant past exchanges" in result


class TestCrossSessionMemory:
    """Coverage for widening build_session_context's ANN search from

    session_id-only to restaurant_id (so a relevant exchange from a
    different, past session surfaces too), plus the age labeling and
    staleness cutoff that come with searching beyond one session's lifetime.
    """

    @pytest.mark.asyncio
    async def test_recent_cross_session_turn_gets_age_label(self) -> None:
        from src.services.vector.base import SearchResult

        other_session_id = uuid.uuid4()
        content = "Last time I asked, the ambiance was described as charming."
        five_days_ago = datetime.now(tz=UTC).timestamp() - 5 * 86400

        messages = [_make_message("user", "current question")]
        db = _make_db_session(messages=messages)
        embedder = _make_embedder()
        ann = [
            SearchResult(
                id="cross1",
                score=0.9,
                payload={
                    "role": "user",
                    "content": content,
                    "session_id": str(other_session_id),
                    "created_at_ts": five_days_ago,
                },
            )
        ]
        store = _make_vector_store(ann_results=ann)

        result = await build_session_context(
            session_id=uuid.uuid4(),
            restaurant_id=1,
            current_query="how is the ambiance?",
            db_session=db,
            vector_store=store,
            embedder=embedder,
        )

        assert content in result
        assert "5 day(s) ago" in result

    @pytest.mark.asyncio
    async def test_cross_session_turn_older_than_cutoff_is_excluded(self) -> None:
        from src.core.session import MAX_CROSS_SESSION_AGE_DAYS
        from src.services.vector.base import SearchResult

        other_session_id = uuid.uuid4()
        content = "Ancient conversation about the menu."
        too_old = datetime.now(tz=UTC).timestamp() - (MAX_CROSS_SESSION_AGE_DAYS + 1) * 86400

        messages = [_make_message("user", "current question")]
        db = _make_db_session(messages=messages)
        embedder = _make_embedder()
        ann = [
            SearchResult(
                id="cross2",
                score=0.9,
                payload={
                    "role": "user",
                    "content": content,
                    "session_id": str(other_session_id),
                    "created_at_ts": too_old,
                },
            )
        ]
        store = _make_vector_store(ann_results=ann)

        result = await build_session_context(
            session_id=uuid.uuid4(),
            restaurant_id=1,
            current_query="what's on the menu?",
            db_session=db,
            vector_store=store,
            embedder=embedder,
        )

        assert content not in result

    @pytest.mark.asyncio
    async def test_cross_session_turn_missing_timestamp_is_excluded(self) -> None:
        from src.services.vector.base import SearchResult

        content = "A turn with no timestamp at all."
        messages = [_make_message("user", "current question")]
        db = _make_db_session(messages=messages)
        embedder = _make_embedder()
        ann = [
            SearchResult(
                id="cross3",
                score=0.9,
                payload={
                    "role": "user",
                    "content": content,
                    "session_id": str(uuid.uuid4()),
                },
            )
        ]
        store = _make_vector_store(ann_results=ann)

        result = await build_session_context(
            session_id=uuid.uuid4(),
            restaurant_id=1,
            current_query="query",
            db_session=db,
            vector_store=store,
            embedder=embedder,
        )

        assert content not in result

    @pytest.mark.asyncio
    async def test_search_filters_by_restaurant_id_not_session_id(self) -> None:
        embedder = _make_embedder()
        store = _make_vector_store(ann_results=[])
        db = _make_db_session(messages=[])

        await build_session_context(
            session_id=uuid.uuid4(),
            restaurant_id=42,
            current_query="query",
            db_session=db,
            vector_store=store,
            embedder=embedder,
        )

        _, kwargs = store.search.call_args
        assert kwargs["filters"] == {"restaurant_id": 42}


class TestMaybeTriggerSummary:
    def _make_count_db(self, count: int, summary: str | None = None) -> MagicMock:
        count_result = MagicMock()
        count_result.scalar_one = MagicMock(return_value=count)

        session_row = MagicMock()
        session_row.summary = summary

        db = MagicMock()
        db.execute = AsyncMock(return_value=count_result)
        db.get = AsyncMock(return_value=session_row)
        return db

    @pytest.mark.asyncio
    async def test_does_not_trigger_below_threshold(self) -> None:
        db = self._make_count_db(count=10)
        llm_client = MagicMock()

        with patch("src.core.session.fire_and_forget") as mock_fire_and_forget:
            await maybe_trigger_summary(
                session_id=uuid.uuid4(),
                db_session=db,
                llm_client=llm_client,
                summary_trigger=50,
            )
            mock_fire_and_forget.assert_not_called()

    @pytest.mark.asyncio
    async def test_does_not_trigger_when_summary_already_exists(self) -> None:
        db = self._make_count_db(count=60, summary="Existing summary.")
        llm_client = MagicMock()

        with patch("src.core.session.fire_and_forget") as mock_fire_and_forget:
            await maybe_trigger_summary(
                session_id=uuid.uuid4(),
                db_session=db,
                llm_client=llm_client,
                summary_trigger=50,
            )
            mock_fire_and_forget.assert_not_called()

    @pytest.mark.asyncio
    async def test_triggers_summary_task_when_count_at_threshold(self) -> None:
        db = self._make_count_db(count=50, summary=None)
        llm_client = MagicMock()

        with patch("src.core.session.fire_and_forget") as mock_fire_and_forget:
            await maybe_trigger_summary(
                session_id=uuid.uuid4(),
                db_session=db,
                llm_client=llm_client,
                summary_trigger=50,
            )
            mock_fire_and_forget.assert_called_once()

    @pytest.mark.asyncio
    async def test_triggers_summary_task_when_count_above_threshold(self) -> None:
        db = self._make_count_db(count=75, summary=None)
        llm_client = MagicMock()

        with patch("src.core.session.fire_and_forget") as mock_fire_and_forget:
            await maybe_trigger_summary(
                session_id=uuid.uuid4(),
                db_session=db,
                llm_client=llm_client,
                summary_trigger=50,
            )
            mock_fire_and_forget.assert_called_once()


class TestGenerateAndSaveSummary:
    def _make_db_with_messages(
        self, messages: list[MagicMock], session_summary: str | None = None
    ) -> MagicMock:
        session_row = MagicMock()
        session_row.summary = session_summary

        scalars = MagicMock()
        scalars.all = MagicMock(return_value=messages)
        execute_result = MagicMock()
        execute_result.scalars = MagicMock(return_value=scalars)

        db = MagicMock()
        db.execute = AsyncMock(return_value=execute_result)
        db.get = AsyncMock(return_value=session_row)
        db.commit = AsyncMock()
        # _generate_and_save_summary opens its own session via
        # `async with get_session_factory()() as db_session`, not a session
        # passed in by the caller -- so the mock must behave as an async
        # context manager whose __aenter__ returns itself.
        db.__aenter__ = AsyncMock(return_value=db)
        db.__aexit__ = AsyncMock(return_value=False)
        return db

    def _patch_session_factory(self, db: MagicMock):
        # get_session_factory is imported locally inside _generate_and_save_summary
        # (from src.services.database import get_session_factory), so it must be
        # patched at its defining module, not as an attribute of src.core.session.
        factory = MagicMock(return_value=db)
        return patch("src.services.database.get_session_factory", return_value=factory)

    @pytest.mark.asyncio
    async def test_saves_summary_to_session_row(self) -> None:
        messages = [
            _make_message("user", "What is the best dish?"),
            _make_message("assistant", "The biryani is most praised."),
        ]
        db = self._make_db_with_messages(messages)
        session_row = await db.get(None, None)
        session_row.summary = None

        llm_client = MagicMock()
        llm_client.complete = AsyncMock(return_value="Biryani is the most popular item.")

        with self._patch_session_factory(db):
            await _generate_and_save_summary(
                session_id=uuid.uuid4(),
                llm_client=llm_client,
            )

        assert session_row.summary == "Biryani is the most popular item."
        db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_swallows_llm_failure_gracefully(self) -> None:
        messages = [_make_message("user", "Hello")]
        db = self._make_db_with_messages(messages)

        llm_client = MagicMock()
        llm_client.complete = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

        with self._patch_session_factory(db):
            await _generate_and_save_summary(
                session_id=uuid.uuid4(),
                llm_client=llm_client,
            )

        db.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_does_not_save_when_session_row_missing(self) -> None:
        messages = [_make_message("user", "Hello")]
        db = self._make_db_with_messages(messages, session_summary=None)
        db.get = AsyncMock(return_value=None)

        llm_client = MagicMock()
        llm_client.complete = AsyncMock(return_value="A summary.")

        with self._patch_session_factory(db):
            await _generate_and_save_summary(
                session_id=uuid.uuid4(),
                llm_client=llm_client,
            )

        db.commit.assert_not_called()


class TestStoreSessionTurn:
    @pytest.mark.asyncio
    async def test_successful_upsert(self) -> None:
        embedder = _make_embedder()
        store = _make_vector_store()
        await store_session_turn(
            session_id=uuid.uuid4(),
            restaurant_id=1,
            role="user",
            content="Hello!",
            embedder=embedder,
            vector_store=store,
        )
        store.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_payload_includes_restaurant_id(self) -> None:
        """Regression guard for cross-session memory: build_session_context

        filters by restaurant_id, so every stored turn must carry it.
        """
        embedder = _make_embedder()
        store = _make_vector_store()
        await store_session_turn(
            session_id=uuid.uuid4(),
            restaurant_id=7,
            role="user",
            content="Hello!",
            embedder=embedder,
            vector_store=store,
        )
        _, args, _ = store.upsert.mock_calls[0]
        points = args[1]
        assert points[0]["payload"]["restaurant_id"] == 7

    @pytest.mark.asyncio
    async def test_failure_does_not_raise(self) -> None:
        embedder = _make_embedder()
        store = MagicMock()
        store.upsert = AsyncMock(side_effect=RuntimeError("Qdrant unavailable"))
        await store_session_turn(
            session_id=uuid.uuid4(),
            restaurant_id=1,
            role="user",
            content="Hello!",
            embedder=embedder,
            vector_store=store,
        )
