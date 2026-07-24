"""Unit tests for session context building."""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.session import (
    _generate_and_save_summary,
    build_recall_context,
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


class TestSessionScopedContext:
    """build_session_context is scoped to the CURRENT conversation only. It used
    to search restaurant-wide, which bled turns from unrelated past chats into
    every answer; cross-conversation recall now lives solely in
    build_recall_context (TestBuildRecallContext)."""

    @pytest.mark.asyncio
    async def test_ann_search_is_filtered_by_session_not_restaurant(self) -> None:
        embedder = _make_embedder()
        store = _make_vector_store(ann_results=[])
        db = _make_db_session(messages=[])
        session_id = uuid.uuid4()

        await build_session_context(
            session_id=session_id,
            restaurant_id=42,
            current_query="query",
            db_session=db,
            vector_store=store,
            embedder=embedder,
        )

        _, kwargs = store.search.call_args
        assert kwargs["filters"] == {"session_id": session_id}
        assert "restaurant_id" not in kwargs["filters"]

    @pytest.mark.asyncio
    async def test_no_cross_conversation_age_labels(self) -> None:
        # All ANN hits are now same-session, so the "(from a past conversation)"
        # labelling is gone entirely.
        from src.services.vector.base import SearchResult

        db = _make_db_session(messages=[_make_message("user", "current question")])
        embedder = _make_embedder()
        ann = [
            SearchResult(
                id="s1",
                score=0.9,
                payload={"role": "user", "content": "an earlier turn in this chat"},
            )
        ]
        store = _make_vector_store(ann_results=ann)

        result = await build_session_context(
            session_id=uuid.uuid4(),
            restaurant_id=1,
            current_query="follow up",
            db_session=db,
            vector_store=store,
            embedder=embedder,
        )

        assert "from a past conversation" not in result


class TestMaybeTriggerSummary:
    def _make_count_db(
        self,
        count: int,
        summary: str | None = None,
        summary_message_count: int | None = None,
    ) -> MagicMock:
        count_result = MagicMock()
        count_result.scalar_one = MagicMock(return_value=count)

        session_row = MagicMock()
        session_row.summary = summary
        session_row.summary_message_count = summary_message_count

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
    async def test_does_not_retrigger_before_refresh_interval(self) -> None:
        # Summary covers 50 of 60 messages; only 10 new, refresh wants 20.
        db = self._make_count_db(count=60, summary="Existing.", summary_message_count=50)
        llm_client = MagicMock()

        with patch("src.core.session.fire_and_forget") as mock_fire_and_forget:
            await maybe_trigger_summary(
                session_id=uuid.uuid4(),
                db_session=db,
                llm_client=llm_client,
                summary_trigger=50,
                refresh_every=20,
            )
            mock_fire_and_forget.assert_not_called()

    @pytest.mark.asyncio
    async def test_retriggers_once_refresh_interval_elapsed(self) -> None:
        # The old behaviour returned early whenever a summary existed, so this
        # case -- 20 messages accumulated past a summary -- never refreshed.
        db = self._make_count_db(count=70, summary="Existing.", summary_message_count=50)
        llm_client = MagicMock()

        with patch("src.core.session.fire_and_forget") as mock_fire_and_forget:
            await maybe_trigger_summary(
                session_id=uuid.uuid4(),
                db_session=db,
                llm_client=llm_client,
                summary_trigger=50,
                refresh_every=20,
            )
            mock_fire_and_forget.assert_called_once()

    @pytest.mark.asyncio
    async def test_triggers_for_legacy_summary_with_no_coverage_recorded(self) -> None:
        # Rows written by the old one-shot path have a summary but a NULL
        # coverage count; they must be re-summarized once, not skipped forever.
        db = self._make_count_db(count=60, summary="Legacy.", summary_message_count=None)
        llm_client = MagicMock()

        with patch("src.core.session.fire_and_forget") as mock_fire_and_forget:
            await maybe_trigger_summary(
                session_id=uuid.uuid4(),
                db_session=db,
                llm_client=llm_client,
                summary_trigger=50,
                refresh_every=20,
            )
            mock_fire_and_forget.assert_called_once()

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
        self,
        messages: list[MagicMock],
        session_summary: str | None = None,
        summary_message_count: int | None = None,
    ) -> MagicMock:
        session_row = MagicMock()
        session_row.summary = session_summary
        session_row.summary_message_count = summary_message_count

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
        assert session_row.summary_message_count == 2
        db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_refresh_folds_previous_summary_in_rather_than_rereading_all(self) -> None:
        """Constant-cost refresh: the prompt carries the existing summary plus
        only the messages past it, so a 200-message session does not send a
        40k-token prompt every time it refreshes."""
        new_messages = [
            _make_message("user", "And the desserts?"),
            _make_message("assistant", "Desserts average 4.2 stars."),
        ]
        db = self._make_db_with_messages(
            new_messages, session_summary="Discussed mains.", summary_message_count=50
        )
        session_row = await db.get(None, None)

        llm_client = MagicMock()
        llm_client.complete = AsyncMock(return_value="Discussed mains and desserts.")

        with self._patch_session_factory(db):
            await _generate_and_save_summary(session_id=uuid.uuid4(), llm_client=llm_client)

        prompt = llm_client.complete.await_args.kwargs["prompt"]
        assert "Discussed mains." in prompt
        assert "And the desserts?" in prompt
        # Coverage advances by the new messages only, not to some live count.
        assert session_row.summary_message_count == 52

    @pytest.mark.asyncio
    async def test_no_new_messages_skips_the_llm_call(self) -> None:
        db = self._make_db_with_messages(
            [], session_summary="Already current.", summary_message_count=10
        )
        llm_client = MagicMock()
        llm_client.complete = AsyncMock()

        with self._patch_session_factory(db):
            await _generate_and_save_summary(session_id=uuid.uuid4(), llm_client=llm_client)

        llm_client.complete.assert_not_awaited()

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

    @pytest.mark.asyncio
    async def test_answer_is_stored_in_payload_but_not_embedded(self) -> None:
        """The reply rides along in the payload; only the question is embedded.

        Embedding the reply separately would double the collection and let long
        replies outrank short questions for the fixed relevant_k slots.
        """
        embedder = _make_embedder()
        store = _make_vector_store()
        await store_session_turn(
            session_id=uuid.uuid4(),
            restaurant_id=1,
            role="user",
            content="What is my worst dish?",
            embedder=embedder,
            vector_store=store,
            answer="The truffle mac, at 2.1 stars.",
        )
        _, args, _ = store.upsert.mock_calls[0]
        payload = args[1][0]["payload"]
        assert payload["answer"] == "The truffle mac, at 2.1 stars."
        assert payload["content"] == "What is my worst dish?"
        embedder.embed_one.assert_awaited_once_with("What is my worst dish?")

    @pytest.mark.asyncio
    async def test_answer_key_omitted_when_not_supplied(self) -> None:
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
        _, args, _ = store.upsert.mock_calls[0]
        assert "answer" not in args[1][0]["payload"]


class TestPairedAnswerInContext:
    @pytest.mark.asyncio
    async def test_paired_answer_is_surfaced_and_attributed(self) -> None:
        """The gap this closes: before pairing, a fact the assistant stated was
        unreachable by semantic search once it left the recent-messages window,
        because only user questions were ever indexed."""
        from src.services.vector.base import SearchResult

        embedder = _make_embedder()
        store = _make_vector_store(
            ann_results=[
                SearchResult(
                    id=str(uuid.uuid4()),
                    score=0.9,
                    payload={
                        "role": "user",
                        "content": "What is my worst dish?",
                        "answer": "The truffle mac, at 2.1 stars.",
                        "session_id": str(uuid.uuid4()),
                        "created_at_ts": int(datetime.now(tz=UTC).timestamp()),
                    },
                )
            ]
        )
        db = _make_db_session(messages=[])

        result = await build_session_context(
            session_id=uuid.uuid4(),
            restaurant_id=1,
            current_query="what was that dish you mentioned?",
            db_session=db,
            vector_store=store,
            embedder=embedder,
        )

        assert "The truffle mac, at 2.1 stars." in result
        # Framed as something previously said, not as fresh evidence -- without
        # this the model restates stale figures as current fact.
        assert "Assistant previously answered:" in result


class TestPurgeExpiredSessions:
    @pytest.mark.asyncio
    async def test_deletes_from_postgres_and_qdrant(self) -> None:
        from src.core.session import purge_expired_sessions

        result = MagicMock()
        result.rowcount = 3
        db = MagicMock()
        db.execute = AsyncMock(return_value=result)
        db.commit = AsyncMock()
        store = MagicMock()
        store.delete_by_filter = AsyncMock()

        deleted = await purge_expired_sessions(db, store, ttl_days=30)

        assert deleted == 3
        db.commit.assert_awaited_once()
        collection, filters = store.delete_by_filter.await_args.args
        assert collection == "session_memory"
        assert "created_before" in filters

    @pytest.mark.asyncio
    async def test_qdrant_failure_does_not_lose_postgres_progress(self) -> None:
        """Postgres is already committed by then; a Qdrant blip must not raise
        and undo the reported count -- the next sweep retries on an absolute
        cutoff, so the leftover points are picked up anyway."""
        from src.core.session import purge_expired_sessions

        result = MagicMock()
        result.rowcount = 2
        db = MagicMock()
        db.execute = AsyncMock(return_value=result)
        db.commit = AsyncMock()
        store = MagicMock()
        store.delete_by_filter = AsyncMock(side_effect=RuntimeError("Qdrant down"))

        deleted = await purge_expired_sessions(db, store, ttl_days=30)

        assert deleted == 2


class TestBuildRecallContext:
    """Recency-based recall: this session, else the most recent prior one --
    never a semantically-matched random old chat."""

    @staticmethod
    def _exec(all_list=None, first_val=None) -> MagicMock:
        scalars = MagicMock()
        scalars.all = MagicMock(return_value=all_list or [])
        scalars.first = MagicMock(return_value=first_val)
        result = MagicMock()
        result.scalars = MagicMock(return_value=scalars)
        return result

    @pytest.mark.asyncio
    async def test_uses_current_conversation_when_it_has_turns(self) -> None:
        msgs = [_make_message("user", "how is my food?"), _make_message("assistant", "well-rated")]
        db = MagicMock()
        db.execute = AsyncMock(return_value=self._exec(all_list=msgs))

        result = await build_recall_context(uuid.uuid4(), restaurant_id=1, db_session=db)

        assert "[This conversation]" in result
        assert "how is my food?" in result
        # Only one query needed -- the current session had turns.
        assert db.execute.await_count == 1

    @pytest.mark.asyncio
    async def test_falls_back_to_prior_session_summary_when_current_is_empty(self) -> None:
        prior = MagicMock()
        prior.id = uuid.uuid4()
        prior.summary = "We discussed slow service and cold food."
        prior.last_activity_at = datetime.now(tz=UTC)

        db = MagicMock()
        # 1) current messages -> empty. 2) prior session lookup -> prior.
        db.execute = AsyncMock(side_effect=[self._exec(all_list=[]), self._exec(first_val=prior)])

        result = await build_recall_context(uuid.uuid4(), restaurant_id=1, db_session=db)

        assert "[Your previous conversation" in result
        assert "slow service" in result

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_current_and_no_prior(self) -> None:
        db = MagicMock()
        db.execute = AsyncMock(side_effect=[self._exec(all_list=[]), self._exec(first_val=None)])

        result = await build_recall_context(uuid.uuid4(), restaurant_id=1, db_session=db)

        assert result == ""
