"""Unit tests for fire_and_forget — no I/O required."""

import asyncio
import gc

import pytest

from src.utils.background import _background_tasks, fire_and_forget


@pytest.mark.asyncio
async def test_task_runs_to_completion() -> None:
    ran = False

    async def coro():
        nonlocal ran
        await asyncio.sleep(0.01)
        ran = True

    fire_and_forget(coro())
    await asyncio.sleep(0.05)
    assert ran is True


@pytest.mark.asyncio
async def test_task_survives_garbage_collection_with_no_local_reference() -> None:
    """Regression test for the exact bug this module fixes.

    asyncio.create_task() alone returns a Task the event loop only weakly
    references -- with nothing else holding it, gc.collect() can reap it
    mid-execution with no error raised, silently dropping whatever it was
    doing (confirmed live: a chat message that never got persisted). This
    test forces a collection right after scheduling to prove fire_and_forget's
    module-level set keeps the task alive regardless.
    """
    ran = False

    async def coro():
        nonlocal ran
        await asyncio.sleep(0.05)
        ran = True

    fire_and_forget(coro())
    gc.collect()
    await asyncio.sleep(0.1)
    assert ran is True


@pytest.mark.asyncio
async def test_task_is_discarded_from_registry_after_completion() -> None:
    async def coro():
        await asyncio.sleep(0.01)

    task = fire_and_forget(coro())
    assert task in _background_tasks
    await task
    assert task not in _background_tasks


@pytest.mark.asyncio
async def test_exception_in_task_does_not_propagate_to_caller() -> None:
    async def coro():
        raise ValueError("boom")

    task = fire_and_forget(coro())
    await asyncio.sleep(0.01)
    assert task.done()
    with pytest.raises(ValueError, match="boom"):
        await task
