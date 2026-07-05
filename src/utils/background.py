"""Fire-and-forget asyncio task helper.

asyncio.create_task() returns a Task the event loop only holds a *weak*
reference to -- a Task with no other strong reference anywhere can be
garbage-collected mid-execution, silently, with no exception raised. This bit
three call sites in this codebase (chat.py's post-response persistence/cache
write, session.py's rolling summary, ingest.py's background ingest job), all
using the bare `asyncio.create_task(coro)` pattern with no reference kept --
confirmed live via a direct repro: a message_id returned from /chat/query
never appeared in Postgres even 4.5+ seconds later, with zero error logged.
See https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task
("Important: Save a reference to the result of this function...").
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

_background_tasks: set[asyncio.Task[Any]] = set()


def fire_and_forget(
    coro: Coroutine[Any, Any, Any], *, name: str | None = None
) -> asyncio.Task[Any]:
    """Schedule coro as a task that keeps running even though nothing awaits it.

    Keeps a strong reference in a module-level set until the task completes,
    so it can't be garbage-collected mid-execution.
    """
    task = asyncio.create_task(coro, name=name)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task
