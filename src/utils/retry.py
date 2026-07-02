import asyncio
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

import structlog

logger = structlog.get_logger()

T = TypeVar("T")


async def fetch_with_retry(
    coro_factory: Callable[[], Coroutine[Any, Any, T]],
    max_attempts: int = 3,
    backoff_base: float = 1.5,
    label: str = "unknown",
) -> T:
    """Retry a coroutine with exponential backoff.

    Works alongside circuit breakers: retry handles transient failures (429s,
    brief network blips); the circuit breaker handles sustained degradation.
    """
    for attempt in range(max_attempts):
        try:
            return await coro_factory()
        except Exception as exc:
            if attempt == max_attempts - 1:
                logger.error(
                    "retry_exhausted",
                    label=label,
                    attempts=max_attempts,
                    error=str(exc),
                    exc_info=True,
                )
                raise
            wait = backoff_base**attempt
            logger.warning(
                "retry_attempt",
                label=label,
                attempt=attempt + 1,
                max_attempts=max_attempts,
                wait_seconds=round(wait, 2),
                error=str(exc),
            )
            await asyncio.sleep(wait)
    raise RuntimeError("unreachable")  # pragma: no cover
