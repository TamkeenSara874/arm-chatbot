"""Asyncio-native circuit breaker. No tornado dependency required."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Coroutine
from typing import Any

import structlog

logger = structlog.get_logger()

_CLOSED = "closed"
_OPEN = "open"
_HALF_OPEN = "half_open"


class CircuitBreakerError(Exception):
    """Raised when the circuit is open and a call is rejected."""


class AsyncCircuitBreaker:
    """Thread-safe circuit breaker for async coroutines.

    States: closed (normal) -> open (rejecting after fail_max failures) ->
    half_open (one probe allowed after reset_timeout) -> closed or open.
    """

    def __init__(self, name: str, fail_max: int = 5, reset_timeout: float = 60.0) -> None:
        self.name = name
        self._fail_max = fail_max
        self._reset_timeout = reset_timeout
        self._state = _CLOSED
        self._fail_count = 0
        self._opened_at: float | None = None
        self._lock = threading.RLock()

    @property
    def fail_counter(self) -> int:
        return self._fail_count

    async def call_async(
        self,
        func: Callable[..., Coroutine[Any, Any, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        with self._lock:
            if self._state == _OPEN:
                elapsed = time.monotonic() - (self._opened_at or 0.0)
                if elapsed >= self._reset_timeout:
                    self._state = _HALF_OPEN
                else:
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' is open; retry in "
                        f"{self._reset_timeout - elapsed:.0f}s"
                    )

        try:
            result = await func(*args, **kwargs)
        except Exception as exc:
            with self._lock:
                self._fail_count += 1
                logger.error(
                    "circuit_breaker_failure",
                    name=self.name,
                    fail_count=self._fail_count,
                    error=str(exc),
                )
                if self._fail_count >= self._fail_max and self._state != _OPEN:
                    self._state = _OPEN
                    self._opened_at = time.monotonic()
                    logger.warning(
                        "circuit_breaker_opened",
                        name=self.name,
                        fail_count=self._fail_count,
                    )
            raise

        with self._lock:
            self._fail_count = 0
            if self._state == _HALF_OPEN:
                self._state = _CLOSED
                self._opened_at = None
                logger.info("circuit_breaker_closed", name=self.name)

        return result


def _breaker(name: str, fail_max: int = 5, reset_timeout: int = 60) -> AsyncCircuitBreaker:
    return AsyncCircuitBreaker(name=name, fail_max=fail_max, reset_timeout=float(reset_timeout))


groq_breaker = _breaker("groq")
openai_breaker = _breaker("openai")
qdrant_breaker = _breaker("qdrant", fail_max=3, reset_timeout=30)
