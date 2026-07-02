"""Unit tests for fetch_with_retry — no I/O required."""

import pytest

from src.utils.retry import fetch_with_retry


@pytest.mark.asyncio
async def test_returns_on_first_success() -> None:
    calls = 0

    async def coro():
        nonlocal calls
        calls += 1
        return "ok"

    result = await fetch_with_retry(coro, max_attempts=3)
    assert result == "ok"
    assert calls == 1


@pytest.mark.asyncio
async def test_retries_on_failure_then_succeeds() -> None:
    calls = 0

    async def coro():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ValueError("transient")
        return "success"

    result = await fetch_with_retry(coro, max_attempts=3, backoff_base=0.01)
    assert result == "success"
    assert calls == 3


@pytest.mark.asyncio
async def test_raises_after_max_attempts() -> None:
    calls = 0

    async def coro():
        nonlocal calls
        calls += 1
        raise RuntimeError("always fails")

    with pytest.raises(RuntimeError, match="always fails"):
        await fetch_with_retry(coro, max_attempts=3, backoff_base=0.01)

    assert calls == 3
