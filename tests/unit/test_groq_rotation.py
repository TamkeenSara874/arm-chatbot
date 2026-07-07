"""Unit tests for RotatingGroqClient -- multi-key rotation on rate limits."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from groq import RateLimitError

from src.services.llm.base import AllModelsFailedError
from src.services.llm.groq_rotation import RotatingGroqClient, _parse_retry_after


def _rate_limit_error(message: str = "Error code: 429 - rate limit reached.") -> RateLimitError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(429, request=request, text=message)
    return RateLimitError(message, response=response, body=None)


def _mock_completion(content: str = "response text") -> MagicMock:
    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(content=content))]
    completion.usage = None
    return completion


class TestParseRetryAfter:
    def test_parses_minutes_and_seconds(self) -> None:
        msg = "Please try again in 2m33.792s. Need more tokens?"
        assert abs(_parse_retry_after(msg) - 153.792) < 0.01

    def test_parses_seconds_only(self) -> None:
        msg = "Please try again in 45.5s."
        assert abs(_parse_retry_after(msg) - 45.5) < 0.01

    def test_falls_back_to_default_when_unparseable(self) -> None:
        assert _parse_retry_after("some unrelated error") == 60.0


class TestRotatingGroqClient:
    def test_requires_at_least_one_key(self) -> None:
        with pytest.raises(ValueError):
            RotatingGroqClient(api_keys=[], model="llama-3.3-70b-versatile")

    @pytest.mark.asyncio
    async def test_uses_first_key_when_available(self) -> None:
        client = RotatingGroqClient(api_keys=["key1", "key2"], model="llama-3.3-70b-versatile")
        client._clients[0].chat.completions.create = AsyncMock(
            return_value=_mock_completion("hello")
        )
        client._clients[1].chat.completions.create = AsyncMock(
            return_value=_mock_completion("should not be used")
        )

        result = await client.complete("prompt", system="sys")

        assert result == "hello"
        client._clients[0].chat.completions.create.assert_awaited_once()
        client._clients[1].chat.completions.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rotates_to_next_key_on_rate_limit(self) -> None:
        client = RotatingGroqClient(api_keys=["key1", "key2"], model="llama-3.3-70b-versatile")
        client._clients[0].chat.completions.create = AsyncMock(side_effect=_rate_limit_error())
        client._clients[1].chat.completions.create = AsyncMock(
            return_value=_mock_completion("from key2")
        )

        result = await client.complete("prompt", system="sys")

        assert result == "from key2"
        assert 0 in client._cooldown_until

    @pytest.mark.asyncio
    async def test_skips_key_already_in_cooldown(self) -> None:
        client = RotatingGroqClient(api_keys=["key1", "key2"], model="llama-3.3-70b-versatile")
        client._mark_rate_limited(0, _rate_limit_error("try again in 60s"))
        client._clients[1].chat.completions.create = AsyncMock(
            return_value=_mock_completion("from key2")
        )

        result = await client.complete("prompt", system="sys")

        assert result == "from key2"
        client._clients[1].chat.completions.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_raises_when_all_keys_rate_limited(self) -> None:
        client = RotatingGroqClient(api_keys=["key1", "key2"], model="llama-3.3-70b-versatile")
        client._clients[0].chat.completions.create = AsyncMock(side_effect=_rate_limit_error())
        client._clients[1].chat.completions.create = AsyncMock(side_effect=_rate_limit_error())

        with pytest.raises(AllModelsFailedError):
            await client.complete("prompt", system="sys")

    @pytest.mark.asyncio
    async def test_round_robins_across_successive_calls(self) -> None:
        client = RotatingGroqClient(api_keys=["key1", "key2"], model="llama-3.3-70b-versatile")
        client._clients[0].chat.completions.create = AsyncMock(
            return_value=_mock_completion("from key1")
        )
        client._clients[1].chat.completions.create = AsyncMock(
            return_value=_mock_completion("from key2")
        )

        first = await client.complete("prompt1")
        second = await client.complete("prompt2")

        assert first == "from key1"
        assert second == "from key2"

    @pytest.mark.asyncio
    async def test_usage_callback_invoked_on_success(self) -> None:
        client = RotatingGroqClient(api_keys=["key1"], model="llama-3.3-70b-versatile")
        completion = _mock_completion("hi")
        completion.usage = MagicMock(prompt_tokens=100, completion_tokens=20)
        client._clients[0].chat.completions.create = AsyncMock(return_value=completion)
        callback = MagicMock()

        await client.complete("prompt", usage_callback=callback)

        callback.assert_called_once_with(100, 20, 0)

    @pytest.mark.asyncio
    async def test_stream_not_implemented(self) -> None:
        # stream() has no `yield` in its body (matches GroqClient's own
        # stream()), so it's a plain coroutine, not an async generator --
        # awaiting it directly raises NotImplementedError as expected. (An
        # `async for` over it would TypeError before reaching that, same
        # pre-existing quirk GroqClient already has -- FallbackLLMClient's
        # generic `except Exception` still moves on to the next provider
        # either way, so this never breaks the fallback chain in practice.)
        client = RotatingGroqClient(api_keys=["key1"], model="llama-3.3-70b-versatile")
        with pytest.raises(NotImplementedError):
            await client.stream("prompt")
