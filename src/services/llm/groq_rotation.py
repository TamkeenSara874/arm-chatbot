from __future__ import annotations

import asyncio
import re
import time
from collections.abc import AsyncIterator
from typing import Any

import structlog
from groq import AsyncGroq, RateLimitError

from src.services.llm.base import AllModelsFailedError, BaseLLMClient, BaseModelT, UsageCallback
from src.utils.circuit_breaker import AsyncCircuitBreaker, CircuitBreakerError
from src.utils.metrics import llm_request_latency, llm_request_total
from src.utils.retry import fetch_with_retry

logger = structlog.get_logger()

_DEFAULT_COOLDOWN_SECONDS = 60.0
_RETRY_AFTER_RE = re.compile(r"try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s", re.IGNORECASE)


def _parse_retry_after(message: str) -> float:
    """Extract Groq's "Please try again in 2m33.792s" hint from an error message.

    Falls back to _DEFAULT_COOLDOWN_SECONDS if the message doesn't match --
    Groq's TPD (tokens-per-day) limit resets are usually minutes away, far
    longer than the few-second backoff fetch_with_retry uses elsewhere, so a
    short blind retry against the same key is pointless.
    """
    match = _RETRY_AFTER_RE.search(message)
    if not match:
        return _DEFAULT_COOLDOWN_SECONDS
    minutes = float(match.group(1) or 0)
    seconds = float(match.group(2))
    return minutes * 60 + seconds


class RotatingGroqClient(BaseLLMClient):
    """Round-robins across multiple free-tier Groq API keys.

    Groq's free tier rate-limits per API key/org (e.g. 100k tokens/day for
    llama-3.3-70b-versatile). A single key gets exhausted by real traffic
    within a day of moderate use, after which every decomposition call pays
    a retry-then-fallback-to-paid-OpenAI tax. Holding N keys and rotating to
    the next available one on a 429 gives N x the free daily quota before
    ever needing the paid fallback -- and skips the pointless short-backoff
    retry against a key that told us it needs minutes, not seconds.
    """

    def __init__(self, api_keys: list[str], model: str) -> None:
        if not api_keys:
            raise ValueError("RotatingGroqClient requires at least one API key")
        self._api_keys = api_keys
        self._clients = [AsyncGroq(api_key=k) for k in api_keys]
        self.model = model
        self._index = 0
        self._cooldown_until: dict[int, float] = {}
        self._lock = asyncio.Lock()
        # One breaker per key, not a shared breaker -- a single bad/exhausted
        # key tripping a shared breaker would incorrectly block every other
        # (healthy) key too, defeating the entire point of holding N
        # independent quotas.
        self._breakers = [AsyncCircuitBreaker(f"groq_key_{i}") for i in range(len(api_keys))]

    def _is_available(self, idx: int) -> bool:
        return self._cooldown_until.get(idx, 0.0) <= time.monotonic()

    async def _acquire_next(self) -> tuple[int, AsyncGroq] | None:
        """Return the next available (round-robin) key, or None if all are cooling down."""
        async with self._lock:
            for _ in range(len(self._clients)):
                idx = self._index % len(self._clients)
                self._index += 1
                if self._is_available(idx):
                    return idx, self._clients[idx]
            return None

    def _mark_rate_limited(self, idx: int, exc: RateLimitError) -> None:
        cooldown = _parse_retry_after(str(exc))
        self._cooldown_until[idx] = time.monotonic() + cooldown
        logger.warning(
            "groq_key_rate_limited",
            key_index=idx,
            cooldown_seconds=round(cooldown, 1),
            keys_total=len(self._clients),
        )

    async def _call_with_rotation(self, call_fn, label: str):
        last_exc: Exception | None = None
        attempts = 0
        while attempts < len(self._clients):
            acquired = await self._acquire_next()
            if acquired is None:
                break
            idx, client = acquired
            breaker = self._breakers[idx]
            attempts += 1
            start = time.perf_counter()
            try:

                async def _call(_breaker=breaker, _client=client) -> Any:
                    return await _breaker.call_async(call_fn, _client)

                result = await fetch_with_retry(
                    _call,
                    label=f"{label}:key{idx}",
                    dont_retry=(RateLimitError, CircuitBreakerError),
                )
                llm_request_total.labels(provider="groq", model=self.model, intent=label).inc()
                return result
            except RateLimitError as exc:
                self._mark_rate_limited(idx, exc)
                last_exc = exc
                continue
            except CircuitBreakerError as exc:
                # Treat like a rate limit for rotation purposes: cool this key
                # down and move on, rather than retrying a breaker that will
                # reject every attempt instantly anyway.
                self._cooldown_until[idx] = time.monotonic() + _DEFAULT_COOLDOWN_SECONDS
                logger.warning(
                    "groq_key_circuit_open", key_index=idx, keys_total=len(self._clients)
                )
                last_exc = exc
                continue
            except Exception as exc:
                # Any other error survived fetch_with_retry's retries on this
                # key -- still try the remaining keys before giving up
                # entirely, matching the existing "try every key" semantics.
                last_exc = exc
                continue
            finally:
                llm_request_latency.labels(provider="groq", model=self.model).observe(
                    time.perf_counter() - start
                )
        raise AllModelsFailedError(
            f"All {len(self._clients)} Groq API keys are unavailable"
        ) from last_exc

    async def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.3,
        usage_callback: UsageCallback | None = None,
    ) -> str:
        async def _call(client: AsyncGroq) -> str:
            response = await client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            )
            if usage_callback and response.usage:
                # Groq's usage payload has no prompt-caching signal.
                usage_callback(response.usage.prompt_tokens, response.usage.completion_tokens, 0)
            return response.choices[0].message.content or ""

        return await self._call_with_rotation(_call, label="complete")

    async def complete_structured(
        self,
        prompt: str,
        system: str,
        response_format: type[BaseModelT],
        max_tokens: int = 1024,
        temperature: float = 0.3,
        usage_callback: UsageCallback | None = None,
    ) -> BaseModelT:
        async def _call(client: AsyncGroq) -> BaseModelT:
            response = await client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = response.choices[0].message.content or "{}"
            if usage_callback and response.usage:
                # Groq's usage payload has no prompt-caching signal.
                usage_callback(response.usage.prompt_tokens, response.usage.completion_tokens, 0)
            return response_format.model_validate_json(raw)

        return await self._call_with_rotation(_call, label="complete_structured")

    async def stream(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.3,
        usage_callback: UsageCallback | None = None,
    ) -> AsyncIterator[str]:
        raise NotImplementedError(
            "RotatingGroqClient is used for decomposition only, not streaming"
        )
