from __future__ import annotations

import time
from collections.abc import AsyncIterator

import structlog
from groq import AsyncGroq

from src.services.llm.base import BaseLLMClient, BaseModelT, UsageCallback
from src.utils.circuit_breaker import groq_breaker
from src.utils.metrics import llm_request_latency, llm_request_total
from src.utils.retry import fetch_with_retry

logger = structlog.get_logger()


class GroqClient(BaseLLMClient):
    """Groq inference client. Used for query decomposition only."""

    def __init__(self, api_key: str, model: str) -> None:
        self.client = AsyncGroq(api_key=api_key)
        self.model = model

    async def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.3,
        usage_callback: UsageCallback | None = None,
    ) -> str:
        async def _call() -> str:
            response = await self.client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            )
            if usage_callback and response.usage:
                usage_callback(response.usage.prompt_tokens, response.usage.completion_tokens)
            return response.choices[0].message.content or ""

        start = time.perf_counter()
        try:
            result = await fetch_with_retry(
                lambda: groq_breaker.call_async(_call), label="groq.complete"
            )
            llm_request_total.labels(provider="groq", model=self.model, intent="complete").inc()
            return result
        finally:
            llm_request_latency.labels(provider="groq", model=self.model).observe(
                time.perf_counter() - start
            )

    async def complete_structured(
        self,
        prompt: str,
        system: str,
        response_format: type[BaseModelT],
        max_tokens: int = 1024,
        temperature: float = 0.3,
        usage_callback: UsageCallback | None = None,
    ) -> BaseModelT:
        # Groq supports JSON mode; validate the output against the Pydantic schema
        async def _call() -> BaseModelT:
            response = await self.client.chat.completions.create(
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
                usage_callback(response.usage.prompt_tokens, response.usage.completion_tokens)
            return response_format.model_validate_json(raw)

        start = time.perf_counter()
        try:
            result = await fetch_with_retry(
                lambda: groq_breaker.call_async(_call), label="groq.complete_structured"
            )
            llm_request_total.labels(
                provider="groq", model=self.model, intent="complete_structured"
            ).inc()
            return result
        finally:
            llm_request_latency.labels(provider="groq", model=self.model).observe(
                time.perf_counter() - start
            )

    async def stream(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.3,
        usage_callback: UsageCallback | None = None,
    ) -> AsyncIterator[str]:
        raise NotImplementedError("GroqClient is used for decomposition only, not streaming")
