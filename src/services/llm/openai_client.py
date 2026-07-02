from __future__ import annotations

import time
from collections.abc import AsyncIterator

import structlog
from openai import AsyncOpenAI

from src.services.llm.base import BaseLLMClient, BaseModelT
from src.utils.circuit_breaker import openai_breaker
from src.utils.metrics import llm_request_latency, llm_request_total
from src.utils.retry import fetch_with_retry

logger = structlog.get_logger()


class OpenAIClient(BaseLLMClient):
    """OpenAI client. One instance per model; use factory to get simple/complex variants."""

    def __init__(self, api_key: str, model: str) -> None:
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.3,
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
            return response.choices[0].message.content or ""

        start = time.perf_counter()
        try:
            result = await fetch_with_retry(
                lambda: openai_breaker.call_async(_call), label="openai.complete"
            )
            llm_request_total.labels(provider="openai", model=self.model, intent="complete").inc()
            return result
        finally:
            llm_request_latency.labels(provider="openai", model=self.model).observe(
                time.perf_counter() - start
            )

    async def complete_structured(
        self,
        prompt: str,
        system: str,
        response_format: type[BaseModelT],
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> BaseModelT:
        async def _call() -> BaseModelT:
            response = await self.client.beta.chat.completions.parse(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                response_format=response_format,
            )
            parsed = response.choices[0].message.parsed
            if parsed is None:
                raise ValueError("OpenAI structured output returned None parsed result")
            return parsed

        start = time.perf_counter()
        try:
            result = await fetch_with_retry(
                lambda: openai_breaker.call_async(_call), label="openai.complete_structured"
            )
            llm_request_total.labels(
                provider="openai", model=self.model, intent="complete_structured"
            ).inc()
            return result
        finally:
            llm_request_latency.labels(provider="openai", model=self.model).observe(
                time.perf_counter() - start
            )

    async def stream(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        async def _init():
            return await self.client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                stream=True,
            )

        stream_resp = await fetch_with_retry(
            lambda: openai_breaker.call_async(_init), label="openai.stream"
        )
        llm_request_total.labels(provider="openai", model=self.model, intent="stream").inc()

        async for chunk in stream_resp:
            content = chunk.choices[0].delta.content
            if content:
                yield content
