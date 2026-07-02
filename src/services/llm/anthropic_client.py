from __future__ import annotations

import time
from collections.abc import AsyncIterator

import structlog
from anthropic import AsyncAnthropic

from src.services.llm.base import BaseLLMClient, BaseModelT
from src.utils.circuit_breaker import anthropic_breaker
from src.utils.metrics import llm_request_latency, llm_request_total
from src.utils.retry import fetch_with_retry

logger = structlog.get_logger()


class AnthropicClient(BaseLLMClient):
    """Anthropic client. Used as fallback for complex generation."""

    def __init__(self, api_key: str, model: str) -> None:
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model

    async def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        async def _call() -> str:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            block = response.content[0]
            return block.text if hasattr(block, "text") else ""

        start = time.perf_counter()
        try:
            result = await fetch_with_retry(
                lambda: anthropic_breaker.call_async(_call), label="anthropic.complete"
            )
            llm_request_total.labels(
                provider="anthropic", model=self.model, intent="complete"
            ).inc()
            return result
        finally:
            llm_request_latency.labels(provider="anthropic", model=self.model).observe(
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
        # Use tool_use to enforce structured output; Anthropic always returns
        # tool_use blocks when tool_choice forces a specific tool.
        schema = response_format.model_json_schema()

        async def _call() -> BaseModelT:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                tools=[
                    {
                        "name": "structured_output",
                        "description": "Return the response as a structured object.",
                        "input_schema": schema,
                    }
                ],
                tool_choice={"type": "tool", "name": "structured_output"},
            )
            for block in response.content:
                if block.type == "tool_use":
                    return response_format.model_validate(block.input)
            raise ValueError("Anthropic did not return a tool_use block")

        start = time.perf_counter()
        try:
            result = await fetch_with_retry(
                lambda: anthropic_breaker.call_async(_call),
                label="anthropic.complete_structured",
            )
            llm_request_total.labels(
                provider="anthropic", model=self.model, intent="complete_structured"
            ).inc()
            return result
        finally:
            llm_request_latency.labels(provider="anthropic", model=self.model).observe(
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
            return await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )

        stream_resp = await fetch_with_retry(
            lambda: anthropic_breaker.call_async(_init), label="anthropic.stream"
        )
        llm_request_total.labels(
            provider="anthropic", model=self.model, intent="stream"
        ).inc()

        async with stream_resp as s:
            async for event in s:
                if (
                    hasattr(event, "type")
                    and event.type == "content_block_delta"
                    and hasattr(event.delta, "text")
                ):
                    yield event.delta.text
