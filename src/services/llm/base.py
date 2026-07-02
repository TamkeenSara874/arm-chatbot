from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TypeVar

import structlog
from pydantic import BaseModel

logger = structlog.get_logger()

BaseModelT = TypeVar("BaseModelT", bound=BaseModel)


class AllModelsFailedError(Exception):
    """Raised when every provider in the fallback chain has failed."""


class BaseLLMClient(ABC):
    """Abstract LLM client. All providers implement this interface."""

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        ...

    @abstractmethod
    async def complete_structured(
        self,
        prompt: str,
        system: str,
        response_format: type[BaseModelT],
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> BaseModelT:
        ...

    @abstractmethod
    async def stream(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        ...


class FallbackLLMClient(BaseLLMClient):
    """Tries providers in order, moving to the next on any failure."""

    def __init__(self, clients: list[BaseLLMClient]) -> None:
        if not clients:
            raise ValueError("FallbackLLMClient requires at least one client")
        self.clients = clients

    async def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        last_exc: Exception | None = None
        for client in self.clients:
            try:
                return await client.complete(prompt, system, max_tokens, temperature)
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "llm_provider_failed",
                    provider=type(client).__name__,
                    method="complete",
                    error=str(exc),
                )
        raise AllModelsFailedError("All LLM providers failed for complete()") from last_exc

    async def complete_structured(
        self,
        prompt: str,
        system: str,
        response_format: type[BaseModelT],
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> BaseModelT:
        last_exc: Exception | None = None
        for client in self.clients:
            try:
                return await client.complete_structured(
                    prompt, system, response_format, max_tokens, temperature
                )
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "llm_provider_failed",
                    provider=type(client).__name__,
                    method="complete_structured",
                    error=str(exc),
                )
        raise AllModelsFailedError("All LLM providers failed for complete_structured()") from last_exc

    async def stream(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        last_exc: Exception | None = None
        for client in self.clients:
            try:
                async for chunk in client.stream(prompt, system, max_tokens, temperature):
                    yield chunk
                return
            except NotImplementedError:
                continue
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "llm_provider_failed",
                    provider=type(client).__name__,
                    method="stream",
                    error=str(exc),
                )
        raise AllModelsFailedError("All LLM providers failed for stream()") from last_exc
