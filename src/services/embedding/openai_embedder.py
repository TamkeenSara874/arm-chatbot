from __future__ import annotations

import structlog
from openai import AsyncOpenAI

from src.services.embedding.base import BaseEmbedder, EmbeddingUsageCallback
from src.utils.circuit_breaker import openai_breaker
from src.utils.retry import fetch_with_retry

logger = structlog.get_logger()


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, api_key: str, model: str, embedding_dim: int, batch_size: int = 100) -> None:
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self._dim = embedding_dim
        self.batch_size = batch_size

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(
        self, texts: list[str], usage_callback: EmbeddingUsageCallback | None = None
    ) -> list[list[float]]:
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            async def _call(b: list[str] = batch) -> tuple[list[list[float]], int]:
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=b,
                )
                total_tokens = response.usage.total_tokens if response.usage else 0
                return [item.embedding for item in response.data], total_tokens

            embeddings, total_tokens = await fetch_with_retry(
                lambda: openai_breaker.call_async(_call), label="openai.embed"
            )
            if usage_callback and total_tokens:
                usage_callback(total_tokens)
            all_embeddings.extend(embeddings)
            logger.debug(
                "embed_batch_complete",
                batch_start=i,
                batch_size=len(batch),
                total_so_far=len(all_embeddings),
            )

        return all_embeddings
