from abc import ABC, abstractmethod
from collections.abc import Callable

# Invoked with total tokens billed for one embed() call. Optional and
# additive, same pattern as BaseLLMClient's UsageCallback.
EmbeddingUsageCallback = Callable[[int], None]


class BaseEmbedder(ABC):
    """Abstract embedding client."""

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    async def embed(
        self, texts: list[str], usage_callback: EmbeddingUsageCallback | None = None
    ) -> list[list[float]]: ...

    async def embed_one(self, text: str) -> list[float]:
        results = await self.embed([text])
        return results[0]

    async def verify_dim(self) -> None:
        """Fail fast if the model returns a different dimension than configured."""
        vector = await self.embed_one("dimension check")
        if len(vector) != self.dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dim}, got {len(vector)}"
            )
