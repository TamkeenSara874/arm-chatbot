from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchResult:
    id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


class BaseVectorStore(ABC):
    """Abstract vector store. All providers implement this interface."""

    @abstractmethod
    async def upsert(self, collection: str, points: list[dict[str, Any]]) -> None:
        """Upsert points into a collection.

        Each point dict must have: id (str), vector (list[float]),
        and optionally payload (dict).
        """
        ...

    @abstractmethod
    async def search(
        self,
        collection: str,
        query_vector: list[float],
        limit: int = 20,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        ...

    @abstractmethod
    async def delete(self, collection: str, ids: list[str]) -> None:
        ...

    @abstractmethod
    async def update_payload(
        self, collection: str, point_id: str, payload: dict[str, Any]
    ) -> None:
        ...
