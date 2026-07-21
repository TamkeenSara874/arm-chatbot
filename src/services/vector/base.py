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
    ) -> list[SearchResult]: ...

    @abstractmethod
    async def hybrid_search(
        self,
        collection: str,
        dense_vector: list[float],
        sparse_indices: list[int],
        sparse_values: list[float],
        limit: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Hybrid dense + sparse search fused server-side with RRF.

        Each point dict passed to upsert() must carry a named-vector dict
        with keys 'dense' (list[float]) and 'sparse' (dict with 'indices' and 'values').
        """
        ...

    @abstractmethod
    async def delete(self, collection: str, ids: list[str]) -> None: ...

    @abstractmethod
    async def delete_by_filter(self, collection: str, filters: dict[str, Any]) -> None:
        """Delete every point matching filters.

        Needed by the session reaper, which knows a cutoff timestamp but not
        the point ids -- fetching ids first would mean paging the whole
        expired set back over the wire just to send it again.
        """
        ...

    @abstractmethod
    async def update_payload(
        self, collection: str, point_id: str, payload: dict[str, Any]
    ) -> None: ...
