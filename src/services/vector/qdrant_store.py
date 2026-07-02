from __future__ import annotations

from typing import Any

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    PointStruct,
    Prefetch,
    Range,
    SparseVector,
)

from src.services.vector.base import BaseVectorStore, SearchResult
from src.utils.circuit_breaker import qdrant_breaker
from src.utils.retry import fetch_with_retry

logger = structlog.get_logger()


class QdrantStore(BaseVectorStore):
    def __init__(self, url: str, api_key: str | None = None) -> None:
        self.client = AsyncQdrantClient(url=url, api_key=api_key or None)

    async def upsert(self, collection: str, points: list[dict[str, Any]]) -> None:
        """Upsert points into a collection.

        Accepts two point formats:
        - Dense only: {"id": str, "vector": list[float], "payload": dict}
        - Named vectors: {"id": str, "vector": {"dense": list[float], "sparse": {"indices": [...], "values": [...]}}, "payload": dict}
        """
        qdrant_points: list[PointStruct] = []
        for p in points:
            vec = p["vector"]
            if isinstance(vec, dict):
                vector_field: Any = {
                    name: SparseVector(indices=v["indices"], values=v["values"])
                    if isinstance(v, dict)
                    else v
                    for name, v in vec.items()
                }
            else:
                vector_field = vec
            qdrant_points.append(
                PointStruct(id=p["id"], vector=vector_field, payload=p.get("payload", {}))
            )

        async def _call() -> None:
            await self.client.upsert(collection_name=collection, points=qdrant_points, wait=True)

        await fetch_with_retry(lambda: qdrant_breaker.call_async(_call), label="qdrant.upsert")

    async def hybrid_search(
        self,
        collection: str,
        dense_vector: list[float],
        sparse_indices: list[int],
        sparse_values: list[float],
        limit: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Dense + sparse hybrid search fused with RRF inside Qdrant.

        Both prefetch arms apply the same payload filter so multi-tenant isolation
        is enforced at the database level for both vector types.
        """
        qdrant_filter = self._build_filter(filters) if filters else None

        async def _call() -> list[SearchResult]:
            response = await self.client.query_points(
                collection_name=collection,
                prefetch=[
                    Prefetch(
                        query=dense_vector,
                        using="dense",
                        limit=limit,
                        filter=qdrant_filter,
                    ),
                    Prefetch(
                        query=SparseVector(indices=sparse_indices, values=sparse_values),
                        using="sparse",
                        limit=limit,
                        filter=qdrant_filter,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
            return [
                SearchResult(id=str(p.id), score=p.score, payload=p.payload or {})
                for p in response.points
            ]

        return await fetch_with_retry(
            lambda: qdrant_breaker.call_async(_call), label="qdrant.hybrid_search"
        )

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        limit: int = 20,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Dense-only ANN search. Used for correction_embeddings and session_memory collections."""
        qdrant_filter = self._build_filter(filters) if filters else None

        async def _call() -> list[SearchResult]:
            results = await self.client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=qdrant_filter,
                with_payload=True,
            )
            return [
                SearchResult(id=str(r.id), score=r.score, payload=r.payload or {}) for r in results
            ]

        return await fetch_with_retry(
            lambda: qdrant_breaker.call_async(_call), label="qdrant.search"
        )

    async def delete(self, collection: str, ids: list[str]) -> None:
        async def _call() -> None:
            await self.client.delete(collection_name=collection, points_selector=ids, wait=True)

        await fetch_with_retry(lambda: qdrant_breaker.call_async(_call), label="qdrant.delete")

    async def update_payload(self, collection: str, point_id: str, payload: dict[str, Any]) -> None:
        async def _call() -> None:
            await self.client.set_payload(
                collection_name=collection,
                payload=payload,
                points=[point_id],
                wait=True,
            )

        await fetch_with_retry(
            lambda: qdrant_breaker.call_async(_call), label="qdrant.update_payload"
        )

    def _build_filter(self, filters: dict[str, Any]) -> Filter | None:
        conditions: list[FieldCondition] = []

        if "restaurant_id" in filters:
            conditions.append(
                FieldCondition(
                    key="restaurant_id",
                    match=MatchValue(value=filters["restaurant_id"]),
                )
            )

        if "session_id" in filters:
            conditions.append(
                FieldCondition(
                    key="session_id",
                    match=MatchValue(value=str(filters["session_id"])),
                )
            )

        range_kwargs: dict[str, float] = {}
        if filters.get("date_from") is not None:
            range_kwargs["gte"] = float(filters["date_from"])
        if filters.get("date_to") is not None:
            range_kwargs["lte"] = float(filters["date_to"])
        if range_kwargs:
            conditions.append(FieldCondition(key="review_date_ts", range=Range(**range_kwargs)))

        rating_kwargs: dict[str, float] = {}
        if filters.get("rating_min") is not None:
            rating_kwargs["gte"] = float(filters["rating_min"])
        if filters.get("rating_max") is not None:
            rating_kwargs["lte"] = float(filters["rating_max"])
        if rating_kwargs:
            conditions.append(FieldCondition(key="rating", range=Range(**rating_kwargs)))

        return Filter(must=conditions) if conditions else None
