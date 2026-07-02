from __future__ import annotations

from typing import Any

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
)

from src.services.vector.base import BaseVectorStore, SearchResult
from src.utils.circuit_breaker import qdrant_breaker
from src.utils.retry import fetch_with_retry

logger = structlog.get_logger()


class QdrantStore(BaseVectorStore):
    def __init__(self, url: str, api_key: str | None = None) -> None:
        self.client = AsyncQdrantClient(url=url, api_key=api_key or None)

    async def upsert(self, collection: str, points: list[dict[str, Any]]) -> None:
        qdrant_points = [
            PointStruct(
                id=p["id"],
                vector=p["vector"],
                payload=p.get("payload", {}),
            )
            for p in points
        ]

        async def _call() -> None:
            await self.client.upsert(collection_name=collection, points=qdrant_points, wait=True)

        await fetch_with_retry(lambda: qdrant_breaker.call_async(_call), label="qdrant.upsert")

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        limit: int = 20,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
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
