from __future__ import annotations

from typing import Any

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchAny,
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


async def _create_collection_if_missing(
    qdrant: AsyncQdrantClient, name: str, **create_kwargs: Any
) -> None:
    """Create a collection unless it already exists, tolerating the race
    where another concurrent process (e.g. a sibling uvicorn worker on first
    boot against a brand-new Qdrant instance) creates it between our
    collection_exists() check and this create_collection() call.

    The collection_exists()-then-create() pattern alone has a TOCTOU gap: with
    multiple workers starting at once against an empty Qdrant, two can both
    see "missing" and both call create_collection(), and the loser gets a 409
    Conflict. That 409 means the collection now exists exactly as intended --
    it's not a real failure, so it's caught and logged rather than raised
    (which previously crashed the whole app on startup).
    """
    from qdrant_client.http.exceptions import UnexpectedResponse

    if await qdrant.collection_exists(name):
        logger.info("qdrant_collection_exists", collection=name)
        return
    try:
        await qdrant.create_collection(collection_name=name, **create_kwargs)
        logger.info("qdrant_collection_created", collection=name)
    except UnexpectedResponse as exc:
        if exc.status_code == 409:
            logger.info("qdrant_collection_created_concurrently", collection=name)
        else:
            raise


async def ensure_collections(settings: Any) -> None:
    """Idempotently create the Qdrant collections this app depends on.

    Must run to completion before anything upserts into these collections.
    Callable from multiple entry points (API startup, the standalone seed
    script) since _create_collection_if_missing() makes repeat/concurrent
    calls a no-op -- whichever process gets there first does the real work.
    """
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams

    qdrant = AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    try:
        # review_chunks uses named vectors: dense (ANN) + sparse (BM25-style via fastembed)
        await _create_collection_if_missing(
            qdrant,
            settings.qdrant_collection_reviews,
            vectors_config={
                "dense": VectorParams(size=settings.embedding_dim, distance=Distance.COSINE),
            },
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )

        # correction_embeddings, session_memory, and chat_cache use flat dense vectors only
        for name in [
            settings.qdrant_collection_corrections,
            settings.qdrant_collection_session_memory,
            settings.qdrant_collection_chat_cache,
        ]:
            await _create_collection_if_missing(
                qdrant,
                name,
                vectors_config=VectorParams(size=settings.embedding_dim, distance=Distance.COSINE),
            )
    finally:
        await qdrant.close()


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
            # AsyncQdrantClient has no search() method in the installed
            # qdrant-client version (search() was replaced by query_points());
            # confirmed via a live 500 from /chat/correct -- query_points() is
            # what hybrid_search() above already uses successfully.
            response = await self.client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=qdrant_filter,
                with_payload=True,
            )
            return [
                SearchResult(id=str(p.id), score=p.score, payload=p.payload or {})
                for p in response.points
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

        if filters.get("source_filter"):
            conditions.append(
                FieldCondition(key="source", match=MatchAny(any=filters["source_filter"]))
            )

        return Filter(must=conditions) if conditions else None
