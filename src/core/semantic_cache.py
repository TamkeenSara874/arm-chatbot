from __future__ import annotations

import uuid
from typing import Any

import structlog

from src.services.cache import RedisCache
from src.services.embedding.base import BaseEmbedder
from src.services.vector.base import BaseVectorStore

logger = structlog.get_logger()

CHAT_CACHE_NAMESPACE = uuid.UUID("6f6d0b3a-8f8e-4b3a-9c1a-0f6a2f9c9a11")


async def find_cached_response(
    query: str,
    restaurant_id: int,
    embedder: BaseEmbedder,
    vector_store: BaseVectorStore,
    redis_cache: RedisCache,
    collection: str,
    threshold: float = 0.95,
) -> tuple[dict[str, Any] | None, list[float]]:
    """Look up a semantically similar cached response for this (decomposed) query.

    Returns (value, query_vector) -- the vector is returned so callers can reuse
    it for retrieval on a miss instead of embedding the same text twice.
    """
    try:
        vector = await embedder.embed_one(query)
    except Exception as exc:
        logger.warning("semantic_cache_embed_failed", error=str(exc))
        return None, []

    try:
        results = await vector_store.search(
            collection,
            vector,
            limit=1,
            score_threshold=threshold,
            filters={"restaurant_id": restaurant_id},
        )
    except Exception as exc:
        logger.warning("semantic_cache_search_failed", error=str(exc))
        return None, vector

    if not results:
        return None, vector

    matched_query = results[0].payload.get("query")
    if not matched_query:
        return None, vector

    value = await redis_cache.get(restaurant_id, matched_query)
    return value, vector


async def store_cached_response(
    query: str,
    restaurant_id: int,
    value: dict[str, Any],
    embedder: BaseEmbedder,
    vector_store: BaseVectorStore,
    redis_cache: RedisCache,
    collection: str,
    precomputed_vector: list[float] | None = None,
) -> None:
    """Persist a response under the Redis exact-key AND the semantic cache index.

    Redis remains the single value store; the Qdrant point only carries the
    query text needed to re-derive the same Redis key on a semantic hit, so
    the response blob is never duplicated across stores.
    """
    await redis_cache.set(restaurant_id, query, value)

    try:
        vector = precomputed_vector or await embedder.embed_one(query)
        point_id = str(uuid.uuid5(CHAT_CACHE_NAMESPACE, f"{restaurant_id}:{query.strip().lower()}"))
        await vector_store.upsert(
            collection,
            [
                {
                    "id": point_id,
                    "vector": vector,
                    "payload": {"restaurant_id": restaurant_id, "query": query},
                }
            ],
        )
    except Exception as exc:
        logger.warning("semantic_cache_store_failed", error=str(exc))
