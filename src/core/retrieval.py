from __future__ import annotations

import asyncio
import time

import structlog

from src.services.embedding.base import BaseEmbedder
from src.services.embedding.sparse_embedder import compute_sparse_vector
from src.services.vector.base import BaseVectorStore, SearchResult

logger = structlog.get_logger()


async def hybrid_retrieve(
    query: str,
    restaurant_id: int,
    embedder: BaseEmbedder,
    vector_store: BaseVectorStore,
    collection: str,
    top_k: int = 6,
    date_from: float | None = None,
    date_to: float | None = None,
    rating_min: float | None = None,
    rating_max: float | None = None,
    reranker_model: str | None = None,
) -> list[SearchResult]:
    """Hybrid retrieval using Qdrant native dense + sparse RRF.

    Dense and sparse vectors are computed concurrently, then passed to Qdrant's
    query_points endpoint which fuses them server-side with Reciprocal Rank Fusion.
    This is multi-worker safe -- there is no in-process state.
    """
    filters: dict = {"restaurant_id": restaurant_id}
    if date_from is not None:
        filters["date_from"] = date_from
    if date_to is not None:
        filters["date_to"] = date_to
    if rating_min is not None:
        filters["rating_min"] = rating_min
    if rating_max is not None:
        filters["rating_max"] = rating_max

    ann_limit = max(top_k * 5, 50)

    t0 = time.perf_counter()
    dense_task = asyncio.create_task(embedder.embed_one(query))
    sparse_task = asyncio.create_task(compute_sparse_vector(query))

    try:
        dense_vector, sparse_vec = await asyncio.gather(dense_task, sparse_task)
    except Exception as exc:
        logger.error("retrieval_embedding_failed", error=str(exc))
        return []
    embed_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    try:
        results = await vector_store.hybrid_search(
            collection=collection,
            dense_vector=dense_vector,
            sparse_indices=sparse_vec.indices,
            sparse_values=sparse_vec.values,
            limit=ann_limit if reranker_model else top_k,
            filters=filters,
        )
    except Exception as exc:
        logger.warning("hybrid_search_failed", error=str(exc))
        return []
    search_ms = (time.perf_counter() - t1) * 1000.0

    if not results:
        logger.info(
            "retrieval_breakdown", embed_ms=round(embed_ms, 1), search_ms=round(search_ms, 1), rerank_ms=0.0
        )
        return []

    if reranker_model:
        from src.core.reranker import rerank

        # Capped independent of top_k -- for aggregation queries (top_k=20)
        # this was results[:80], and cross-encoder scoring is CPU-bound with
        # cost roughly linear in candidate count. Reranking measurably 80
        # candidates vs. 24 for a simple query (top_k=6) was the dominant
        # cost in a live reproduction that took 52s end-to-end, ~45s of it
        # before the first token. Reranking quality gains plateau well
        # before 30 candidates; capping here keeps latency bounded across
        # all top_k values while still returning the full top_k results.
        candidates = results[: min(top_k * 4, 30)]
        t2 = time.perf_counter()
        reranked = await rerank(query, candidates, model_name=reranker_model, top_k=top_k)
        rerank_ms = (time.perf_counter() - t2) * 1000.0
        logger.info(
            "retrieval_breakdown",
            embed_ms=round(embed_ms, 1),
            search_ms=round(search_ms, 1),
            rerank_ms=round(rerank_ms, 1),
            candidate_count=len(candidates),
        )
        return reranked

    logger.info(
        "retrieval_breakdown", embed_ms=round(embed_ms, 1), search_ms=round(search_ms, 1), rerank_ms=0.0
    )
    return results[:top_k]
