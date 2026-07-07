from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog

from src.services.embedding.base import BaseEmbedder
from src.services.embedding.sparse_embedder import compute_sparse_vector
from src.services.vector.base import BaseVectorStore, SearchResult
from src.utils.metrics import retrieval_latency

if TYPE_CHECKING:
    from src.models.schemas import DecomposedQuery

logger = structlog.get_logger()


@dataclass
class RetrievalTiming:
    """Populated in place by hybrid_retrieve() when passed in, so a caller
    can get the embed/search/rerank breakdown without changing the function's
    return type (list[SearchResult] stays untouched -- three existing test
    call sites rely on that)."""

    embed_ms: float = 0.0
    search_ms: float = 0.0
    rerank_ms: float = 0.0


@dataclass
class RetrievalParams:
    top_k: int
    date_from: float | None
    date_to: float | None
    rating_min: float | None
    rating_max: float | None
    is_aggregation: bool


def build_retrieval_params(decomposed: DecomposedQuery) -> RetrievalParams:
    """Derive retrieval filters/top_k from a decomposed query.

    Malformed ISO date strings are silently suppressed to None rather than
    raising -- a best-effort filter is preferable to hard-failing the whole
    query over a date the model didn't format quite right.
    """
    # Improvement queries are inherently broad ("how can I improve?") -- force
    # the wider top_k regardless of whether the model set needs_aggregation,
    # rather than relying on it to infer that on its own every time.
    is_aggregation = decomposed.needs_aggregation or decomposed.intent == "improvement"
    top_k = 20 if is_aggregation else 6

    date_from: float | None = None
    date_to: float | None = None
    if decomposed.date_filter:
        if decomposed.date_filter.from_date:
            with contextlib.suppress(ValueError):
                date_from = (
                    datetime.fromisoformat(decomposed.date_filter.from_date)
                    .replace(tzinfo=UTC)
                    .timestamp()
                )
        if decomposed.date_filter.to_date:
            with contextlib.suppress(ValueError):
                date_to = (
                    datetime.fromisoformat(decomposed.date_filter.to_date)
                    .replace(tzinfo=UTC)
                    .timestamp()
                )

    rating_min = decomposed.rating_filter.min if decomposed.rating_filter else None
    rating_max = decomposed.rating_filter.max if decomposed.rating_filter else None

    return RetrievalParams(
        top_k=top_k,
        date_from=date_from,
        date_to=date_to,
        rating_min=rating_min,
        rating_max=rating_max,
        is_aggregation=is_aggregation,
    )


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
    precomputed_dense_vector: list[float] | None = None,
    timing: RetrievalTiming | None = None,
) -> list[SearchResult]:
    """Hybrid retrieval using Qdrant native dense + sparse RRF.

    Dense and sparse vectors are computed concurrently, then passed to Qdrant's
    query_points endpoint which fuses them server-side with Reciprocal Rank Fusion.
    This is multi-worker safe -- there is no in-process state.

    precomputed_dense_vector lets a caller that already embedded this exact
    query text (e.g. a semantic cache lookup that just missed) skip a second,
    redundant embedding call.

    timing, if provided, is populated in place with the embed/search/rerank
    breakdown -- these numbers were already computed for the "retrieval_breakdown"
    log line below, just never surfaced to the caller.
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
    sparse_task = asyncio.create_task(compute_sparse_vector(query))

    try:
        if precomputed_dense_vector is not None:
            dense_vector = precomputed_dense_vector
            sparse_vec = await sparse_task
        else:
            dense_task = asyncio.create_task(embedder.embed_one(query))
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
    retrieval_latency.observe(search_ms / 1000.0)

    if not results:
        logger.info(
            "retrieval_breakdown",
            embed_ms=round(embed_ms, 1),
            search_ms=round(search_ms, 1),
            rerank_ms=0.0,
        )
        if timing is not None:
            timing.embed_ms = embed_ms
            timing.search_ms = search_ms
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
        if timing is not None:
            timing.embed_ms = embed_ms
            timing.search_ms = search_ms
            timing.rerank_ms = rerank_ms
        return reranked

    logger.info(
        "retrieval_breakdown",
        embed_ms=round(embed_ms, 1),
        search_ms=round(search_ms, 1),
        rerank_ms=0.0,
    )
    if timing is not None:
        timing.embed_ms = embed_ms
        timing.search_ms = search_ms
    return results[:top_k]
