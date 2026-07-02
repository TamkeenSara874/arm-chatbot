from __future__ import annotations

import asyncio
import re

import structlog
from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.db_entities import ReviewChunkMeta
from src.services.embedding.base import BaseEmbedder
from src.services.vector.base import BaseVectorStore, SearchResult

logger = structlog.get_logger()

# Module-level BM25 index cache keyed by restaurant_id.
#
# HARD CONSTRAINT: run Uvicorn with --workers 1 while this cache is in use.
# Multiple workers each build independent indexes from the same Postgres data,
# but the RRF scores from their independent BM25 instances are non-comparable,
# silently degrading retrieval quality. See docs/architecture.md Known Limitations.
_bm25_cache: dict[int, tuple[BM25Okapi, list[dict]]] = {}
_bm25_lock = asyncio.Lock()


def invalidate_bm25_cache(restaurant_id: int) -> None:
    """Remove the BM25 index for a restaurant so it is rebuilt on the next query."""
    _bm25_cache.pop(restaurant_id, None)
    logger.info("bm25_cache_invalidated", restaurant_id=restaurant_id)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


async def _get_or_build_bm25(
    restaurant_id: int,
    db_session: AsyncSession,
) -> tuple[BM25Okapi, list[dict]] | None:
    if restaurant_id in _bm25_cache:
        return _bm25_cache[restaurant_id]

    async with _bm25_lock:
        if restaurant_id in _bm25_cache:
            return _bm25_cache[restaurant_id]

        stmt = select(ReviewChunkMeta).where(
            ReviewChunkMeta.restaurant_id == restaurant_id,
            ReviewChunkMeta.has_content.is_(True),
        )
        result = await db_session.execute(stmt)
        rows = result.scalars().all()

        if not rows:
            return None

        corpus: list[dict] = []
        tokenized: list[list[str]] = []

        for row in rows:
            text = row.chunk_text or ""
            if not text.strip():
                continue
            review_date_iso = row.review_date.isoformat() if row.review_date else None
            review_date_ts = int(row.review_date.timestamp()) if row.review_date else None
            corpus.append(
                {
                    "id": row.chunk_id,
                    "text": text,
                    "rating": row.rating,
                    "sentiment_label": row.sentiment_label,
                    "sentiment_rating_agree": row.sentiment_rating_agree,
                    "review_date": review_date_iso,
                    "review_date_ts": review_date_ts,
                    "username": row.username,
                    "source": row.source,
                    "food_entities": [],
                    "has_injection_attempt": row.has_injection_attempt,
                    "date_inferred": row.date_inferred,
                }
            )
            tokenized.append(_tokenize(text))

        if not tokenized:
            return None

        bm25 = BM25Okapi(tokenized)
        _bm25_cache[restaurant_id] = (bm25, corpus)
        logger.info("bm25_corpus_built", restaurant_id=restaurant_id, chunks=len(corpus))
        return _bm25_cache[restaurant_id]


async def hybrid_retrieve(
    query: str,
    restaurant_id: int,
    embedder: BaseEmbedder,
    vector_store: BaseVectorStore,
    collection: str,
    db_session: AsyncSession,
    top_k: int = 6,
    rrf_k: int = 60,
    date_from: float | None = None,
    date_to: float | None = None,
    rating_min: float | None = None,
    rating_max: float | None = None,
) -> list[SearchResult]:
    """Hybrid retrieval: dense ANN from Qdrant + BM25, fused via RRF (k=60).

    Dense and BM25 searches run concurrently. If Qdrant is unavailable, the system
    falls back to BM25-only with a WARNING logged. Returns SearchResult objects ordered
    by RRF score; each result's .score field holds that RRF score.
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

    dense_task = asyncio.create_task(
        _dense_search(vector_store, collection, embedder, query, ann_limit, filters)
    )
    bm25_task = asyncio.create_task(
        _bm25_search(
            query, restaurant_id, db_session, ann_limit, date_from, date_to, rating_min, rating_max
        )
    )

    dense_results, bm25_results = await asyncio.gather(dense_task, bm25_task)

    if not dense_results and not bm25_results:
        return []

    if not dense_results:
        logger.warning("dense_retrieval_unavailable_using_bm25_only", restaurant_id=restaurant_id)

    return _fuse_and_rank(dense_results, bm25_results, top_k=top_k, rrf_k=rrf_k)


async def _dense_search(
    vector_store: BaseVectorStore,
    collection: str,
    embedder: BaseEmbedder,
    query: str,
    limit: int,
    filters: dict,
) -> list[SearchResult]:
    try:
        vector = await embedder.embed_one(query)
        return await vector_store.search(collection, vector, limit=limit, filters=filters)
    except Exception as exc:
        logger.warning("dense_retrieval_failed", error=str(exc))
        return []


async def _bm25_search(
    query: str,
    restaurant_id: int,
    db_session: AsyncSession,
    limit: int,
    date_from: float | None = None,
    date_to: float | None = None,
    rating_min: float | None = None,
    rating_max: float | None = None,
) -> list[SearchResult]:
    try:
        cache = await _get_or_build_bm25(restaurant_id, db_session)
        if cache is None:
            return []

        bm25, corpus = cache
        scores = bm25.get_scores(_tokenize(query))
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[: limit * 2]

        results: list[SearchResult] = []
        for idx, score in indexed:
            if score <= 0:
                continue
            entry = corpus[idx]

            # Post-retrieval date/rating filter (BM25 has no payload filter)
            ts = entry.get("review_date_ts")
            if date_from is not None and ts is not None and ts < date_from:
                continue
            if date_to is not None and ts is not None and ts > date_to:
                continue
            r = entry.get("rating")
            if rating_min is not None and r is not None and r < rating_min:
                continue
            if rating_max is not None and r is not None and r > rating_max:
                continue

            results.append(SearchResult(id=entry["id"], score=float(score), payload=entry))
            if len(results) >= limit:
                break

        return results
    except Exception as exc:
        logger.warning("bm25_search_failed", error=str(exc))
        return []


def _fuse_and_rank(
    dense: list[SearchResult],
    bm25: list[SearchResult],
    top_k: int,
    rrf_k: int = 60,
) -> list[SearchResult]:
    """Merge dense and BM25 results using RRF and return the top_k SearchResults.

    The returned .score field contains the RRF score. When both sources return the
    same chunk_id, the dense payload takes precedence because Qdrant payloads include
    food_entities while the BM25 corpus entry does not.
    """
    rrf_scores: dict[str, float] = {}
    for rank, r in enumerate(dense, start=1):
        rrf_scores[r.id] = rrf_scores.get(r.id, 0.0) + 1.0 / (rrf_k + rank)
    for rank, r in enumerate(bm25, start=1):
        rrf_scores[r.id] = rrf_scores.get(r.id, 0.0) + 1.0 / (rrf_k + rank)

    payload_map: dict[str, dict] = {r.id: r.payload for r in bm25}
    for r in dense:
        payload_map[r.id] = r.payload

    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

    return [
        SearchResult(
            id=chunk_id,
            score=rrf_scores[chunk_id],
            payload=payload_map.get(chunk_id, {}),
        )
        for chunk_id in sorted_ids
    ]
