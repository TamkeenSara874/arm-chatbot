from __future__ import annotations

import asyncio
import functools
import math
from typing import TYPE_CHECKING

import structlog

from src.services.vector.base import SearchResult

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = structlog.get_logger()

_model_cache: dict[str, CrossEncoder] = {}
_load_lock: asyncio.Lock | None = None


def _get_load_lock() -> asyncio.Lock:
    global _load_lock
    if _load_lock is None:
        _load_lock = asyncio.Lock()
    return _load_lock


def is_warmed_up(model_name: str) -> bool:
    """True once load_reranker(model_name) has completed at least once.

    Exposed for /health/ready so a chat query is never the first thing that
    triggers the ~20-30s model download/load.
    """
    return model_name in _model_cache


def _sigmoid(x: float) -> float:
    """Normalize a cross-encoder logit to [0, 1] via sigmoid.

    Clamps the input to [-50, 50] to avoid overflow in math.exp.
    ms-marco-MiniLM-L-6-v2 produces logits roughly in [-10, 10],
    so sigmoid maps -5 -> 0.007 and +5 -> 0.993, which is a meaningful range
    for blending with the recency and rating signals in rank_results().
    """
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, x))))


def _load_cross_encoder(model_name: str) -> CrossEncoder:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


async def load_reranker(model_name: str) -> CrossEncoder:
    """Return a cached CrossEncoder, downloading and loading it on first call.

    Thread-safe via asyncio.Lock. The CPU-bound model load runs in a thread
    executor to avoid blocking the event loop.
    """
    if model_name in _model_cache:
        return _model_cache[model_name]
    async with _get_load_lock():
        if model_name in _model_cache:
            return _model_cache[model_name]
        loop = asyncio.get_event_loop()
        model = await loop.run_in_executor(None, _load_cross_encoder, model_name)
        _model_cache[model_name] = model
        logger.info("reranker_loaded", model=model_name)
        return model


async def rerank(
    query: str,
    results: list[SearchResult],
    model_name: str,
    top_k: int | None = None,
) -> list[SearchResult]:
    """Score (query, chunk) pairs with a cross-encoder and return top_k results.

    Sigmoid-normalizes raw cross-encoder logits to [0, 1] so they can serve as
    the semantic relevance signal in the rank_results() composite formula alongside
    recency and rating signals.

    Falls back to the original RRF ordering if the model fails.
    Runs the synchronous CrossEncoder.predict() in a thread executor.
    """
    if not results:
        return results

    import time

    from src.utils.metrics import rerank_latency

    model = await load_reranker(model_name)
    pairs = [(query, r.payload.get("text", "")) for r in results]

    try:
        t0 = time.perf_counter()
        loop = asyncio.get_event_loop()
        raw_scores: list[float] = await loop.run_in_executor(
            None, functools.partial(model.predict, pairs)
        )
        elapsed = time.perf_counter() - t0
        rerank_latency.observe(elapsed)
        logger.debug(
            "reranker_scored",
            candidates=len(results),
            elapsed_ms=round(elapsed * 1000, 1),
        )
    except Exception as exc:
        logger.warning("reranker_failed_falling_back_to_rrf_order", error=str(exc))
        return results[:top_k] if top_k else results

    reranked = [
        SearchResult(id=r.id, score=_sigmoid(float(s)), payload=r.payload)
        for r, s in zip(results, raw_scores, strict=True)
    ]
    reranked.sort(key=lambda x: x.score, reverse=True)
    return reranked[:top_k] if top_k else reranked
