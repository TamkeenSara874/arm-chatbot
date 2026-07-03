from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from fastembed import SparseTextEmbedding as _SparseModel

logger = structlog.get_logger()

_MODEL_NAME = "Qdrant/bm25"
_model_instance: _SparseModel | None = None
_model_lock: asyncio.Lock | None = None


def _get_model_lock() -> asyncio.Lock:
    global _model_lock
    if _model_lock is None:
        _model_lock = asyncio.Lock()
    return _model_lock


@dataclass(frozen=True, slots=True)
class SparseVector:
    indices: list[int]
    values: list[float]


async def _get_model() -> _SparseModel:
    global _model_instance
    if _model_instance is not None:
        return _model_instance
    async with _get_model_lock():
        if _model_instance is not None:
            return _model_instance
        loop = asyncio.get_running_loop()
        _model_instance = await loop.run_in_executor(None, _load_model)
        logger.info("sparse_embedder_loaded", model=_MODEL_NAME)
        return _model_instance


def _load_model() -> _SparseModel:
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(model_name=_MODEL_NAME)


async def warmup_sparse_embedder() -> None:
    """Pre-load the sparse embedding model at startup to avoid first-request delay."""
    await _get_model()


def is_warmed_up() -> bool:
    """True once warmup_sparse_embedder() (or any embed call) has completed."""
    return _model_instance is not None


async def compute_sparse_vector(text: str) -> SparseVector:
    """Return a BM25-style sparse vector for a single text string."""
    model = await _get_model()
    loop = asyncio.get_running_loop()

    def _embed() -> SparseVector:
        result = list(model.embed([text]))[0]
        return SparseVector(indices=result.indices.tolist(), values=result.values.tolist())

    return await loop.run_in_executor(None, _embed)


async def compute_sparse_vectors_batch(texts: list[str]) -> list[SparseVector]:
    """Return BM25-style sparse vectors for a batch of texts."""
    if not texts:
        return []
    model = await _get_model()
    loop = asyncio.get_running_loop()

    def _embed_batch() -> list[SparseVector]:
        results = list(model.embed(texts))
        return [SparseVector(indices=r.indices.tolist(), values=r.values.tolist()) for r in results]

    return await loop.run_in_executor(None, _embed_batch)
