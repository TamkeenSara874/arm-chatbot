from __future__ import annotations

import asyncio
import functools
import math
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from src.services.vector.base import SearchResult

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

    from src.core.retrieval import RetrievalTiming

logger = structlog.get_logger()

# Sibling files export_dynamic_quantized_onnx_model() does NOT copy into the
# export directory -- confirmed empirically it writes only the .onnx weights
# file, leaving the directory unloadable on its own (AutoConfig can't find a
# model_type). model.safetensors is deliberately excluded: the onnx backend
# never reads the original FP32 weights, only the exported .onnx file.
_ONNX_EXPORT_SIBLING_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
)

# Immutable commit for cross-encoder/ms-marco-MiniLM-L6-v2 (config.py's
# default reranker_model), used to pin the snapshot_download() call in
# _ensure_quantized_export() below -- see the comment at that call site.
_RERANKER_SNAPSHOT_REVISION = "c5ee24cb16019beea0893ab7796b1df96625c6b8"

_model_cache: dict[str, CrossEncoder] = {}
_load_lock: asyncio.Lock | None = None

# The cross-encoder is trained for direct question/passage relevance
# (MS MARCO style): it has no real signal for "is this one review evidence
# for a broad theme" and produces near-identical, deeply negative logits for
# every candidate on genuinely broad/meta questions (e.g. "summarize the
# negative reviews") regardless of actual content -- confirmed empirically:
# real discrimination shows a raw-logit spread of several points or more
# (e.g. -11.0 to +5.1, or -11.4 to -6.8), while true degenerate collapse
# shows a spread under ~0.2. This threshold sits comfortably between the two.
_DEGENERATE_LOGIT_SPREAD = 1.0


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
    ms-marco-MiniLM-L6-v2 produces logits roughly in [-10, 10],
    so sigmoid maps -5 -> 0.007 and +5 -> 0.993, which is a meaningful range
    for blending with the recency and rating signals in rank_results().
    """
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, x))))


def _load_cross_encoder(model_name: str) -> CrossEncoder:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def _quantized_export_dir(model_name: str, quantization_config: str) -> Path:
    """Where a quantized export for (model_name, quantization_config) lives.

    Inside the existing hf-cache Docker volume (docker-compose.yml already
    mounts this so the reranker's downloaded weights survive container
    recreation) -- the quantized export belongs in the same place, produced
    once, not re-exported on every cold start.
    """
    from huggingface_hub import constants

    hf_home = Path(constants.HF_HOME)
    return hf_home / "onnx-quantized" / model_name.replace("/", "__") / quantization_config


def _quantized_file_name(quantization_config: str) -> str:
    return f"model_quint8_{quantization_config}.onnx"


def _ensure_quantized_export(model_name: str, quantization_config: str, export_dir: Path) -> None:
    """Export a quantized ONNX copy of model_name into export_dir, if not already done."""
    from sentence_transformers import CrossEncoder
    from sentence_transformers.backend import export_dynamic_quantized_onnx_model

    quantized_file = export_dir / "onnx" / _quantized_file_name(quantization_config)
    if quantized_file.exists():
        return

    export_dir.mkdir(parents=True, exist_ok=True)
    base_model = CrossEncoder(model_name, backend="onnx")
    export_dynamic_quantized_onnx_model(base_model, quantization_config, str(export_dir))

    # export_dynamic_quantized_onnx_model() only writes the .onnx weights
    # file -- copy the small config/tokenizer siblings from the model's own
    # downloaded snapshot so export_dir is a self-contained, independently
    # loadable model directory (confirmed empirically this step is required;
    # loading straight from export_dir without it fails AutoConfig resolution).
    from huggingface_hub import snapshot_download

    # Pinned to an immutable commit, not the "main" branch pointer -- a repo
    # push otherwise could silently swap what gets downloaded here (bandit
    # B615). Deliberately unconditional: if reranker_model is ever pointed at
    # a different repo, this pin fails to resolve with a loud, visible error
    # instead of silently trusting whatever's on that other repo's main
    # branch -- update _RERANKER_SNAPSHOT_REVISION alongside the model.
    snapshot_dir = Path(snapshot_download(model_name, revision=_RERANKER_SNAPSHOT_REVISION))
    for name in _ONNX_EXPORT_SIBLING_FILES:
        src = snapshot_dir / name
        if src.exists():
            shutil.copy(src, export_dir / name)


def _load_cross_encoder_onnx_quantized(model_name: str, quantization_config: str) -> CrossEncoder:
    from sentence_transformers import CrossEncoder

    export_dir = _quantized_export_dir(model_name, quantization_config)
    _ensure_quantized_export(model_name, quantization_config, export_dir)
    return CrossEncoder(
        str(export_dir),
        backend="onnx",
        model_kwargs={"file_name": _quantized_file_name(quantization_config)},
    )


async def load_reranker(model_name: str) -> CrossEncoder:
    """Return a cached CrossEncoder, downloading and loading it on first call.

    Thread-safe via asyncio.Lock. The CPU-bound model load (and, if enabled,
    the one-time quantized export) runs in a thread executor to avoid
    blocking the event loop. Reads reranker_onnx_quantized/
    reranker_onnx_quantization_config from Settings directly (matching how
    other modules, e.g. src/api/routes/chat.py, call get_settings() at point
    of use rather than threading Settings through every function) -- so this
    function's signature and every existing call site stay unchanged.
    """
    if model_name in _model_cache:
        return _model_cache[model_name]
    async with _get_load_lock():
        if model_name in _model_cache:
            return _model_cache[model_name]

        from src.config import get_settings

        settings = get_settings()
        loop = asyncio.get_event_loop()
        if settings.reranker_onnx_quantized:
            loader = functools.partial(
                _load_cross_encoder_onnx_quantized,
                model_name,
                settings.reranker_onnx_quantization_config,
            )
        else:
            loader = functools.partial(_load_cross_encoder, model_name)
        model = await loop.run_in_executor(None, loader)
        _model_cache[model_name] = model
        logger.info(
            "reranker_loaded", model=model_name, onnx_quantized=settings.reranker_onnx_quantized
        )
        return model


async def _predict_raw(query: str, texts: list[str], model_name: str) -> list[float]:
    """Raw (pre-sigmoid) cross-encoder logits for (query, text) pairs.

    Shared by rerank() and score_for_highlight() so both reuse the same
    warm, local model and thread-executor dispatch -- no network I/O either
    way, unlike the OpenAI-embedding approach highlighting used previously.
    """
    from src.config import get_settings

    model = await load_reranker(model_name)
    pairs = [(query, t) for t in texts]
    batch_size = get_settings().reranker_batch_size
    loop = asyncio.get_event_loop()
    raw_scores: list[float] = await loop.run_in_executor(
        None, functools.partial(model.predict, pairs, batch_size=batch_size)
    )
    return raw_scores


async def score_for_highlight(query: str, sentences: list[str], model_name: str) -> list[float]:
    """Sigmoid-normalized cross-encoder relevance scores for (query, sentence) pairs.

    Used by ranking.py to pick which sentence within a review snippet to
    highlight. Reuses rerank()'s already-warm local model instead of a fresh
    OpenAI embedding call -- measured at 2-7s/request on real evidence sets
    before this change, since that call is a real network round-trip; this
    is local CPU inference with no I/O. Unlike rerank(), there's no
    degenerate-spread fallback: a flat/unhelpful score still just picks
    *some* sentence rather than corrupting a whole candidate ranking, so
    there's nothing worth falling back from. Returns [] (no highlight) if
    the model itself fails to load or predict.
    """
    if not sentences:
        return []
    try:
        raw_scores = await _predict_raw(query, sentences, model_name)
    except Exception as exc:
        logger.warning("highlight_scoring_failed", error=str(exc))
        return []
    return [_sigmoid(float(s)) for s in raw_scores]


async def rerank(
    query: str,
    results: list[SearchResult],
    model_name: str,
    top_k: int | None = None,
    timing: RetrievalTiming | None = None,
) -> list[SearchResult]:
    """Score (query, chunk) pairs with a cross-encoder and return top_k results.

    Sigmoid-normalizes raw cross-encoder logits to [0, 1] so they can serve as
    the semantic relevance signal in the rank_results() composite formula alongside
    recency and rating signals.

    Falls back to the original RRF ordering if the model fails.
    Runs the synchronous CrossEncoder.predict() in a thread executor.

    timing, if provided, has its .reranked flag set to False on either
    fallback path below -- see RetrievalTiming's own docstring for why a
    caller displaying relevance needs to know this.
    """
    if not results:
        return results

    import time

    from src.utils.metrics import rerank_latency

    texts = [r.payload.get("text", "") for r in results]

    try:
        t0 = time.perf_counter()
        raw_scores = await _predict_raw(query, texts, model_name)
        elapsed = time.perf_counter() - t0
        rerank_latency.observe(elapsed)
        logger.debug(
            "reranker_scored",
            candidates=len(results),
            elapsed_ms=round(elapsed * 1000, 1),
        )
    except Exception as exc:
        logger.warning("reranker_failed_falling_back_to_rrf_order", error=str(exc))
        if timing is not None:
            timing.reranked = False
        return results[:top_k] if top_k else results

    # A single candidate trivially has zero spread -- that's not the model
    # failing to discriminate, there's nothing to discriminate between.
    spread = max(raw_scores) - min(raw_scores) if len(raw_scores) > 1 else None
    if spread is not None and spread < _DEGENERATE_LOGIT_SPREAD:
        # The model isn't discriminating between candidates at all for this
        # query -- its (compressed-near-zero) sigmoid scores would be a worse
        # relevance signal than the retrieval step's own fusion score, and
        # would render as a wall of "0% match" badges. Keep the pre-rerank
        # (Qdrant-native hybrid fusion) order and scores instead.
        logger.info(
            "reranker_degenerate_falling_back_to_native_order",
            spread=round(float(spread), 3),
            candidates=len(results),
        )
        if timing is not None:
            timing.reranked = False
        return results[:top_k] if top_k else results

    reranked = [
        SearchResult(id=r.id, score=_sigmoid(float(s)), payload=r.payload)
        for r, s in zip(results, raw_scores, strict=True)
    ]
    reranked.sort(key=lambda x: x.score, reverse=True)
    return reranked[:top_k] if top_k else reranked
