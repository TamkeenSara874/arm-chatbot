from __future__ import annotations

import json
import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime

import structlog

logger = structlog.get_logger()


@dataclass
class RequestTrace:
    """Per-request trace capturing stage latencies, token usage, cost, and quality signals.

    Call emit() at the end of each request to flush the structured log and
    Prometheus metrics. Designed to be created in the route handler and passed
    through the pipeline as a mutable context object.
    """

    session_id: str
    restaurant_id: int
    intent: str = ""
    complexity: str = ""

    decomp_ms: float = 0.0
    retrieval_ms: float = 0.0
    embed_ms: float = 0.0
    search_ms: float = 0.0
    rerank_ms: float = 0.0
    ranking_ms: float = 0.0
    generation_ms: float = 0.0

    decomp_model: str = ""
    generation_model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0

    confidence: float = 0.0
    evidence_count: int = 0
    low_evidence: bool = False
    cache_hit: bool = False
    groundedness_ok: bool = True

    @property
    def total_ms(self) -> float:
        return (
            self.decomp_ms
            + self.retrieval_ms
            + self.rerank_ms
            + self.ranking_ms
            + self.generation_ms
        )

    def record_tokens(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int = 0,
        cached_tokens: int = 0,
    ) -> None:
        """Accumulate token counts and update the running cost estimate.

        cached_tokens (a subset of prompt_tokens OpenAI's automatic prompt
        caching served from cache) is tracked separately for observability
        only -- it doesn't yet feed estimate_cost, since a per-model cached
        discount isn't wired up there.
        """
        from src.utils.cost_tracker import estimate_cost

        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.cached_tokens += cached_tokens
        self.cost_usd += estimate_cost(model, prompt_tokens, completion_tokens)

    def emit(self) -> None:
        """Write a structured log entry and record Prometheus stage metrics."""
        from src.utils.metrics import pipeline_stage_latency, request_cost_usd

        for stage, ms in [
            ("decomp", self.decomp_ms),
            ("retrieval", self.retrieval_ms),
            ("rerank", self.rerank_ms),
            ("ranking", self.ranking_ms),
            ("generation", self.generation_ms),
        ]:
            if ms > 0:
                pipeline_stage_latency.labels(stage=stage).observe(ms / 1000.0)

        if self.cost_usd > 0:
            request_cost_usd.observe(self.cost_usd)

        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "session_id": self.session_id,
            "restaurant_id": self.restaurant_id,
            "intent": self.intent,
            "complexity": self.complexity,
            "decomp_ms": round(self.decomp_ms, 1),
            "retrieval_ms": round(self.retrieval_ms, 1),
            "embed_ms": round(self.embed_ms, 1),
            "search_ms": round(self.search_ms, 1),
            "rerank_ms": round(self.rerank_ms, 1),
            "ranking_ms": round(self.ranking_ms, 1),
            "generation_ms": round(self.generation_ms, 1),
            "total_ms": round(self.total_ms, 1),
            "cache_hit": self.cache_hit,
            "decomp_model": self.decomp_model,
            "generation_model": self.generation_model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "confidence": round(self.confidence, 3),
            "evidence_count": self.evidence_count,
            "low_evidence": self.low_evidence,
            "groundedness_ok": self.groundedness_ok,
        }

        logger.info("request_trace", **record)

        try:
            os.makedirs("logs", exist_ok=True)
            with open("logs/request_traces.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError:
            pass


@dataclass
class IngestTrace:
    """Per-ingest-job trace capturing stage latencies, token usage, and cost.

    Mirrors RequestTrace's emit() pattern so seed/upload ingestion runs get
    the same structured-log + JSONL observability chat requests get, per the
    "log cost/latency/accuracy for every seed ingestion end to end" requirement.
    """

    job_id: str
    restaurant_id: int

    entity_extraction_ms: float = 0.0
    embedding_upsert_ms: float = 0.0

    entity_model: str = ""
    embedding_model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0
    cost_usd: float = 0.0

    total_reviews: int = 0
    total_chunks: int = 0
    skipped_empty: int = 0
    skipped_already_processed: int = 0

    @property
    def total_ms(self) -> float:
        return self.entity_extraction_ms + self.embedding_upsert_ms

    def record_entity_tokens(
        self, model: str, prompt_tokens: int, completion_tokens: int = 0
    ) -> None:
        from src.utils.cost_tracker import estimate_cost

        self.entity_model = model
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.cost_usd += estimate_cost(model, prompt_tokens, completion_tokens)

    def record_embedding_tokens(self, model: str, total_tokens: int) -> None:
        from src.utils.cost_tracker import estimate_cost

        self.embedding_model = model
        self.embedding_tokens += total_tokens
        self.cost_usd += estimate_cost(model, total_tokens, 0)

    def emit(self) -> None:
        """Write a structured log entry for the completed ingest job."""
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "job_id": self.job_id,
            "restaurant_id": self.restaurant_id,
            "entity_extraction_ms": round(self.entity_extraction_ms, 1),
            "embedding_upsert_ms": round(self.embedding_upsert_ms, 1),
            "total_ms": round(self.total_ms, 1),
            "entity_model": self.entity_model,
            "embedding_model": self.embedding_model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "embedding_tokens": self.embedding_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "total_reviews": self.total_reviews,
            "total_chunks": self.total_chunks,
            "skipped_empty": self.skipped_empty,
            "skipped_already_processed": self.skipped_already_processed,
        }

        logger.info("ingest_trace", **record)

        try:
            os.makedirs("logs", exist_ok=True)
            with open("logs/ingest_traces.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError:
            pass


@contextmanager
def timed(trace: RequestTrace, field_name: str) -> Generator[None, None, None]:
    """Time a block of code and store the elapsed milliseconds into trace.<field_name>."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        setattr(trace, field_name, (time.perf_counter() - t0) * 1000.0)
