from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass

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
    rerank_ms: float = 0.0
    ranking_ms: float = 0.0
    generation_ms: float = 0.0

    decomp_model: str = ""
    generation_model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    confidence: float = 0.0
    evidence_count: int = 0
    low_evidence: bool = False
    cache_hit: bool = False

    @property
    def total_ms(self) -> float:
        return (
            self.decomp_ms
            + self.retrieval_ms
            + self.rerank_ms
            + self.ranking_ms
            + self.generation_ms
        )

    def record_tokens(self, model: str, prompt_tokens: int, completion_tokens: int = 0) -> None:
        """Accumulate token counts and update the running cost estimate."""
        from src.utils.cost_tracker import estimate_cost

        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
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

        logger.info(
            "request_trace",
            session_id=self.session_id,
            restaurant_id=self.restaurant_id,
            intent=self.intent,
            complexity=self.complexity,
            decomp_ms=round(self.decomp_ms, 1),
            retrieval_ms=round(self.retrieval_ms, 1),
            rerank_ms=round(self.rerank_ms, 1),
            ranking_ms=round(self.ranking_ms, 1),
            generation_ms=round(self.generation_ms, 1),
            total_ms=round(self.total_ms, 1),
            cache_hit=self.cache_hit,
            decomp_model=self.decomp_model,
            generation_model=self.generation_model,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            cost_usd=round(self.cost_usd, 6),
            confidence=round(self.confidence, 3),
            evidence_count=self.evidence_count,
            low_evidence=self.low_evidence,
        )


@contextmanager
def timed(trace: RequestTrace, field_name: str) -> Generator[None, None, None]:
    """Time a block of code and store the elapsed milliseconds into trace.<field_name>."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        setattr(trace, field_name, (time.perf_counter() - t0) * 1000.0)
