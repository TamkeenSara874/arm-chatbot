"""Unit tests for cost estimation and per-request tracing infrastructure."""

from src.utils.cost_tracker import estimate_cost
from src.utils.tracing import IngestTrace, RequestTrace, timed


class TestEstimateCost:
    def test_gpt41_cost(self) -> None:
        cost = estimate_cost("gpt-4.1", prompt_tokens=1000, completion_tokens=500)
        expected = (1000 * 2.00 + 500 * 8.00) / 1_000_000
        assert abs(cost - expected) < 1e-9

    def test_gpt4o_mini_cost(self) -> None:
        cost = estimate_cost("gpt-4o-mini", prompt_tokens=2000, completion_tokens=200)
        expected = (2000 * 0.15 + 200 * 0.60) / 1_000_000
        assert abs(cost - expected) < 1e-9

    def test_embedding_has_no_output_cost(self) -> None:
        cost = estimate_cost("text-embedding-3-large", prompt_tokens=500)
        assert cost == 500 * 0.13 / 1_000_000

    def test_groq_free_tier_is_zero(self) -> None:
        cost = estimate_cost("llama-3.3-70b-versatile", prompt_tokens=5000, completion_tokens=500)
        assert cost == 0.0

    def test_unknown_model_returns_zero(self) -> None:
        cost = estimate_cost("some-unknown-model", prompt_tokens=1000, completion_tokens=100)
        assert cost == 0.0

    def test_zero_tokens_returns_zero(self) -> None:
        assert estimate_cost("gpt-4.1", 0, 0) == 0.0


class TestRequestTrace:
    def test_total_ms_sums_all_stages(self) -> None:
        trace = RequestTrace(session_id="s1", restaurant_id=1)
        trace.decomp_ms = 100.0
        trace.retrieval_ms = 200.0
        trace.rerank_ms = 50.0
        trace.ranking_ms = 10.0
        trace.generation_ms = 300.0
        assert trace.total_ms == 660.0

    def test_record_tokens_accumulates_cost(self) -> None:
        trace = RequestTrace(session_id="s1", restaurant_id=1)
        trace.record_tokens("gpt-4.1", prompt_tokens=1000, completion_tokens=200)
        expected = (1000 * 2.00 + 200 * 8.00) / 1_000_000
        assert abs(trace.cost_usd - expected) < 1e-9
        assert trace.prompt_tokens == 1000
        assert trace.completion_tokens == 200

    def test_record_tokens_multiple_calls_accumulate(self) -> None:
        trace = RequestTrace(session_id="s1", restaurant_id=1)
        trace.record_tokens("gpt-4o-mini", prompt_tokens=500, completion_tokens=100)
        trace.record_tokens("text-embedding-3-large", prompt_tokens=200)
        assert trace.prompt_tokens == 700
        assert trace.completion_tokens == 100
        assert trace.cost_usd > 0

    def test_timed_context_manager_records_elapsed_ms(self) -> None:
        trace = RequestTrace(session_id="s1", restaurant_id=1)
        with timed(trace, "decomp_ms"):
            pass
        assert trace.decomp_ms >= 0.0

    def test_timed_records_to_correct_field(self) -> None:
        trace = RequestTrace(session_id="s1", restaurant_id=1)
        with timed(trace, "generation_ms"):
            _ = sum(range(10_000))
        assert trace.generation_ms > 0.0
        assert trace.decomp_ms == 0.0

    def test_emit_does_not_raise(self) -> None:
        trace = RequestTrace(session_id="s1", restaurant_id=1)
        trace.intent = "factual"
        trace.complexity = "simple"
        trace.decomp_ms = 120.0
        trace.generation_ms = 800.0
        trace.record_tokens("gpt-4o-mini", 300, 80)
        trace.confidence = 0.88
        trace.evidence_count = 4
        trace.emit()

    def test_groundedness_ok_defaults_true(self) -> None:
        trace = RequestTrace(session_id="s1", restaurant_id=1)
        assert trace.groundedness_ok is True


class TestIngestTrace:
    def test_total_ms_sums_stages(self) -> None:
        trace = IngestTrace(job_id="j1", restaurant_id=1)
        trace.entity_extraction_ms = 100.0
        trace.embedding_upsert_ms = 200.0
        assert trace.total_ms == 300.0

    def test_record_entity_tokens_accumulates_cost(self) -> None:
        trace = IngestTrace(job_id="j1", restaurant_id=1)
        trace.record_entity_tokens("gpt-4o-mini", prompt_tokens=1000, completion_tokens=200)
        expected = (1000 * 0.15 + 200 * 0.60) / 1_000_000
        assert abs(trace.cost_usd - expected) < 1e-9
        assert trace.prompt_tokens == 1000
        assert trace.completion_tokens == 200
        assert trace.entity_model == "gpt-4o-mini"

    def test_record_embedding_tokens_accumulates_cost(self) -> None:
        trace = IngestTrace(job_id="j1", restaurant_id=1)
        trace.record_embedding_tokens("text-embedding-3-large", total_tokens=5000)
        expected = 5000 * 0.13 / 1_000_000
        assert abs(trace.cost_usd - expected) < 1e-9
        assert trace.embedding_tokens == 5000
        assert trace.embedding_model == "text-embedding-3-large"

    def test_costs_from_both_sources_accumulate(self) -> None:
        trace = IngestTrace(job_id="j1", restaurant_id=1)
        trace.record_entity_tokens("gpt-4o-mini", 1000, 200)
        trace.record_embedding_tokens("text-embedding-3-large", 5000)
        expected = (1000 * 0.15 + 200 * 0.60) / 1_000_000 + 5000 * 0.13 / 1_000_000
        assert abs(trace.cost_usd - expected) < 1e-9

    def test_emit_does_not_raise(self) -> None:
        trace = IngestTrace(job_id="j1", restaurant_id=1)
        trace.entity_extraction_ms = 500.0
        trace.embedding_upsert_ms = 1500.0
        trace.record_entity_tokens("gpt-4o-mini", 1000, 200)
        trace.record_embedding_tokens("text-embedding-3-large", 5000)
        trace.total_reviews = 100
        trace.total_chunks = 250
        trace.skipped_empty = 3
        trace.emit()
