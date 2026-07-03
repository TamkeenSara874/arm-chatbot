# Evaluation Report

## Methodology

**Latency/cost numbers** are drawn from `logs/request_traces.jsonl` (121 real requests, both from the automated LLM-as-judge harness and manual smoke testing) and `logs/ingest_traces.jsonl`, both emitted by `RequestTrace`/`IngestTrace` (`src/utils/tracing.py`) on every chat query and ingest run — not synthetic numbers. `cost_usd` reflects real provider-reported token usage (`stream_options={"include_usage": true}` for streaming calls, `response.usage` for non-streaming), not a character-count estimate.

**Caveat on absolute latency:** these traces were captured on a single local Docker Desktop stack (Windows + WSL2) while the same machine was simultaneously running `npm install`, Docker image builds, and multiple background test/debug processes — not an isolated benchmark environment. Retrieval latency in particular (cross-encoder reranking is CPU-bound) is almost certainly inflated by that contention. Treat the relative breakdown (which stage dominates) as reliable; treat absolute numbers as an upper bound, not a clean baseline. A follow-up benchmark on an idle machine (or a dedicated CI runner) would give a tighter number.

**Retrieval/answer quality** comes from `tests/e2e/test_eval_fixture.py`, an LLM-as-judge harness driven by `tests/fixtures/rag_chatbot_eval_fixture.json` (26 hand-written test cases spanning guardrails, count queries, reports, simple/complex generation, date/rating filters, sentiment-rating conflicts, multi-tenancy, session context, corrections, prompt injection, staleness, zero-data, cache correctness, and retrieval fallback). Each case is graded two ways: mechanical checks against fields the API actually returns (model routing, cost, evidence ratings — free and exact), and an LLM-judge call (`gpt-4o-mini`) grading the fixture's natural-language assertions against the real answer text (the only part that spends OpenAI budget, kept deliberately small).

## Query-time latency (n=121 real requests)

| Path | n | p50 | p90 | Notes |
|---|---|---|---|---|
| Cache hit | 31 | 0ms (pipeline) | 0ms (pipeline) | Skips the entire pipeline; only network/serialization time remains |
| Guardrail / fast-path (no generation) | 29 | ~700ms | ~1.4s | Single Groq decomposition call, no retrieval or generation |
| Full pipeline (retrieval + generation), non-cached | 37 | | | |
| — retrieval stage | | 13.4s | 25.6s | Includes cross-encoder reranking; see contention caveat above |
| — generation stage | | 4.6s | 7.4s | Streamed, `gpt-4o-mini` (simple) / `gpt-4.1` (complex) |
| — decomposition stage | 90 | 0.86s | 1.5s | Groq `llama-3.3-70b-versatile` |

**Read this as:** the guardrail/count-query/cache fast paths (the majority of realistic traffic for a review-analytics bot — "how many reviews," repeated questions, out-of-scope declines) are sub-1.5-second and free. Retrieval is the dominant cost in the full-generation path; a clean re-benchmark is the next step to separate genuine reranker cost from dev-machine contention.

## Cost (n=53 requests with non-zero LLM cost)

- Total measured: **$0.093** across 53 billed requests, **$0.00176 average per billed request**, **$0.0145 max** (a complex `gpt-4.1` query).
- 68 of 121 requests (56%) cost **$0.00** — cache hits, guardrail declines, and `count_query` fast paths never touch a paid LLM call.
- Ingestion cost model (from `IngestTrace`, unit-validated but not yet exercised on the full ~2,753-review dataset since the tracer was added after the last full ingest run): entity extraction (`gpt-4o-mini`, ~$0.15/$0.60 per 1M in/out tokens) + embedding (`text-embedding-3-large`, $0.13/1M tokens). At the dataset's scale this is projected at well under $1 total for a full re-ingest — recommend running `docker compose run --rm seed python scripts/seed.py --force` once to capture the real number in `logs/ingest_traces.jsonl`.

## Retrieval / answer quality (LLM-as-judge harness)

- **18 of 20 runnable test cases passed** on the most recent full run (2 additional fixture cases — `SC-02`, a 50+ turn session, and `RB-01`, a simulated Qdrant outage — are explicitly skipped with a documented reason rather than faked; see `tests/e2e/test_eval_fixture.py:SKIPPED_TEST_IDS`).
- **Rating filter accuracy: 100%** (mechanically verified, not judged) — every cited review for "5-star reviews mentioning ambiance" had `rating=5`; every cited review for "complaints under 3 stars" had `rating<3`, across all evidence returned in testing.
- **Groundedness heuristic: 0 overclaims flagged** across all 121 traced requests (`groundedness_ok=true` on every row) — no answer stated a review/mention count exceeding the retrieved evidence size.
- **Zero-data handling:** a never-ingested restaurant_id correctly triggers the hard hallucination gate (`model_used="no_evidence_gate"`) rather than generating a plausible-sounding but fabricated answer.
- **Correction flow:** verified end-to-end (submit a correction → re-ask the same question → answer reflects the correction) after fixing a real bug this harness caught (correction intent was hardcoded to `"factual"`, causing `find_correction()`'s intent cross-check to reject almost every correction even though it was stored correctly).
- Building and running this harness against the live stack surfaced **6 previously-hidden production bugs**, all now fixed and re-verified: a FastAPI/decorator incompatibility that 422'd every real chat/correction/report request, a nonexistent Qdrant health-check method, a deprecated Qdrant search API breaking corrections and session-memory recall, a DB-session race in background tasks that intermittently dropped message/cache writes, the correction-intent bug above, and cache hits misreporting `complexity` regardless of what actually generated the cached answer.

## What's not covered yet

- No held-out, human-labeled relevance judgments (precision@k / NDCG in the classical IR sense) — the dataset has no ground-truth "relevant chunks per query" labels to compute against. The groundedness heuristic and LLM-judge assertions are the closest proxy currently in place.
- Overall code coverage is ~54% (CI gate set at a 50% floor, documented in `.github/workflows/ci.yml`); `src/api/routes/*` and `src/workers/ingest_worker.py` are exercised by the e2e harness above but not by fast unit tests yet.
