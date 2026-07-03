# Architecture Notes

See [`architecture_diagram.html`](architecture_diagram.html) for the full visual diagram. This document covers the key design decisions and tradeoffs in bullet form.

## Stack and why

- **FastAPI + Postgres + Qdrant + Redis + React/Vite.** Async throughout since this is an I/O-bound AI system (LLM calls, vector search, DB queries) — see `docs/architecture_diagram.html` for the request-level flow.
- **Qdrant hybrid search (dense + native sparse), not a separate BM25 library.** Dense (`text-embedding-3-large`, 3072-dim) and sparse (fastembed `Qdrant/bm25`) vectors are stored as two named vectors on the same collection, fused server-side via `query_points()` with RRF. This replaced an earlier in-process `rank-bm25` index, which had a hard single-worker constraint (each worker held its own index, giving non-deterministic rankings under `--workers > 1`). The native approach removes that constraint entirely — sparse ranking is now handled by Qdrant itself, consistent across any number of API workers.
- **Groq (`llama-3.3-70b-versatile`) for decomposition, OpenAI for generation.** Groq's free tier is fast enough for the classification/decomposition step (which doesn't need top-tier reasoning) and costs nothing; the two generation models (`gpt-4o-mini` simple, `gpt-4.1` complex) are the ones actually producing the user-facing answer, where quality matters more than cost per call.
- **Cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`), warmed up at startup.** Both the reranker and the sparse-embedding model are loaded during FastAPI's `lifespan()` startup, which blocks until complete — `uvicorn` does not start accepting connections until warmup finishes. This means the model load happens once per container start, never on a user's first chat message. Confirmed via `/health/ready`'s `reranker_warmed_up`/`sparse_embedder_warmed_up` fields. Originally `BAAI/bge-reranker-base` (278M params); live per-stage timing (`retrieval_breakdown` log) showed it taking 13-22s of CPU inference for just 24-30 candidates — the dominant end-to-end latency cost by a wide margin — so it was replaced with MiniLM-L-6-v2 (~22M params), a much smaller model purpose-built for reranking.

## Chunking strategy

- Reviews ≤256 tokens become a single chunk; longer reviews use a sliding window (32-token overlap) on sentence boundaries (NLTK `sent_tokenize`), not raw character splitting. Reviews are short-form text, so most stay as single chunks — chunking exists for the minority of long, multi-topic reviews where returning a whole review as one chunk would dilute a specific complaint/praise with unrelated content.

## Guardrail and intent design

- Guardrail is a **cheap, code-only lookup**, not an LLM call: decomposition classifies intent once (Groq), then `check_guardrail(intent)` does a dict lookup. Guardrailed intents (`out_of_scope`, `manipulation_request`, `multi_location`, `allergen`) return a canned decline with zero additional cost.
- **`ui_question` vs `report_howto` are deliberately split.** Generic platform/dashboard navigation questions ("how do I log in?", "why is this icon red?") redirect to AIO's CareBot, since this bot has no visibility into the platform's UI. But "how do I download my report?" is answered directly with real instructions (Report button → Download PDF) — it's a feature this app actually has, so declining or redirecting it would be a worse user experience than just answering.
- `DecomposedQuery.intent` is a `Literal[...]` of the exact intent set the decomposition prompt documents, not a bare `str`. A hallucinated/malformed intent fails Pydantic validation, which triggers `decompose_query()`'s retry-then-safe-fallback (defaults to `factual`) instead of silently bypassing the guardrail because the intent string didn't match any known guardrail key.

## Evidence ranking and hallucination control

- Composite score = `injection_penalty * (w_rrf*rrf + w_recency*recency_decay + w_rating*effective_rating)`. `effective_rating` uses the pre-computed sentiment label instead of the raw star rating whenever they disagree (`sentiment_rating_agree=False`), so a sarcastic 5-star complaint doesn't rank as if it were genuinely positive.
- **Hard hallucination gate:** if retrieval returns zero evidence, the pipeline never calls the generation LLM at all — it returns a canned "I couldn't find any reviews matching that" answer. This is a stronger guarantee than prompt instructions alone ("never fabricate" is a soft constraint the model can still violate under real traffic).
- **Groundedness heuristic (`src/core/groundedness.py`):** a regex-based, code-only check that flags an answer if it states a review/mention count higher than the number of evidence chunks actually retrieved (excluding star ratings and years, which coincidentally look like counts). This runs on every response with zero added cost/latency and discounts `confidence` when triggered — it's a proxy for "did the model overclaim," not a full accuracy judge.
- **Compound queries** (a countable half + a generative half, e.g. "how many positive reviews do I have, and how can I improve?") always route through the complex prompt with the DB-exact count passed in verbatim (`exact_count`), so the model states a real number instead of trying to (mis)count evidence chunks itself.

## Cost and latency controls

- **Real token-based costs, not estimates.** Every LLM/embedding call requests real usage (`stream_options={"include_usage": true}` for streaming, `response.usage` for non-streaming) and reports it via `RequestTrace.record_tokens()` / `IngestTrace.record_entity_tokens()` / `record_embedding_tokens()`. `cost_usd` in every trace and every ingest run reflects actual provider-billed tokens against `MODEL_COSTS_USD_PER_1M`, not a character-count approximation.
- **Fast paths bypass generation entirely:** `count_query` intent hits a direct Postgres `COUNT(*)` (target <100ms, $0); the guardrail path costs $0 beyond the one decomposition call; a cache hit costs $0 and skips the whole pipeline.
- **Response cache:** SHA-256(query text) → Redis, TTL 24h. Review data only changes on re-ingestion (not continuously), and ingestion already calls `cache.invalidate_restaurant()` to bust the whole restaurant's cache on new data, so a long TTL is close to pure cost savings. Corrections additionally call `cache.invalidate_query()` to bust just that one query's cached entry, so a correction is never masked by a stale cache hit regardless of TTL length.
- **Background persistence never blocks the response.** Message persistence, session-memory writes, and cache writes run as a fire-and-forget `asyncio.create_task()` after the SSE stream's `done` event, using their own DB session (not the request-scoped one, which FastAPI tears down as soon as the route returns — reusing it caused an intermittent "transaction is closed" race, now fixed).

## Fail-closed ingestion, fail-open UI

- `scripts/seed.py` retries a failed ingest up to 3 times (60s, then 120s backoff, sized to typical OpenAI rate-limit reset windows). If all 3 fail, it exits 1 and **the backend container never starts** (`depends_on: seed: condition: service_completed_successfully`) — the system refuses to serve queries against a partially or incorrectly ingested dataset.
- The frontend container is **not** gated on the backend's readiness (only ordered to start after it, via plain `depends_on`). If the backend never comes up, the React app still loads and renders normally; only the restaurant-selector's `GET /restaurants` call fails, surfacing "Server unavailable. Is the backend running?" with a Retry button rather than a blank page or an infinite spinner.

## Known limitations / deviations from the standard

- **Multi-tenancy is demo-only.** The dropdown lets a user pick any seeded restaurant; a real JWT with an enforced `restaurant_id` claim (already implemented) is what actually prevents cross-tenant data access — the frontend's restaurant choice is never trusted server-side.
- **CI coverage gate is a 50% floor, not the 80% target.** `src/api/routes/*`, `src/workers/ingest_worker.py`, and most of `src/services/llm|vector` are only exercised by the live e2e eval harness (needs Postgres/Qdrant/Redis + real LLM keys), which CI intentionally excludes. Closing the gap needs FastAPI `TestClient` + `dependency_overrides` route tests, which don't need live infra — not yet written. Documented inline in `.github/workflows/ci.yml`.
- **The eval fixture's simulated-Qdrant-outage case (RB-01) is based on a now-stale assumption.** It expects the system to "degrade gracefully to BM25-only retrieval" if Qdrant goes down — but since sparse (BM25-style) vectors moved server-side into Qdrant itself, a Qdrant outage now takes down both dense and sparse retrieval together; there is no independent in-process BM25 fallback anymore. The eval harness documents this as an intentionally skipped case rather than faking a pass.
