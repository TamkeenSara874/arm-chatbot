# ARM Review Chatbot

A production-grade RAG chatbot that lets restaurant owners ask plain-English questions about their customer reviews and receive instant, evidence-backed answers.

Built for the AIO Internship project using FastAPI, Qdrant, PostgreSQL, Redis, and React.

---

## What It Does

- Answers natural-language questions like "What do customers say about our service?" or "What should we improve?"
- Streams answers token-by-token with source citations from real reviews
- Routes simple queries to GPT-4o-mini (~$0.0003 typical) and complex aggregation to GPT-4.1 (~$0.005-0.015)
- Short-circuits count queries directly to PostgreSQL with zero LLM cost
- Caches repeated questions in Redis (24-hour TTL)
- Generates downloadable PDF insight reports covering sentiment, ratings, and entity mentions
- Remembers conversation context across turns using semantic retrieval from Qdrant

---

## Architecture

```
React Frontend (Vite + TypeScript + Tailwind)
        │  REST + SSE stream
        ▼
FastAPI /api/v1/
        │
        ├─ Redis cache  (TTL 24h, SHA256 key)
        ├─ Guardrail    (out-of-scope → polite decline; report_howto → answered directly; ui_question → CareBot)
        ├─ Query decomposition  Groq llama-3.3-70b  (~300ms)
        │       └─ intent · filters · sub-queries · complexity
        │
        ├─ count_query  → Postgres COUNT(*)  $0, <100ms
        ├─ report       → OpenAI tool call + DB/Qdrant aggregation
        ├─ simple       → GPT-4o-mini  ~$0.0003
        └─ complex      → GPT-4.1       ~$0.005-0.015
                │
                ├─ Hybrid retrieval: Qdrant dense ANN + native sparse (fastembed BM25) + correction ANN
                ├─ Cross-encoder reranking: cross-encoder/ms-marco-MiniLM-L-6-v2 (local, free)
                ├─ Evidence ranking: RRF + recency decay + rating signal
                └─ SSE token stream → structured metadata in final event

Storage
  PostgreSQL   chat_session · chat_message · chat_correction · review_chunk_meta · ingest_job
  Qdrant       review_chunks (3072-dim dense + sparse)  ·  correction_embeddings  ·  session_memory
  Redis        response cache · rate-limit counters
```

**Cost per query** — measured from real provider-reported token usage (`logs/request_traces.jsonl`), not estimated; see `docs/evaluation_report.md` for methodology.

| Query type   | Model             | Typical cost |
|-------------|-------------------|-------------|
| Count / direct | PostgreSQL only | $0.000     |
| Cache hit    | —                 | $0.000 (no embedding needed — cache key is a hash of the query text) |
| Guardrail decline / redirect | Groq decomposition only | $0.000 |
| Simple       | GPT-4o-mini       | ~$0.0002-0.0003 |
| Complex      | GPT-4.1           | ~$0.005-0.015     |

---

## Stack

| Layer          | Technology                                              |
|---------------|---------------------------------------------------------|
| API           | FastAPI 0.115 · uvicorn · slowapi (rate limiting)       |
| LLM           | OpenAI GPT-4.1 / GPT-4o-mini · Groq llama-3.3-70b    |
| Embeddings    | text-embedding-3-large (3072 dims)                      |
| Vector DB     | Qdrant 1.10                                             |
| Reranker      | cross-encoder/ms-marco-MiniLM-L-6-v2 (local, sentence-transformers) |
| Database      | PostgreSQL 16 + SQLAlchemy async + Alembic              |
| Cache         | Redis 7                                                 |
| Frontend      | React 18 · TypeScript · Vite · Tailwind CSS             |
| Observability | structlog (JSON) · Prometheus metrics · Sentry          |

---

## Quick Start

### Prerequisites

- Docker Desktop
- API keys: OpenAI, Groq
- HuggingFace token (for reranker model download)

### 1. Configure environment

```bash
cp .env.example .env
# Fill in OPENAI_API_KEY, GROQ_API_KEY, HF_TOKEN
# Leave others at defaults for local dev
```

### 2. Start the stack

```bash
docker compose up
```

This starts PostgreSQL, Qdrant, Redis, the FastAPI backend (with migrations + seed data), and the React frontend.

**First-run timing** (subsequent runs are much faster — see below):
- `pip install` (backend + seed, cold cache): **2-5 min**, mostly `sentence-transformers`/`torch`. Cached in a named volume after the first run.
- Reranker + sparse-embedder model download (~110MB + ~50MB from HuggingFace): **~30-60s**, also cached in a named volume — only re-downloads if you run `docker compose down -v` (removes volumes).
- Real ingestion of the ~2,753-review dataset (entity extraction + embedding, real OpenAI calls): **several minutes**, bounded by OpenAI/Groq rate limits, not by this app's own code.
- Frontend `npm install`: **30s-2min** depending on network.

**Total first run: budget 10-15 minutes.** Every run after that (`docker compose up` without `-v`) reuses the pip/model/node_modules volumes and the seed service exits in under a second (manifest hash unchanged) — expect the whole stack ready in well under a minute.

**For a demo:** don't run `docker compose up` live for the first time — pre-warm it beforehand (once the caches are populated, startup is fast and reliable) and just do `docker compose up` (or `restart` if it's already been up) right before you present.

### 3. Open the app

Frontend: http://localhost:5173  
API docs: http://localhost:8000/docs  
Health: http://localhost:8000/health/ready

---

## Development

```bash
# Install Python deps (requires Python 3.11+)
pip install -e ".[dev]"
pre-commit install

# Run backend only (requires local Postgres/Qdrant/Redis)
uvicorn src.api.main:app --reload --port 8000

# Run frontend only
cd frontend && npm install && npm run dev

# Lint
ruff check src tests

# Tests
pytest --cov=src -v

# Seed data for restaurant_id=1 and 2
python scripts/seed.py

# Evaluate retrieval quality (P@5, Recall@5, MRR)
python scripts/eval_retrieval.py --k 5

# Load test (50 concurrent queries, P50/P95 latency)
python scripts/load_test.py --n 50
```

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/chat/sessions` | Create a new chat session |
| POST | `/api/v1/chat/query` | Send a message (SSE stream) |
| GET | `/api/v1/chat/sessions/{id}/history` | Fetch message history |
| POST | `/api/v1/chat/correct` | Submit a correction |
| POST | `/api/v1/chat/report` | Generate an insights report |
| POST | `/api/v1/ingest` | Upload a reviews JSON file |
| GET | `/api/v1/ingest/{job_id}/status` | Poll ingestion progress |
| GET | `/api/v1/restaurants` | List available restaurants |
| GET | `/api/v1/health` | Liveness check |
| GET | `/api/v1/health/ready` | Readiness check (DB + Qdrant) |
| GET | `/api/v1/health/metrics` | Prometheus metrics (auth required) |

All endpoints except `/health` and `/health/ready` require `Authorization: Bearer <API_KEY>`.

**SSE stream format** (`POST /api/v1/chat/query`)

```
event: token
data: " word"

event: token
data: " by"

event: done
data: {"message_id":"...","response":{...},"latency_ms":1240,"cost_usd":0.0012}
```

---

## Ingest Format

Upload a JSON file exported from ARM's review pipeline. Expected structure:

```json
{
  "<any key>": [
    {
      "review": "Amazing biryani!",
      "createdAt": "2025-01-15T12:00:00",
      "rating": 5,
      "isRead": true,
      "username": "John D.",
      "sentiment": "Positive",
      "source": "Google"
    }
  ]
}
```

- Encoding: UTF-8 BOM (`utf-8-sig`)
- `sentiment` is pre-computed by ARM's pipeline — it is used as-is, not recomputed
- Malformed `createdAt` falls back to `datetime.now()` with a WARNING logged
- Missing fields get safe defaults (rating → null, username → "Anonymous", sentiment → "Neutral")

---

## Known Limitations

**Multi-tenancy (demo only):** The dropdown lets any user query any restaurant. Production requires a JWT with a `restaurant_id` claim that the backend enforces, ignoring whatever the frontend sends.

**Cost is exact, not estimated:** Both streaming and non-streaming LLM calls request `stream_options={"include_usage": true}` / read `response.usage` directly, so `cost_usd` in every trace reflects real provider-reported token counts and known model pricing rather than a character-count approximation.

---

## Observability

Every request emits a `request_trace` structured log entry:

```json
{
  "event": "request_trace",
  "intent": "specific_aspect",
  "complexity": "simple",
  "decomp_ms": 312.4,
  "retrieval_ms": 187.2,
  "rerank_ms": 54.1,
  "ranking_ms": 3.8,
  "generation_ms": 891.0,
  "total_ms": 1448.5,
  "cost_usd": 0.001240,
  "confidence": 0.87,
  "evidence_count": 4,
  "cache_hit": false
}
```

Prometheus metrics exposed at `/api/v1/health/metrics` (Bearer auth required):

- `pipeline_stage_latency_seconds` — histogram per stage (decomp, retrieval, rerank, ranking, generation)
- `request_cost_usd` — histogram of per-request cost
- `cache_hit_total` — counter
- `active_sessions_total` — gauge
- `guardrail_triggered_total` — counter by type

Errors are reported to Sentry when `SENTRY_DSN` is set.

---

## Environment Variables

See [`.env.example`](.env.example) for the full list with comments. Minimum required for local dev:

```bash
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
HF_TOKEN=hf_...        # for reranker model download
API_KEY=any-local-key  # Bearer token for API calls
```
