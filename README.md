# ARM Review Chatbot

A production-grade RAG chatbot that lets restaurant owners ask plain-English questions about their customer reviews and receive instant, evidence-backed answers.

Built for the AIO Internship project using FastAPI, Qdrant, PostgreSQL, Redis, and React.

---

## What It Does

- Answers natural-language questions like "What do customers say about our service?" or "What should we improve?"
- Streams answers token-by-token with source citations from real reviews
- Routes simple queries to GPT-4o-mini (~$0.0003 typical) and complex aggregation to GPT-4.1 (~$0.005-0.015)
- Short-circuits count queries directly to PostgreSQL with zero LLM cost
- Grounds broad "how can I improve?" questions in the real complaint themes found first, before adding any general advice, and cites real counts (e.g. "3 of the 20 reviews mention...") instead of vague words like "several" — never invents a sales/revenue number reviews don't contain
- Answers questions about the conversation itself (e.g. "what did we talk about before?") from conversation memory alone, with no fake review evidence or confidence score attached, and honestly distinguishes this conversation from an earlier, separate one
- Caches repeated questions two ways: exact-text (Redis) and semantic/paraphrase (Qdrant similarity search)
- Generates a downloadable PDF insights report with charts (rating distribution, sentiment, source breakdown, top praised/complained) alongside the narrative summary
- Remembers conversation context across turns, and across a restaurant's *entire* chat history (not just the current session), via semantic retrieval from Qdrant
- Lets restaurant owners flag a wrong answer; a single flag is treated as an unverified aside, 3+ distinct flags become a confirmed correction that supersedes future evidence
- Supports live, single-review ingestion the moment a review posts or is edited, in addition to batch file upload — and a retried batch job never re-processes reviews it already finished
- Logs in per restaurant via a real access key (not an open restaurant picker) — closes a real gap where a shared dev key alone used to be enough to query any restaurant's data

---

## Architecture

```
React Frontend (Vite + TypeScript + Tailwind)
        │  LoginPage → POST /auth/token (restaurant_id + restaurant_key → JWT)
        │  REST + SSE stream (Bearer JWT)
        ▼
FastAPI /api/v1/
        │
        ├─ Cache check  (Redis exact-text, TTL 24h  +  Qdrant semantic, cosine ≥0.95)
        ├─ Guardrail    (out-of-scope → polite decline; report_howto → answered directly; ui_question → CareBot)
        ├─ Query decomposition  Groq llama-3.3-70b  (~300ms, rotates across free-tier keys on rate limit)
        │       └─ intent · filters · sub-queries · complexity · needs_aggregation
        │       (session context scoped to reference resolution only -- never
        │        allowed to override classification of a self-contained query)
        │
        ├─ conversation_recall → session context only, $0 retrieval, gpt-4o-mini
        ├─ count_query  → Postgres COUNT(*)  $0, <100ms (deterministic sentiment-
        │                 keyword override backs up decomposition's own extraction)
        ├─ report       → OpenAI tool call + DB/Qdrant aggregation → markdown + charts
        ├─ simple       → GPT-4o-mini  ~$0.0003
        └─ complex      → GPT-4.1       ~$0.005-0.015
                │
                ├─ Retrieval params: top_k=20 for aggregation/improvement questions, else 6
                ├─ Hybrid retrieval: Qdrant dense ANN + native sparse (fastembed BM25) + correction ANN
                ├─ Cross-encoder reranking: cross-encoder/ms-marco-MiniLM-L6-v2, ONNX int8 quantized, batch_size=8
                ├─ Evidence ranking: reranker relevance + recency decay + rating signal
                ├─ Correction lookup: is_consensus (3+ flags) → supersedes evidence; single flag → unverified aside
                ├─ Session context: last 5 turns + restaurant-wide semantic recall (90-day staleness cutoff)
                └─ SSE token stream → structured metadata in final event

Ingestion
  Batch    POST /ingest         → chunk → entity-extract → embed → upsert, per-batch, resumable on retry
  Live     POST /ingest/review  → same steps for one review; repeat call = update, not duplicate

Storage
  PostgreSQL   chat_session · chat_message · chat_correction · review_chunk_meta · ingest_job · restaurant_credential
  Qdrant       review_chunks (3072-dim dense + sparse) · correction_embeddings · session_memory · chat_cache
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
| Reranker      | cross-encoder/ms-marco-MiniLM-L6-v2, ONNX int8 quantized (local, sentence-transformers) |
| Charts        | Recharts (frontend insights report)                     |
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

# Seed data for restaurant_id=1 (the only restaurant seeded by default --
# scripts/seed.py ingests dataset/dataset.json for RESTAURANT_ID=1 only).
# Per-restaurant login/JWT isolation is real and enforced, but demoing it
# against a second populated tenant requires a second dataset ingested
# manually via POST /api/v1/ingest with restaurant_id=2.
python scripts/seed.py

# Smoke test the running stack
python scripts/smoke_test.py

# Provision/reissue a restaurant's login key
python scripts/create_restaurant_credential.py <restaurant_id>
```

---

## API Reference

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| POST | `/api/v1/auth/token` | Exchange `restaurant_id` + `restaurant_key` for a JWT | Shared `API_KEY` |
| POST | `/api/v1/chat/sessions` | Create a new chat session | Restaurant JWT |
| POST | `/api/v1/chat/query` | Send a message (SSE stream) | Restaurant JWT |
| GET | `/api/v1/chat/sessions/{id}/history` | Fetch message history | Restaurant JWT |
| POST | `/api/v1/chat/correct` | Submit a correction (auto-confirms at 3 flags) | Restaurant JWT |
| POST | `/api/v1/chat/{message_id}/feedback` | Submit thumbs-up feedback | Restaurant JWT |
| POST | `/api/v1/chat/report` | Generate an insights report | Restaurant JWT |
| POST | `/api/v1/ingest` | Upload a reviews JSON file (batch, resumable) | Shared `API_KEY` |
| POST | `/api/v1/ingest/review` | Push one review live (create or update) | Shared `API_KEY` |
| GET | `/api/v1/ingest/{job_id}/status` | Poll ingestion progress | Shared `API_KEY` |
| GET | `/api/v1/restaurants` | List available restaurants (ops/admin, not used by the login flow) | Shared `API_KEY` |
| GET | `/api/v1/health` | Liveness check | None |
| GET | `/api/v1/health/ready` | Readiness check (DB + Qdrant) | None |
| GET | `/api/v1/health/metrics` | Prometheus metrics | Shared `API_KEY` |

Two distinct Bearer tokens exist: the shared `API_KEY` (proves "legitimate client app," used for auth/ingest routes) and a per-restaurant JWT (minted by `/auth/token`, scoped to one `restaurant_id`, used for all `/chat/*` routes — the backend enforces the JWT's claim regardless of what a request body sends).

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
- A retried/resumed batch job skips any review it already fully processed under the current pipeline version — no wasted OpenAI calls on a retry

**Live, single-review ingestion** (`POST /api/v1/ingest/review`) — for pushing one review the moment it's posted or edited, instead of a full batch re-upload:

```json
{
  "restaurant_id": 1,
  "external_review_id": "source-systems-own-review-id",
  "review": "Amazing biryani!",
  "rating": 5,
  "username": "John D.",
  "sentiment": "Positive",
  "source": "Google",
  "created_at": "2025-01-15T12:00:00"
}
```

`external_review_id` must be a stable ID from the source system — calling this again with the same ID is a genuine update (reuses the same chunks, cleans up any now-stale ones), not a duplicate.

---

## Deployment

Production Dockerfiles exist for both services:

- `infra/docker/backend/Dockerfile` — multi-stage FastAPI image, runs migrations then `uvicorn` as a non-root user
- `infra/docker/frontend/Dockerfile` — multi-stage Vite build served by nginx

```bash
docker build -f infra/docker/backend/Dockerfile -t arm-chatbot-backend:latest .
docker build -f infra/docker/frontend/Dockerfile -t arm-chatbot-frontend:latest \
  --build-arg VITE_API_URL=https://your-deployed-backend-url .
```

No cloud target has been chosen yet — these images build and run anywhere that runs a container, but provisioning an actual Railway/ECS/Cloud Run deployment is a separate, not-yet-taken step. See [`docs/runbook.md`](docs/runbook.md) for the full build/push procedure, required secrets, and what's left to wire up a real deploy target.

---

## Known Limitations

**Only restaurant_id=1 is seeded.** The multi-tenancy login/JWT isolation is real and enforced, but only one restaurant's dataset is auto-seeded; demoing a second, populated tenant requires either sourcing a second dataset or ingesting one manually via `POST /api/v1/ingest`.

**Cost is exact, not estimated:** Both streaming and non-streaming LLM calls request `stream_options={"include_usage": true}` / read `response.usage` directly, so `cost_usd` in every trace reflects real provider-reported token counts and known model pricing rather than a character-count approximation.

**No cloud deployment target chosen yet.** Production Dockerfiles exist (`infra/docker/`), but the app currently only runs via local `docker compose`. See [Deployment](#deployment) below.

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

**Optional local dashboard stack (Prometheus + Grafana):** not started by a plain `docker compose up`. Bring it up explicitly with:

```bash
docker compose --profile monitoring up -d prometheus grafana
```

- Prometheus: [http://localhost:9090](http://localhost:9090)
- Grafana: [http://localhost:3001](http://localhost:3001) (not the default 3000 — avoids clashing with other local Grafana instances) — dashboard and datasource are auto-provisioned, nothing to configure manually; log in with `GRAFANA_ADMIN_USER` / `GRAFANA_ADMIN_PASSWORD` (see `.env.example`, defaults to `admin` / `change-me-local-dev-password`)

Errors are reported to Sentry when `SENTRY_DSN` is set.

**Verifying prompt/session-context correctness:** set `LOG_LEVEL=DEBUG` to see `decomposition_prompt` and `generation_prompt` structured log events containing the exact system/user prompt text sent to each LLM call, session context included -- useful for confirming what a model actually saw versus what `request_traces.jsonl`'s classified `intent`/`complexity` fields alone can tell you.

---

## Environment Variables

See [`.env.example`](.env.example) for the full list with comments. Minimum required for local dev:

```bash
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
HF_TOKEN=hf_...        # for reranker model download
API_KEY=any-local-key  # Bearer token for API calls
```
