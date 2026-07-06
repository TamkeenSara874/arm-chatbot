# Deployment Runbook

This covers what exists today (production Docker images, CI, CD build/push) and
what's deliberately not done yet (an actual cloud deploy target — see
[Choosing a target](#choosing-a-target) at the bottom).

## What "deployed" means right now

- Local `docker compose up` is the only environment this app runs in today.
- `infra/docker/backend/Dockerfile` and `infra/docker/frontend/Dockerfile` are
  production-ready images (multi-stage, non-root, healthchecked) but nothing
  currently runs them outside a manual `docker build`/`docker run`.
- `.github/workflows/cd-staging.yml` (on every merge to `main`) and
  `cd-production.yml` (on a `v*` tag) build both images and push them to GHCR
  (`ghcr.io/<org>/<repo>-backend`, `-frontend`). Neither workflow deploys
  anywhere yet — that job is a placeholder until a platform is chosen.

## Building images locally

```bash
# Backend
docker build -f infra/docker/backend/Dockerfile -t arm-chatbot-backend:local .

# Frontend -- VITE_API_URL is a BUILD ARG, not a runtime env var. Vite inlines
# VITE_* vars into the bundle at build time, so this must point at wherever
# the backend will actually be reachable from the browser, not localhost.
docker build -f infra/docker/frontend/Dockerfile -t arm-chatbot-frontend:local \
  --build-arg VITE_API_URL=https://your-backend-host .
```

Both builds use the **repo root** as build context (not `frontend/`), since the
frontend Dockerfile also needs `infra/docker/frontend/nginx.conf`.

## Running the built images locally (smoke test before pushing)

```bash
docker run --rm -p 8000:8000 \
  --env-file .env \
  -e DATABASE_URL=postgresql+asyncpg://postgres:postgres@host.docker.internal:5432/armchatbot \
  arm-chatbot-backend:local

curl http://localhost:8000/health/ready
```

```bash
docker run --rm -p 8080:8080 arm-chatbot-frontend:local
curl http://localhost:8080/
```

## Required environment variables / secrets in production

Every variable in [`.env.example`](../.env.example) needs a real value in
whatever secret manager the eventual platform uses (Railway variables, AWS
Secrets Manager, GCP Secret Manager, etc.) — **never** commit a `.env` file.
The ones that specifically must NOT be left at their `.env.example` default:

| Variable | Why it matters in production |
|---|---|
| `API_KEY` | Shared client-app secret. Rotate from the dev default. |
| `JWT_SECRET` | Signs every restaurant-scoped token. Must be a strong random 32+ char value, different per environment. |
| `DATABASE_URL` | Points at the real Postgres instance, with real credentials. |
| `QDRANT_URL` / `QDRANT_API_KEY` | Qdrant Cloud (or self-hosted) endpoint + key, not the local container. |
| `REDIS_URL` | Managed Redis (e.g. Upstash), not the local container. |
| `OPENAI_API_KEY`, `GROQ_API_KEY` (+ optional `GROQ_API_KEYS`) | Real provider keys. Set spending alerts on the OpenAI account. |
| `HF_TOKEN` | Avoids unauthenticated HuggingFace rate limits on the reranker's first download. |
| `ALLOWED_ORIGINS` | Must be the real deployed frontend origin(s), never `*`. |
| `SENTRY_DSN` | Set so production errors are actually captured. |

Per-restaurant login keys (`restaurant_credential` table) are provisioned
separately, per restaurant, via `python scripts/create_restaurant_credential.py
<restaurant_id>` run against the production database — they are not an env var.

## Database migrations

The backend image's `CMD` runs `alembic upgrade head` before starting
`uvicorn`. **This is safe for a single replica.** If the deploy target ever
runs multiple replicas of the backend simultaneously, migrations must move to
a separate one-shot step (a pre-deploy job/hook) instead of running inside
every replica's container start — concurrent `alembic upgrade head` calls
across replicas is not something this setup has been designed or tested for.

## Rollback

Since CD pushes an immutable tag per commit/release
(`ghcr.io/<repo>-backend:staging-<sha>` or `:<git-tag>`), rolling back is
redeploying the previous known-good tag on whatever platform is running it —
there is no in-place "undo" needed as long as the previous image tag hasn't
been deleted from the registry.

## Verifying a deploy

1. `GET /health` — liveness, no dependencies checked.
2. `GET /health/ready` — confirms Postgres, Qdrant, and Redis are all reachable.
3. Mint a token (`POST /auth/token` with a real restaurant's credential) and
   run one real chat query end-to-end.
4. Check `logs/request_traces.jsonl` (or wherever structured logs land in the
   target platform) for a `request_trace` entry with a sane `total_ms` and
   `cost_usd`.

## Choosing a target

Not yet decided. Once it is, each platform needs roughly this wiring added to
the placeholder job at the bottom of `cd-staging.yml`/`cd-production.yml`:

- **Railway** — simplest: point a Railway service at the GHCR image, set env
  vars in its dashboard/CLI, done. Matches the free-tier-to-paid mapping's
  first step up from local Docker.
- **AWS ECS/Fargate** — needs a task definition referencing the GHCR image, a
  VPC/security group allowing Postgres/Qdrant/Redis connectivity (or moving
  those to RDS/managed equivalents), and an IAM role for pulling from GHCR.
- **GCP Cloud Run** — needs the image pushed/mirrored somewhere Cloud Run can
  pull from (Artifact Registry, or GHCR with a service account), and a Cloud
  Run service per container (backend, frontend) with the same env vars.

Whichever is chosen, Postgres/Qdrant/Redis need a real managed instance too —
see the free-tier-to-paid mapping in the project's root `CLAUDE.md` for the
per-service options.
