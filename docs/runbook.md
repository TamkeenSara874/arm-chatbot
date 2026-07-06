# Deployment Runbook

This covers what exists today (production Docker images, CI, CD build/push)
and the chosen deploy target (Railway — see
[Choosing a target](#choosing-a-target) at the bottom for the concrete setup).

## What "deployed" means right now

- Local `docker compose up` is the only environment this app has been run in
  so far; Railway (below) is the first real deploy target.
- `infra/docker/backend/Dockerfile` and `infra/docker/frontend/Dockerfile` are
  production-ready images (multi-stage, non-root, healthchecked). Railway
  builds directly from these Dockerfiles via its GitHub integration (see
  below for why) rather than consuming a prebuilt image.
- `.github/workflows/cd-staging.yml` (on every merge to `main`) and
  `cd-production.yml` (on a `v*` tag) independently build both images and
  push them to GHCR (`ghcr.io/<org>/<repo>-backend`, `-frontend`) as a
  versioned image mirror for audit/rollback reference. Railway's own deploy
  does not consume these images — see below.

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

**Decided: Railway.** Matches the free-tier-to-paid mapping's first step up
from local Docker, and its GitHub-connected build (below) sidesteps a real
problem the GHCR-image path would otherwise hit.

### Why Railway builds from GitHub directly, not from the GHCR image

`cd-staging.yml`/`cd-production.yml` still build and push both images to GHCR
on every merge/tag — keep them, they're a useful versioned image mirror for
audit/rollback reference. But **Railway's own deploy does not consume those
images.** The frontend Dockerfile bakes `VITE_API_URL` in as a build arg
(Vite inlines `VITE_*` vars into the static bundle at build time -- see
`infra/docker/frontend/Dockerfile`), and production's `nginx.conf` has no
`/api` proxy the way the local dev Vite server does. A prebuilt GHCR image
has no way to know the backend's real Railway URL before that URL exists, and
there's no reverse-proxy fallback in production to paper over it.

Railway solves this natively: point both services at this GitHub repo
directly (Settings → connect repo, root Dockerfile path per service), and use
Railway's **reference variables** to pass the backend's URL into the
frontend's build as a build arg: `VITE_API_URL=https://${{backend.RAILWAY_PUBLIC_DOMAIN}}`.
Railway resolves that at build time once the backend service exists, and
rebuilds the frontend automatically if the backend's domain ever changes.
Railway auto-redeploys both services on every push to the connected branch —
no GitHub Actions deploy job needed for this platform.

### Manual one-time setup (Railway dashboard/CLI -- requires your account)

1. Create a Railway project, connect this GitHub repo.
2. Add the **Postgres** and **Redis** plugins from Railway's template
   catalog -- both are one click, no Dockerfile needed.
3. Qdrant has no Railway-native managed plugin. Use **Qdrant Cloud's free
   tier** (matches the vector-db row in the project's root `CLAUDE.md` tier
   table) rather than self-hosting it as a third Railway service.
4. Create two Railway services from this repo:
   - `backend` — Dockerfile path `infra/docker/backend/Dockerfile`, build
     context repo root, port `8000`.
   - `frontend` — Dockerfile path `infra/docker/frontend/Dockerfile`, build
     context repo root, port `8080`, build arg
     `VITE_API_URL=https://${{backend.RAILWAY_PUBLIC_DOMAIN}}`.
5. Set every variable from [`.env.example`](../.env.example) on the
   `backend` service in Railway's Variables tab, pointing `DATABASE_URL` /
   `REDIS_URL` at the Postgres/Redis plugins' Railway-provided connection
   strings (Railway exposes these as reference variables too, e.g.
   `${{Postgres.DATABASE_URL}}`) and `QDRANT_URL`/`QDRANT_API_KEY` at the
   Qdrant Cloud cluster from step 3. See the table above for which values
   must not be left at their `.env.example` default.
6. After the first successful deploy, run
   `python scripts/create_restaurant_credential.py <restaurant_id>` against
   the production database for every real restaurant (see the migrations
   section above -- this is not an env var).
7. `cd-staging.yml`'s branch pushes are covered by Railway's own
   auto-deploy-on-push -- no extra step needed. `cd-production.yml`'s tagged
   releases are not (Railway doesn't watch tags), so its `deploy` job
   explicitly triggers a redeploy via the Railway CLI: generate a **project
   token** (Railway dashboard → project settings → Tokens) and add it as a
   `RAILWAY_TOKEN` secret in this repo's GitHub Actions settings. Without it,
   `cd-production.yml`'s deploy job fails loudly (by design) rather than
   silently skipping.

### Alternatives considered, not chosen

- **AWS ECS/Fargate** — needs a task definition referencing an image, a
  VPC/security group allowing Postgres/Qdrant/Redis connectivity (or moving
  those to RDS/managed equivalents), and an IAM role for pulling images.
  Meaningfully more setup than this app's current scale calls for.
- **GCP Cloud Run** — needs the image pushed/mirrored somewhere Cloud Run can
  pull from (Artifact Registry, or GHCR with a service account), and a Cloud
  Run service per container with the same env vars. Similar overhead to ECS
  for this app's size.
