# Production image for the FastAPI backend. Multi-stage: install deps in one
# layer, copy only the app source into the final image, run as non-root.
FROM python:3.11-slim AS base
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

FROM base AS deps
COPY pyproject.toml .
# --no-deps + a placeholder src/ package dir would break setuptools' package
# discovery, so install the full editable project (pulls in base deps only --
# dev/test tooling lives in the "dev" extra, never installed here).
RUN mkdir -p src && pip install --upgrade pip && pip install -e .
RUN python -c "import nltk; nltk.download('punkt_tab', quiet=True)"

FROM deps AS production
COPY src/ ./src/
COPY prompts/ ./prompts/
COPY alembic/ ./alembic/
COPY alembic.ini .
RUN useradd --system --no-create-home appuser \
    && mkdir -p /home/appuser/nltk_data /app/logs \
    && cp -r /root/nltk_data/* /home/appuser/nltk_data/ \
    && chown -R appuser:appuser /home/appuser /app
ENV NLTK_DATA=/home/appuser/nltk_data
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["sh", "-c", "alembic upgrade head && uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS:-2}"]
