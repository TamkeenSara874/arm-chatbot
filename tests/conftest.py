import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.health import router as health_router


@pytest.fixture(scope="session")
def health_app() -> FastAPI:
    """Minimal app containing only health routes — no lifespan, no external deps.

    Use this for fast unit tests that don't need Postgres/Qdrant/Redis.
    """
    app = FastAPI()
    app.include_router(health_router)
    return app


@pytest.fixture(scope="session")
def health_client(health_app: FastAPI) -> TestClient:
    return TestClient(health_app)
