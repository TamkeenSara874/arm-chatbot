"""Verify cross-restaurant isolation at the HTTP boundary, not just at hybrid_retrieve().

test_multitenant.py (module-level unit tests) only proves hybrid_retrieve() forwards
whatever restaurant_id it's given to the vector store -- it never touches JWT decoding
or the "body-supplied restaurant_id is ignored" guarantee that require_restaurant_jwt()
actually provides. This file drives a real request through the FastAPI dependency chain
so that guarantee is checked by something that runs in CI, not just asserted in a docstring.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from jose import jwt
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from src.api.dependencies import (
    get_cache,
    get_complex_client,
    get_db,
    get_decomp_client,
    get_embedder,
    get_simple_client,
    get_summary_client,
    get_vector_store,
)
from src.api.routes.chat import router as chat_router
from src.config import get_settings
from src.models.schemas import DecomposedQuery

RESTAURANT_A = 101
RESTAURANT_B = 202


def _mint_jwt(restaurant_id: int, expired: bool = False) -> str:
    settings = get_settings()
    delta = timedelta(hours=-1) if expired else timedelta(hours=settings.jwt_expiry_hours)
    payload = {
        "sub": f"restaurant:{restaurant_id}",
        "restaurant_id": restaurant_id,
        "exp": datetime.now(UTC) + delta,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def _mock_db() -> MagicMock:
    """Enough of an AsyncSession for build_recent_turns_context()'s query to no-op."""
    db = MagicMock()
    scalars_result = MagicMock()
    scalars_result.all.return_value = []
    execute_result = MagicMock()
    execute_result.scalars.return_value = scalars_result
    db.execute = AsyncMock(return_value=execute_result)
    return db


def _mock_decomp_client() -> MagicMock:
    client = MagicMock()
    client.complete_structured = AsyncMock(
        return_value=DecomposedQuery(intent="factual", rephrased_query="test query")
    )
    return client


def _mock_vector_store() -> MagicMock:
    """search() (semantic cache) and hybrid_search() (retrieval) both miss/empty.

    Empty hybrid_search results route the pipeline into the no-evidence-gate
    fast path, so no generation LLM call is needed either.
    """
    store = MagicMock()
    store.search = AsyncMock(return_value=[])
    store.hybrid_search = AsyncMock(return_value=[])
    store.upsert = AsyncMock(return_value=None)
    return store


def _mock_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.embed_one = AsyncMock(return_value=[0.1] * 3072)
    return embedder


def _mock_cache() -> MagicMock:
    cache = MagicMock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock(return_value=None)
    return cache


@pytest.fixture
def isolation_app() -> FastAPI:
    """Minimal app with only chat.router -- heavy dependencies mocked, JWT NOT overridden.

    require_restaurant_jwt is the function under test, so it must run for real.
    """
    app = FastAPI()
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, lambda request, exc: exc)
    app.add_middleware(SlowAPIMiddleware)
    app.include_router(chat_router)

    vector_store = _mock_vector_store()
    app.dependency_overrides[get_db] = lambda: _mock_db()
    app.dependency_overrides[get_decomp_client] = lambda: _mock_decomp_client()
    app.dependency_overrides[get_simple_client] = lambda: MagicMock()
    app.dependency_overrides[get_complex_client] = lambda: MagicMock()
    app.dependency_overrides[get_summary_client] = lambda: MagicMock()
    app.dependency_overrides[get_embedder] = lambda: _mock_embedder()
    app.dependency_overrides[get_vector_store] = lambda: vector_store
    app.dependency_overrides[get_cache] = lambda: _mock_cache()
    app.state._test_vector_store = vector_store
    return app


@pytest.fixture
def isolation_client(isolation_app: FastAPI) -> TestClient:
    return TestClient(isolation_app)


class TestMultitenantHttpIsolation:
    def test_jwt_restaurant_id_wins_over_body_restaurant_id(
        self, isolation_client: TestClient, isolation_app: FastAPI
    ) -> None:
        """A JWT for restaurant A must resolve to A even if the body claims restaurant B."""
        token = _mint_jwt(RESTAURANT_A)

        with patch("src.core.retrieval.compute_sparse_vector") as sparse_mock:
            from src.services.embedding.sparse_embedder import SparseVector

            sparse_mock.return_value = SparseVector(indices=[0, 1], values=[0.5, 0.5])

            response = isolation_client.post(
                "/api/v1/chat/query",
                json={
                    "session_id": "11111111-1111-1111-1111-111111111111",
                    "restaurant_id": RESTAURANT_B,
                    "message": "how is the food?",
                },
                headers={"Authorization": f"Bearer {token}"},
            )

        assert response.status_code == 200

        vector_store = isolation_app.state._test_vector_store
        assert vector_store.hybrid_search.called, "retrieval must have run"
        _, call_kwargs = vector_store.hybrid_search.call_args
        resolved_restaurant_id = call_kwargs.get("filters", {}).get("restaurant_id")
        assert resolved_restaurant_id == RESTAURANT_A, (
            f"Expected retrieval scoped to JWT's restaurant_id ({RESTAURANT_A}), "
            f"got {resolved_restaurant_id} -- the body's restaurant_id "
            f"({RESTAURANT_B}) must never override the JWT claim."
        )

    def test_expired_jwt_is_rejected(self, isolation_client: TestClient) -> None:
        token = _mint_jwt(RESTAURANT_A, expired=True)

        response = isolation_client.post(
            "/api/v1/chat/query",
            json={
                "session_id": "11111111-1111-1111-1111-111111111111",
                "restaurant_id": RESTAURANT_A,
                "message": "how is the food?",
            },
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 401

    def test_missing_jwt_is_rejected(self, isolation_client: TestClient) -> None:
        response = isolation_client.post(
            "/api/v1/chat/query",
            json={
                "session_id": "11111111-1111-1111-1111-111111111111",
                "restaurant_id": RESTAURANT_A,
                "message": "how is the food?",
            },
        )

        assert response.status_code == 401
