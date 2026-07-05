"""HTTP-level tests for POST /auth/token's per-restaurant credential check.

Regression coverage for the fix that closed a real gap: this route used to
mint a JWT for ANY restaurant_id behind only the shared API_KEY (which the
frontend ships to the browser). It now also requires a restaurant_key that
must match the specific restaurant_id being requested.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.auth import router as auth_router
from src.config import get_settings
from src.models.db_entities import RestaurantCredential
from src.services.database import get_db
from src.utils.restaurant_auth import generate_restaurant_key, hash_restaurant_key

API_KEY = get_settings().api_key


def _mock_db(credential: RestaurantCredential | None) -> MagicMock:
    db = MagicMock()
    db.get = AsyncMock(return_value=credential)
    return db


@pytest.fixture
def app() -> FastAPI:
    app = FastAPI()
    app.include_router(auth_router)
    return app


class TestIssueRestaurantToken:
    def test_correct_restaurant_key_succeeds(self, app: FastAPI) -> None:
        key = generate_restaurant_key()
        credential = RestaurantCredential(restaurant_id=1, key_hash=hash_restaurant_key(key))
        app.dependency_overrides[get_db] = lambda: _mock_db(credential)
        client = TestClient(app)

        response = client.post(
            "/api/v1/auth/token",
            json={"restaurant_id": 1, "restaurant_key": key},
            headers={"Authorization": f"Bearer {API_KEY}"},
        )

        assert response.status_code == 200
        assert response.json()["restaurant_id"] == 1
        assert response.json()["access_token"]

    def test_wrong_restaurant_key_is_rejected(self, app: FastAPI) -> None:
        credential = RestaurantCredential(
            restaurant_id=1, key_hash=hash_restaurant_key(generate_restaurant_key())
        )
        app.dependency_overrides[get_db] = lambda: _mock_db(credential)
        client = TestClient(app)

        response = client.post(
            "/api/v1/auth/token",
            json={"restaurant_id": 1, "restaurant_key": "wrong-key"},
            headers={"Authorization": f"Bearer {API_KEY}"},
        )

        assert response.status_code == 401

    def test_unknown_restaurant_id_is_rejected(self, app: FastAPI) -> None:
        app.dependency_overrides[get_db] = lambda: _mock_db(None)
        client = TestClient(app)

        response = client.post(
            "/api/v1/auth/token",
            json={"restaurant_id": 999, "restaurant_key": "anything"},
            headers={"Authorization": f"Bearer {API_KEY}"},
        )

        assert response.status_code == 401

    def test_missing_restaurant_key_is_rejected_by_validation(self, app: FastAPI) -> None:
        app.dependency_overrides[get_db] = lambda: _mock_db(None)
        client = TestClient(app)

        response = client.post(
            "/api/v1/auth/token",
            json={"restaurant_id": 1},
            headers={"Authorization": f"Bearer {API_KEY}"},
        )

        assert response.status_code == 422

    def test_a_valid_key_for_a_different_restaurant_is_rejected(self, app: FastAPI) -> None:
        """The exact scenario this fix closes: knowing restaurant_id=1's key
        must not grant a token for restaurant_id=2."""
        restaurant_1_key = generate_restaurant_key()
        # Simulate looking up restaurant_id=2's credential (a different key
        # than the one the caller is presenting).
        credential = RestaurantCredential(
            restaurant_id=2, key_hash=hash_restaurant_key(generate_restaurant_key())
        )
        app.dependency_overrides[get_db] = lambda: _mock_db(credential)
        client = TestClient(app)

        response = client.post(
            "/api/v1/auth/token",
            json={"restaurant_id": 2, "restaurant_key": restaurant_1_key},
            headers={"Authorization": f"Bearer {API_KEY}"},
        )

        assert response.status_code == 401

    def test_missing_shared_api_key_is_rejected(self, app: FastAPI) -> None:
        app.dependency_overrides[get_db] = lambda: _mock_db(None)
        client = TestClient(app)

        response = client.post(
            "/api/v1/auth/token",
            json={"restaurant_id": 1, "restaurant_key": "anything"},
        )

        assert response.status_code == 401
