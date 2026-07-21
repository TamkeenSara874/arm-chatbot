"""HTTP-level tests for POST /api/v1/voice/transcribe."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from jose import jwt
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from src.api.dependencies import get_stt_client
from src.api.routes.voice import router as voice_router
from src.config import get_settings

RESTAURANT_ID = 1


def _mint_jwt(restaurant_id: int = RESTAURANT_ID, expired: bool = False) -> str:
    settings = get_settings()
    delta = timedelta(hours=-1) if expired else timedelta(hours=settings.jwt_expiry_hours)
    payload = {
        "sub": f"restaurant:{restaurant_id}",
        "restaurant_id": restaurant_id,
        "exp": datetime.now(UTC) + delta,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def _mock_stt_client(text: str = "how is the food?") -> MagicMock:
    client = MagicMock()
    client.transcribe = AsyncMock(return_value=text)
    return client


@pytest.fixture
def app() -> FastAPI:
    app = FastAPI()
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, lambda request, exc: exc)
    app.add_middleware(SlowAPIMiddleware)
    app.include_router(voice_router)
    return app


class TestTranscribeRoute:
    def test_valid_audio_returns_transcribed_text(self, app: FastAPI) -> None:
        app.dependency_overrides[get_stt_client] = lambda: _mock_stt_client("what do people think?")
        client = TestClient(app)

        response = client.post(
            "/api/v1/voice/transcribe",
            headers={"Authorization": f"Bearer {_mint_jwt()}"},
            files={"file": ("clip.webm", b"fake-audio-bytes", "audio/webm")},
        )

        assert response.status_code == 200
        assert response.json()["text"] == "what do people think?"

    def test_missing_jwt_is_rejected(self, app: FastAPI) -> None:
        app.dependency_overrides[get_stt_client] = lambda: _mock_stt_client()
        client = TestClient(app)

        response = client.post(
            "/api/v1/voice/transcribe",
            files={"file": ("clip.webm", b"fake-audio-bytes", "audio/webm")},
        )

        assert response.status_code == 401

    def test_expired_jwt_is_rejected(self, app: FastAPI) -> None:
        app.dependency_overrides[get_stt_client] = lambda: _mock_stt_client()
        client = TestClient(app)

        response = client.post(
            "/api/v1/voice/transcribe",
            headers={"Authorization": f"Bearer {_mint_jwt(expired=True)}"},
            files={"file": ("clip.webm", b"fake-audio-bytes", "audio/webm")},
        )

        assert response.status_code == 401

    def test_unsupported_content_type_is_rejected(self, app: FastAPI) -> None:
        app.dependency_overrides[get_stt_client] = lambda: _mock_stt_client()
        client = TestClient(app)

        response = client.post(
            "/api/v1/voice/transcribe",
            headers={"Authorization": f"Bearer {_mint_jwt()}"},
            files={"file": ("clip.txt", b"not audio", "text/plain")},
        )

        assert response.status_code == 415

    def test_oversized_audio_is_rejected(self, app: FastAPI) -> None:
        app.dependency_overrides[get_stt_client] = lambda: _mock_stt_client()
        client = TestClient(app)
        settings = get_settings()
        oversized = b"0" * (settings.voice_max_upload_mb * 1024 * 1024 + 1)

        response = client.post(
            "/api/v1/voice/transcribe",
            headers={"Authorization": f"Bearer {_mint_jwt()}"},
            files={"file": ("clip.webm", oversized, "audio/webm")},
        )

        assert response.status_code == 413

    def test_stt_client_failure_returns_502_not_a_raw_exception(self, app: FastAPI) -> None:
        client_mock = MagicMock()
        client_mock.transcribe = AsyncMock(side_effect=RuntimeError("groq down"))
        app.dependency_overrides[get_stt_client] = lambda: client_mock
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/api/v1/voice/transcribe",
            headers={"Authorization": f"Bearer {_mint_jwt()}"},
            files={"file": ("clip.webm", b"fake-audio-bytes", "audio/webm")},
        )

        assert response.status_code == 502
