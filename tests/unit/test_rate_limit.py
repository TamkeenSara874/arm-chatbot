"""Unit tests for the shared rate-limit key function.

Regression coverage for a real bug: rate limiting was keyed by remote IP
alone (get_remote_address), and enforced by two separate, unconfigured
in-memory Limiter instances (one each in chat.py and ingest.py) instead of
the Redis-backed one main.py actually built. rate_limit_key() fixes the
keying; src/api/rate_limit.py's single shared `limiter` fixes the instance
duplication.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

from jose import jwt

from src.api.rate_limit import rate_limit_key
from src.config import get_settings


def _mint_jwt(restaurant_id: int, expired: bool = False) -> str:
    settings = get_settings()
    delta = timedelta(hours=-1) if expired else timedelta(hours=settings.jwt_expiry_hours)
    payload = {"restaurant_id": restaurant_id, "exp": datetime.now(UTC) + delta}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def _mock_request(auth_header: str | None) -> MagicMock:
    request = MagicMock()
    request.headers.get.return_value = auth_header or ""
    return request


class TestRateLimitKey:
    def test_valid_restaurant_jwt_keys_by_restaurant_id(self) -> None:
        token = _mint_jwt(restaurant_id=42)
        request = _mock_request(f"Bearer {token}")
        assert rate_limit_key(request) == "restaurant:42"

    def test_bearer_prefix_is_case_insensitive(self) -> None:
        token = _mint_jwt(restaurant_id=7)
        request = _mock_request(f"bearer {token}")
        assert rate_limit_key(request) == "restaurant:7"

    def test_no_auth_header_falls_back_to_ip(self) -> None:
        request = _mock_request(None)
        with patch("src.api.rate_limit.get_remote_address", return_value="1.2.3.4"):
            assert rate_limit_key(request) == "1.2.3.4"

    def test_shared_api_key_falls_back_to_ip(self) -> None:
        # The shared API_KEY is a plain string, not a JWT -- auth/ingest
        # routes send this as the Bearer token, and it must fail JWT decode
        # cleanly rather than crash the rate limiter.
        request = _mock_request("Bearer change-me-local-dev-key")
        with patch("src.api.rate_limit.get_remote_address", return_value="5.6.7.8"):
            assert rate_limit_key(request) == "5.6.7.8"

    def test_expired_jwt_falls_back_to_ip(self) -> None:
        token = _mint_jwt(restaurant_id=9, expired=True)
        request = _mock_request(f"Bearer {token}")
        with patch("src.api.rate_limit.get_remote_address", return_value="9.9.9.9"):
            assert rate_limit_key(request) == "9.9.9.9"

    def test_jwt_missing_restaurant_id_claim_falls_back_to_ip(self) -> None:
        settings = get_settings()
        token = jwt.encode(
            {"exp": datetime.now(UTC) + timedelta(hours=1)},
            settings.jwt_secret,
            algorithm=settings.jwt_algorithm,
        )
        request = _mock_request(f"Bearer {token}")
        with patch("src.api.rate_limit.get_remote_address", return_value="8.8.8.8"):
            assert rate_limit_key(request) == "8.8.8.8"
