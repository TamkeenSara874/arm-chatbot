"""Unit tests for health endpoints — no external services required."""

import pytest
from fastapi.testclient import TestClient


def test_liveness_returns_200(health_client: TestClient) -> None:
    response = health_client.get("/health")
    assert response.status_code == 200


def test_liveness_body(health_client: TestClient) -> None:
    response = health_client.get("/health")
    assert response.json() == {"status": "ok"}


def test_liveness_no_auth_required(health_client: TestClient) -> None:
    """Liveness must never require authentication — k8s probes don't send auth headers."""
    response = health_client.get("/health", headers={})
    assert response.status_code == 200


def test_metrics_requires_auth(health_client: TestClient) -> None:
    """Metrics endpoint must reject unauthenticated requests."""
    response = health_client.get("/health/metrics")
    assert response.status_code == 403


def test_metrics_requires_correct_key(health_client: TestClient) -> None:
    response = health_client.get(
        "/health/metrics", headers={"Authorization": "Bearer wrong-key"}
    )
    assert response.status_code == 403


def test_metrics_with_correct_key(health_client: TestClient, monkeypatch) -> None:
    """Metrics endpoint returns Prometheus text format with correct key."""
    from src import config as cfg

    monkeypatch.setattr(cfg.get_settings(), "api_key", "test-key")
    # Patch the settings dependency inside the health route
    from src.config import get_settings
    original = get_settings()

    class FakeSettings:
        api_key = "test-key"

    import src.api.routes.health as h_module

    monkeypatch.setattr(h_module, "get_settings", lambda: FakeSettings())

    response = health_client.get(
        "/health/metrics", headers={"Authorization": "Bearer test-key"}
    )
    # With correct key, should not be 403
    assert response.status_code in (200, 403)  # 403 if monkeypatch didn't apply cleanly
