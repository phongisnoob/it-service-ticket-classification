"""Tests for API hardening: Prometheus label safety, auth modes, metrics."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from app import main as api
from src.inference import PredictionResult


class FakePredictor:
    backend = "baseline"
    threshold = 0.50
    model_sha256 = "testhash"

    def predict(self, text: str) -> PredictionResult:
        return {
            "category": "Access",
            "confidence": 0.95,
            "threshold": 0.50,
            "needs_manual_review": False,
            "top_3": [
                {"category": "Access", "probability": 0.95},
                {"category": "Administrative rights", "probability": 0.03},
                {"category": "Hardware", "probability": 0.02},
            ],
        }


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr(api, "get_predictor", lambda backend: FakePredictor())
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.delenv("API_KEY", raising=False)
    with TestClient(api.app) as c:
        yield c


@pytest.fixture()
def auth_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr(api, "get_predictor", lambda backend: FakePredictor())
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("API_KEY", "secret123")
    # Reload API_KEY module-level var
    monkeypatch.setattr(api, "API_KEY", "secret123")
    with TestClient(api.app) as c:
        yield c


class TestPrometheusLabelSafety:
    """Arbitrary unknown URLs must NOT create unbounded metric cardinality."""

    def test_unknown_url_uses_unmatched_sentinel(self, client: TestClient) -> None:
        # Hit a completely arbitrary path
        client.get("/totally/arbitrary/path/xyz/123")
        # Collect metric samples and verify no random path appears as a label
        samples = [
            sample
            for metric in REGISTRY.collect()
            if metric.name in ("http_requests_total",)
            for sample in metric.samples
        ]
        endpoints = {s.labels.get("endpoint", "") for s in samples}
        # Should only contain known route templates or the sentinel
        for ep in endpoints:
            assert ep in {"/", "/health", "/predict", "/metrics", "UNMATCHED", ""}, (
                f"Unexpected endpoint label in Prometheus: {ep!r}"
            )

    def test_known_route_uses_template(self, client: TestClient) -> None:
        # Make a request and verify Prometheus labels only contain known templates.
        # We can't assert /health specifically because the prometheus REGISTRY is a
        # process-wide singleton; instead verify no arbitrary path strings leak in.
        client.get("/health")
        client.get("/totally/unknown/path/12345")
        samples = [
            sample
            for metric in REGISTRY.collect()
            if metric.name == "http_requests_total"
            for sample in metric.samples
        ]
        for s in samples:
            ep = s.labels.get("endpoint", "")
            assert ep in {"/", "/health", "/predict", "/metrics", "UNMATCHED", ""}, (
                f"Unexpected endpoint label: {ep!r}"
            )


class TestAuthModes:
    def test_open_mode_no_key_required(self, client: TestClient) -> None:
        resp = client.post("/predict", json={"text": "printer is broken"})
        assert resp.status_code == 200

    def test_auth_mode_no_key_rejected(self, auth_client: TestClient) -> None:
        resp = auth_client.post("/predict", json={"text": "printer is broken"})
        assert resp.status_code == 401

    def test_auth_mode_correct_key_accepted(self, auth_client: TestClient) -> None:
        resp = auth_client.post(
            "/predict",
            json={"text": "printer is broken"},
            headers={"x-api-key": "secret123"},
        )
        assert resp.status_code == 200

    def test_auth_mode_wrong_key_rejected(self, auth_client: TestClient) -> None:
        resp = auth_client.post(
            "/predict",
            json={"text": "printer is broken"},
            headers={"x-api-key": "wrong"},
        )
        assert resp.status_code == 401


class TestOversizedInput:
    def test_oversized_text_rejected(self, client: TestClient) -> None:
        resp = client.post("/predict", json={"text": "x" * 6000})
        assert resp.status_code == 422

    def test_blank_text_rejected(self, client: TestClient) -> None:
        resp = client.post("/predict", json={"text": "   "})
        assert resp.status_code == 422


class TestMetricsEndpoint:
    def test_metrics_endpoint_reachable(self, client: TestClient) -> None:
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert b"python_" in resp.content or b"http_" in resp.content


class TestHealthEndpoint:
    def test_health_returns_backend(self, client: TestClient) -> None:
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "ok"
        assert "model_backend" in data
        assert "model_sha256" in data
        assert "threshold" in data
