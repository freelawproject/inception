import pytest
from fastapi.testclient import TestClient

from inception.main import app


class TestMonitoringEndpoints:
    """Tests for health check and monitoring endpoints."""

    @pytest.mark.monitoring
    def test_health_check(self, client):
        """Verify health check endpoint returns correct status and information."""
        with TestClient(app) as client:
            response = client.get("/health")
        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert "gpu_available" in data

    @pytest.mark.monitoring
    def test_heartbeat(self, client):
        """Verify heartbeat endpoint is responding."""
        response = client.get("/")
        assert response.status_code == 200
        assert response.text == '"Heartbeat detected."'

    @pytest.mark.monitoring
    def test_service_unavailable(self, client, monkeypatch):
        """Verify appropriate response when service is not initialized."""
        monkeypatch.setattr("inception.main.embedding_service", None)
        response = client.post("/api/v1/embed/query", json={"text": "test"})
        assert response.status_code == 503
        assert "service not initialized" in response.json()["detail"].lower()

    @pytest.mark.monitoring
    def test_metrics(self, client, monkeypatch):
        """Verify appropriate response for the metrics endpoint."""

        response = client.get("/metrics")
        assert response.status_code == 200
        assert "inception_requests_total" in response.content.decode()
        assert "inception_processing_seconds" in response.content.decode()
        assert "inception_errors_total" in response.content.decode()
        assert "inception_chunks_total" in response.content.decode()
        assert "inception_model_load_seconds" in response.content.decode()
