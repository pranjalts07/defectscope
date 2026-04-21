"""
API integration tests using httpx AsyncClient.

These tests mock the predictor so they don't require trained model weights.
To test with real models, remove the mock and ensure checkpoints exist.
"""

import io
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app


def make_test_image_bytes() -> bytes:
    """Create a simple 256x256 RGB PNG in memory."""
    arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


MOCK_PREDICTION = {
    "prediction": "good",
    "confidence": 0.92,
    "anomaly_score": 0.008,
    "anomaly_threshold": 0.012,
    "needs_review": False,
    "heatmap_b64": None,
    "latency_ms": 32.5,
}


@pytest.fixture
def client_with_mock():
    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = MOCK_PREDICTION
    mock_predictor.device = "cpu"
    mock_predictor.anomaly_threshold = 0.012

    with patch.object(app.state, "predictor", mock_predictor, create=True):
        with TestClient(app) as client:
            yield client


class TestHealthEndpoint:
    def test_health_returns_200(self, client_with_mock):
        resp = client_with_mock.get("/health")
        assert resp.status_code == 200

    def test_health_schema(self, client_with_mock):
        resp = client_with_mock.get("/health")
        data = resp.json()
        assert "status" in data
        assert "models_loaded" in data
        assert "device" in data


class TestPredictEndpoint:
    def test_predict_valid_image(self, client_with_mock):
        img_bytes = make_test_image_bytes()
        resp = client_with_mock.post(
            "/predict",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        assert resp.status_code == 200

    def test_predict_response_schema(self, client_with_mock):
        img_bytes = make_test_image_bytes()
        resp = client_with_mock.post(
            "/predict",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        data = resp.json()
        assert data["prediction"] in ("good", "defective")
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["anomaly_score"] >= 0.0
        assert isinstance(data["latency_ms"], float)

    def test_predict_invalid_file_type(self, client_with_mock):
        resp = client_with_mock.post(
            "/predict",
            files={"file": ("doc.txt", b"not an image", "text/plain")},
        )
        assert resp.status_code == 422


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client_with_mock):
        resp = client_with_mock.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_schema(self, client_with_mock):
        resp = client_with_mock.get("/metrics")
        data = resp.json()
        assert "requests_served" in data
        assert "avg_latency_ms" in data
        assert "defect_rate_last_100" in data
