import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.api.api import app
from src.api.drift_router import set_model


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert "total_predictions" in data


def test_model_info(client, monkeypatch):
    monkeypatch.setattr("src.api.api.model", None)
    response = client.get("/model-info")
    assert response.status_code == 503
    assert response.json()["detail"] == "Model not loaded"


def test_predict_endpoint(client, monkeypatch):
    mock_model = MagicMock()
    mock_model.predict.return_value = {
        "text": "тест",
        "cleaned_text": "тест",
        "toxic_prob": 0.9,
        "label": "toxic",
        "confidence": 0.9,
    }
    monkeypatch.setattr("src.api.api.model", mock_model)

    payload = {"texts": ["тест"], "threshold": 0.5}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 1
    assert data["predictions"][0]["label"] == "toxic"
    assert "latency_ms" in data


def test_predict_no_model(client, monkeypatch):
    monkeypatch.setattr("src.api.api.model", None)
    response = client.post("/predict", json={"texts": ["тест"]})
    assert response.status_code == 503
    assert response.json()["detail"] == "Model not loaded"


def test_feedback_not_found(client):
    payload = {"prediction_id": 99999, "is_correct": True}
    response = client.post("/feedback", json=payload)
    assert response.status_code == 404
    assert response.json()["detail"] == "Prediction not found"


def test_stats_endpoint(client, db_session, sample_prediction):
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["total_predictions"] >= 1
    assert "feedback_received" in data
    assert "toxic_rate" in data
    assert "recent_predictions" in data


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; charset=utf-8"
    assert "toxicity_predictions_total" in response.text


def test_drift_check_no_detector(client, monkeypatch):
    monkeypatch.setattr("src.api.drift_router.drift_detector", None)
    response = client.post("/drift/check")
    assert response.status_code == 200
    assert response.json()["detail"] == "Drift detector not initialized"


def test_drift_status(client, monkeypatch):
    mock_detector = MagicMock()
    mock_detector.get_drift_summary.return_value = {
        "status": "ok",
        "total_checks": 5,
        "drift_count": 1,
        "last_check": {"has_drift": False, "timestamp": 1234567890},
    }
    monkeypatch.setattr("src.api.drift_router.drift_detector", mock_detector)
    response = client.get("/drift/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["total_checks"] == 1
