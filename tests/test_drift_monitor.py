from __future__ import annotations
import pytest
from fastapi.testclient import TestClient

import sys
sys.path.insert(0, ".")
from scripts.seed_model import *  # noqa — seeds DB + model file

from app.main import app

SAMPLE_FEATURES = {f"f{i}": round(float(i) * 0.1, 2) for i in range(8)}


@pytest.fixture(scope="module")
def client():
    # Context-manager form triggers on_event("startup") so the model loads.
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict(client):
    r = client.post("/predict", json={
        "entity_id": "test_user_001",
        "features": SAMPLE_FEATURES,
    })
    assert r.status_code == 200
    data = r.json()
    assert "prediction" in data
    assert "probability" in data
    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_logs_to_db(client):
    client.post("/predict", json={
        "entity_id": "test_user_002",
        "features": SAMPLE_FEATURES,
    })
    r = client.get("/predictions/recent?limit=5")
    assert r.status_code == 200
    entity_ids = [p["entity_id"] for p in r.json()]
    assert "test_user_002" in entity_ids


def test_drift_summary(client):
    r = client.get("/drift/summary")
    assert r.status_code == 200
    data = r.json()
    assert "total_drift_events" in data
    assert "active_model" in data


def test_drift_events_endpoint(client):
    r = client.get("/drift/events")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_model_versions(client):
    r = client.get("/model/versions")
    assert r.status_code == 200
    versions = r.json()
    assert len(versions) >= 1
    assert any(v["is_active"] for v in versions)


def test_retrain_trigger(client):
    r = client.post("/model/retrain")
    assert r.status_code == 200
    assert r.json()["status"] == "retraining started"


def test_scheduler_status(client):
    r = client.get("/scheduler/status")
    assert r.status_code == 200
    data = r.json()
    assert data["running"] is True
    assert "interval_minutes" in data
    assert "retrain_threshold" in data
    assert "checks_run" in data


def test_drift_features_endpoint(client):
    r = client.get("/drift/features")
    assert r.status_code == 200
    data = r.json()
    assert "threshold" in data
    assert "drifting_count" in data
    assert isinstance(data["features"], list)
    assert len(data["features"]) == 8
    sample = data["features"][0]
    for key in ("feature", "drifting", "window_width", "mean_estimate", "variance", "n_detections"):
        assert key in sample


def test_feature_detector_status_reports_drift():
    from app.drift_detector import FeatureDriftDetector
    det = FeatureDriftDetector(feature_names=["a", "b"])
    # Feed a stable stream then a sharp shift on "a" to trip ADWIN.
    for _ in range(100):
        det.update("a", 0.0)
    for _ in range(100):
        det.update("a", 5.0)
    for _ in range(100):
        det.update("b", 0.5)

    status = {s["feature"]: s for s in det.status()}
    assert status["a"]["drifting"] is True
    assert status["a"]["mean_estimate"] > 1.0
    assert status["b"]["drifting"] is False
    assert status["b"]["window_width"] > 0


def test_feature_detector_reset_clears_drift_flag():
    from app.drift_detector import FeatureDriftDetector
    det = FeatureDriftDetector(feature_names=["a"])
    for _ in range(100):
        det.update("a", 0.0)
    for _ in range(100):
        det.update("a", 5.0)
    assert det.status()[0]["drifting"] is True
    det.reset_all()
    assert det.status()[0]["drifting"] is False
    assert det.status()[0]["window_width"] == 0


def test_alert_skips_when_no_webhook(monkeypatch):
    from app.alerting import send_drift_alert
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "")
    import importlib, app.alerting as al
    monkeypatch.setattr(al, "ALERT_WEBHOOK_URL", "")
    result = send_drift_alert("f0", 3, "v1")
    assert result is False


def test_alert_payload_shape():
    from app.alerting import _build_payload
    payload = _build_payload("f3", 7, "v2")
    assert payload["feature"] == "f3"
    assert payload["event_count"] == 7
    assert payload["model_version"] == "v2"
    assert "detected_at" in payload
    assert "text" in payload


def test_feature_store_client_returns_none_when_unavailable(monkeypatch):
    from app import feature_store_client as fsc
    monkeypatch.setattr(fsc, "P2_API_URL", "http://localhost:19999")
    result = fsc.fetch_training_data()
    assert result is None


def test_retrain_falls_back_to_synthetic(monkeypatch):
    from app import feature_store_client as fsc
    monkeypatch.setattr(fsc, "P2_API_URL", "http://localhost:19999")
    from app.model_store import ModelStore
    ms = ModelStore()
    metrics = ms.retrain("test_fallback")
    assert metrics["data_source"] == "synthetic"
    assert metrics["accuracy"] > 0


def test_p2_registry_skips_when_unavailable(monkeypatch):
    import app.retraining as rt
    monkeypatch.setattr(rt, "P2_API_URL", "http://localhost:19999")
    result = rt._register_in_p2("v_test", "run123", {"accuracy": 0.9})
    assert result is False
