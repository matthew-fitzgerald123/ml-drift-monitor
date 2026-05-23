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
