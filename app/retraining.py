from __future__ import annotations
import json
import logging
import mlflow
import os
import urllib.request
from datetime import datetime
from dotenv import load_dotenv

from app.model_store import ModelStore, MODEL_DIR
from app.models import ModelVersion
from app.database import SessionLocal

load_dotenv()
log = logging.getLogger(__name__)

_DEFAULT_MLFLOW_URI = "postgresql://localhost/mlplatform"

P2_API_URL = os.getenv("P2_API_URL", "http://localhost:8080")
P2_MODEL_NAME = os.getenv("P2_MODEL_NAME", "drift-monitor-model")


def _register_in_p2(version: str, run_id: str, metrics: dict) -> bool:
    """POST to P2 model registry. Returns True on success, False otherwise."""
    artifact_uri = str(MODEL_DIR / f"{version}.pkl")
    payload = json.dumps({
        "name": P2_MODEL_NAME,
        "version": version,
        "artifact_uri": artifact_uri,
        "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        "params": {"mlflow_run_id": run_id, "source": "drift_monitor"},
    }).encode("utf-8")
    try:
        req = urllib.request.Request(
            f"{P2_API_URL}/models/register",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            ok = resp.status < 300
            if ok:
                log.info("Registered %s in P2 model registry", version)
            return ok
    except Exception as exc:
        log.warning("Could not register %s in P2 model registry: %s", version, exc)
        return False


def retrain_and_promote(trigger: str = "drift") -> dict:
    """
    Retrains the model, logs to MLflow (P2 experiment tracker),
    registers new version, promotes to active.
    """
    store = ModelStore()
    version = f"v_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    print(f"Retraining → {version} (trigger={trigger})")
    metrics = store.retrain(version=version)

    # Separate numeric metrics from string/int metadata for MLflow
    numeric_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    meta_params = {k: str(v) for k, v in metrics.items() if not isinstance(v, (int, float))}

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", _DEFAULT_MLFLOW_URI))
    mlflow.set_experiment("drift_monitor_retraining")
    with mlflow.start_run(run_name=version) as run:
        mlflow.log_params({"trigger": trigger, "version": version, **meta_params})
        mlflow.log_metrics(numeric_metrics)
        run_id = run.info.run_id

    db = SessionLocal()
    try:
        db.query(ModelVersion).update({"is_active": False})
        mv = ModelVersion(
            version=version,
            mlflow_run_id=run_id,
            metrics=metrics,
            is_active=True,
        )
        db.add(mv)
        db.commit()
    finally:
        db.close()

    _register_in_p2(version, run_id, metrics)

    print(f"Promoted {version}, metrics: {metrics}")
    return {"version": version, "run_id": run_id, "metrics": metrics}
