from __future__ import annotations
import mlflow
import os
from datetime import datetime
from dotenv import load_dotenv

from app.model_store import ModelStore
from app.models import ModelVersion
from app.database import SessionLocal

load_dotenv()

mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "postgresql://localhost/mlplatform")
)


def retrain_and_promote(trigger: str = "drift") -> dict:
    """
    Retrains the model, logs to MLflow (P2 experiment tracker),
    registers new version, promotes to active.
    """
    store = ModelStore()
    version = f"v_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    print(f"Retraining → {version} (trigger={trigger})")
    metrics = store.retrain(version=version)

    mlflow.set_experiment("drift_monitor_retraining")
    with mlflow.start_run(run_name=version) as run:
        mlflow.log_params({"trigger": trigger, "version": version})
        mlflow.log_metrics(metrics)
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

    print(f"Promoted {version} — metrics: {metrics}")
    return {"version": version, "run_id": run_id, "metrics": metrics}
