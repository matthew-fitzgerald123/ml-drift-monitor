from __future__ import annotations
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel
import os
from dotenv import load_dotenv

from app.database import get_db, engine
from app.models import Base, PredictionLog, DriftEvent, ModelVersion
from app.model_store import store
from app.drift_detector import FeatureDriftDetector, PredictionDriftDetector
from app.retraining import retrain_and_promote
from app.scheduler import DriftScheduler
from app.alerting import send_drift_alert

load_dotenv()
Base.metadata.create_all(bind=engine)

FEATURE_NAMES = ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7"]
feature_detector = FeatureDriftDetector(feature_names=FEATURE_NAMES)
prediction_detector = PredictionDriftDetector()
_scheduler: DriftScheduler | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _scheduler
    try:
        store.load("v1")
        print(f"Loaded model: {store.version}")
    except FileNotFoundError:
        print("No model found — run: make seed")

    _scheduler = DriftScheduler(store, feature_detector, prediction_detector)
    _scheduler.start()
    yield
    _scheduler.stop()


app = FastAPI(title="ML Drift Monitor", version="1.0.0", lifespan=lifespan)


# ── Inference ─────────────────────────────────────────────

class PredictReq(BaseModel):
    entity_id: str
    features: dict[str, float]


@app.post("/predict", tags=["inference"])
def predict(req: PredictReq, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    if store.model is None:
        raise HTTPException(503, "No model loaded — run: make seed")

    feature_vec = [req.features.get(f, 0.0) for f in FEATURE_NAMES]
    prediction, probability = store.predict(feature_vec)

    log = PredictionLog(
        entity_id=req.entity_id,
        features=req.features,
        prediction=prediction,
        probability=probability,
        model_version=store.version or "unknown",
    )
    db.add(log)
    db.commit()

    background_tasks.add_task(
        check_and_log_drift,
        req.features,
        probability,
        db_url=os.getenv("DATABASE_URL"),
    )

    return {
        "entity_id":     req.entity_id,
        "prediction":    int(prediction),
        "probability":   round(probability, 4),
        "model_version": store.version,
    }


def check_and_log_drift(features: dict, probability: float, db_url: str):
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        drift_results = feature_detector.update_all(features)
        pred_drift = prediction_detector.update(probability)

        for fname, detected in drift_results.items():
            if detected:
                db.add(DriftEvent(
                    feature_name=fname,
                    drift_score=1.0,
                    threshold=feature_detector.threshold,
                    detected=True,
                    window_size=0,
                ))
                send_drift_alert(fname, 1, store.version)

        if pred_drift:
            db.add(DriftEvent(
                feature_name="prediction_probability",
                drift_score=1.0,
                threshold=0.05,
                detected=True,
                window_size=0,
            ))
            send_drift_alert("prediction_probability", 1, store.version)

        db.commit()
    finally:
        db.close()


# ── Monitoring ────────────────────────────────────────────

@app.get("/drift/events", tags=["monitoring"])
def get_drift_events(limit: int = 50, db: Session = Depends(get_db)):
    events = (
        db.query(DriftEvent)
        .filter(DriftEvent.detected == True)
        .order_by(DriftEvent.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "feature":     e.feature_name,
            "drift_score": e.drift_score,
            "detected_at": str(e.created_at),
        }
        for e in events
    ]


@app.get("/drift/summary", tags=["monitoring"])
def drift_summary(db: Session = Depends(get_db)):
    total = db.query(DriftEvent).filter(DriftEvent.detected == True).count()
    recent = (
        db.query(DriftEvent)
        .filter(DriftEvent.detected == True)
        .order_by(DriftEvent.created_at.desc())
        .first()
    )
    return {
        "total_drift_events": total,
        "last_drift_at":      str(recent.created_at) if recent else None,
        "active_model":       store.version,
    }


@app.get("/predictions/recent", tags=["monitoring"])
def recent_predictions(limit: int = 20, db: Session = Depends(get_db)):
    logs = (
        db.query(PredictionLog)
        .order_by(PredictionLog.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "entity_id":     l.entity_id,
            "prediction":    l.prediction,
            "probability":   l.probability,
            "model_version": l.model_version,
            "created_at":    str(l.created_at),
        }
        for l in logs
    ]


# ── Model management ──────────────────────────────────────

@app.post("/model/retrain", tags=["model"])
def trigger_retrain(background_tasks: BackgroundTasks):
    background_tasks.add_task(_retrain_and_reload)
    return {"status": "retraining started"}


def _retrain_and_reload():
    result = retrain_and_promote(trigger="manual")
    store.load(result["version"])
    feature_detector.reset_all()
    prediction_detector.reset()
    print(f"Reloaded model: {result['version']}")


@app.get("/model/versions", tags=["model"])
def list_versions(db: Session = Depends(get_db)):
    versions = db.query(ModelVersion).order_by(ModelVersion.created_at.desc()).all()
    return [
        {
            "version":    v.version,
            "is_active":  v.is_active,
            "metrics":    v.metrics,
            "created_at": str(v.created_at),
        }
        for v in versions
    ]


# ── Scheduler ─────────────────────────────────────────────

@app.get("/scheduler/status", tags=["monitoring"])
def scheduler_status():
    if _scheduler is None:
        raise HTTPException(503, "Scheduler not initialised")
    return _scheduler.status()


# ── Health ────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":        "ok",
        "model_version": store.version,
        "model_loaded":  store.model is not None,
    }
