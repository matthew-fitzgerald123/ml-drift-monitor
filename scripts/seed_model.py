"""
Seeds the initial model version so the API starts with something to serve.
Run: make seed
"""
import sys
sys.path.insert(0, ".")

from app.model_store import ModelStore
from app.database import engine
from app.models import Base, ModelVersion
from sqlalchemy.orm import Session

Base.metadata.create_all(bind=engine)

ms = ModelStore()
metrics = ms.retrain(version="v1")
print(f"Trained v1 — metrics: {metrics}")

with Session(engine) as db:
    db.query(ModelVersion).update({"is_active": False})
    existing = db.query(ModelVersion).filter_by(version="v1").first()
    if existing:
        existing.metrics = metrics
        existing.is_active = True
    else:
        db.add(ModelVersion(version="v1", metrics=metrics, is_active=True))
    db.commit()

print("Seeded model v1 as active.")
