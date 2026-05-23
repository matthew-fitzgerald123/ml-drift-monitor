from __future__ import annotations
from sqlalchemy import Column, String, DateTime, JSON, Integer, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    id            = Column(Integer, primary_key=True, autoincrement=True)
    entity_id     = Column(String, nullable=False, index=True)
    features      = Column(JSON, nullable=False)
    prediction    = Column(Float, nullable=False)
    probability   = Column(Float, nullable=False)
    model_version = Column(String, nullable=False)
    created_at    = Column(DateTime, default=datetime.utcnow, index=True)

class DriftEvent(Base):
    __tablename__ = "drift_events"
    id           = Column(Integer, primary_key=True, autoincrement=True)
    feature_name = Column(String, nullable=False)
    drift_score  = Column(Float, nullable=False)
    threshold    = Column(Float, nullable=False)
    detected     = Column(Boolean, nullable=False)
    window_size  = Column(Integer, nullable=False)
    created_at   = Column(DateTime, default=datetime.utcnow, index=True)

class ModelVersion(Base):
    __tablename__ = "model_versions"
    id            = Column(Integer, primary_key=True, autoincrement=True)
    version       = Column(String, unique=True, nullable=False)
    mlflow_run_id = Column(String, nullable=True)
    metrics       = Column(JSON, nullable=True)
    is_active     = Column(Boolean, default=False)
    created_at    = Column(DateTime, default=datetime.utcnow)
