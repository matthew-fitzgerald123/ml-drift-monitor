# ML Drift Monitor

A production-style ML serving and monitoring system. Serves predictions from a versioned RandomForest model, detects feature and prediction drift in the background after each inference, and can trigger automated retraining when drift is detected.

## Stack

| Component | Library |
|---|---|
| API | FastAPI + uvicorn (port 8081) |
| Drift detection | River 0.21.2 (ADWIN algorithm) |
| Model training | scikit-learn (RandomForest) |
| Experiment tracking | MLflow |
| Scheduling | APScheduler |
| Data store | PostgreSQL + SQLAlchemy |

## Setup

```bash
# Create database
createdb ml_drift_monitor

# Install dependencies
pip install -r requirements.txt

# Seed initial model (trains v1 RandomForest and saves to disk)
make seed
```

## Running

```bash
# Start API server (requires seed model)
make serve

# Run end-to-end demo
make demo

# Run tests
make test
```

## API Endpoints

### Inference

| Method | Path | Description |
|---|---|---|
| POST | `/predict` | Predict + log + trigger background drift check |

### Monitoring

| Method | Path | Description |
|---|---|---|
| GET | `/drift/events` | Recent drift events |
| GET | `/drift/summary` | Total drift count + active model version |
| GET | `/predictions/recent` | Recent prediction log |

### Model Management

| Method | Path | Description |
|---|---|---|
| POST | `/model/retrain` | Trigger manual retraining in background |
| GET | `/model/versions` | List all versioned models with metrics |
| GET | `/health` | Server status + model loaded state |

Interactive docs at `http://localhost:8081/docs`.

## How Drift Detection Works

Each prediction triggers a background task that feeds the 8 input features and the output probability into per-feature ADWIN detectors. ADWIN maintains a sliding window and signals drift when the mean shifts beyond a statistical threshold. When drift is detected, a `DriftEvent` is written to Postgres. Manual or scheduled retraining via `/model/retrain` trains a new model on fresh data, saves it with an incremented version, and resets all detectors.

## Project Structure

```
app/
  drift_detector.py   FeatureDriftDetector + PredictionDriftDetector (ADWIN)
  model_store.py      versioned model load/save
  retraining.py       retrain + promote logic
  main.py             FastAPI app
  models.py           SQLAlchemy models (PredictionLog, DriftEvent, ModelVersion)
  database.py         engine + session
scripts/
  seed_model.py       train and save initial v1 model
notebooks/
  demo.py             simulate predictions and trigger drift
models/               saved model artifacts (created by seed)
tests/
```
