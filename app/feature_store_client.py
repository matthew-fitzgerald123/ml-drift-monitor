"""
HTTP client for pulling historical features from the P2 feature store.

Fetches entity feature vectors from GET /features/{feature_set}/history
and assembles them into a training-ready numpy array. Falls back to
synthetic data if the feature store is unreachable or returns no rows.
"""
from __future__ import annotations

import logging
import os
import urllib.request
import json

import numpy as np

log = logging.getLogger(__name__)

P2_API_URL = os.getenv("P2_API_URL", "http://localhost:8080")
P2_FEATURE_SET = os.getenv("P2_FEATURE_SET", "financial_signals")
FEATURE_NAMES = ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7"]
P2_FEATURE_MAP = os.getenv(
    "P2_FEATURE_MAP",
    "debt_to_income,credit_score,months_employed",
)


def _fetch_historical(feature_set: str, limit: int = 500) -> list[dict]:
    url = f"{P2_API_URL}/features/{feature_set}/history?entity_ids=&limit={limit}"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except Exception as exc:
        log.warning("Could not reach P2 feature store at %s: %s", P2_API_URL, exc)
        return []


def fetch_training_data(
    feature_set: str | None = None,
    min_rows: int = 50,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Pull historical feature rows from P2 and return (X, y).

    The P2 feature store does not store labels; we derive a synthetic
    binary label from a risk threshold so the model trains on real
    feature distributions rather than fully synthetic data.

    Returns None if the feature store is unavailable or has too few rows,
    so callers can fall back to synthetic generation.
    """
    fs = feature_set or P2_FEATURE_SET
    rows = _fetch_historical(fs)

    if len(rows) < min_rows:
        log.info(
            "P2 feature store returned %d rows (need %d) -- will use synthetic fallback",
            len(rows), min_rows,
        )
        return None

    mapped_cols = [c.strip() for c in P2_FEATURE_MAP.split(",") if c.strip()]
    X_list, y_list = [], []

    for row in rows:
        vec = []
        for col in mapped_cols:
            val = row.get(col)
            if val is None:
                break
            try:
                vec.append(float(val))
            except (TypeError, ValueError):
                break
        else:
            # Pad or truncate to 8 features
            vec = vec[:8]
            while len(vec) < 8:
                vec.append(0.0)

            # Derive label: high debt-to-income or low credit score -> risk=1
            dti = float(row.get("debt_to_income", 0.5))
            score = float(row.get("credit_score", 700))
            label = 1 if (dti > 0.4 or score < 650) else 0

            X_list.append(vec)
            y_list.append(label)

    if len(X_list) < min_rows:
        log.info(
            "Only %d usable rows after parsing -- using synthetic fallback",
            len(X_list),
        )
        return None

    log.info("Loaded %d training rows from P2 feature store (%s)", len(X_list), fs)
    return np.array(X_list, dtype=float), np.array(y_list, dtype=int)
