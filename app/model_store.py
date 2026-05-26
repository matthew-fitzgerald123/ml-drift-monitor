from __future__ import annotations
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(exist_ok=True)


class ModelStore:
    def __init__(self):
        self.model = None
        self.version = None

    def load(self, version: str = "latest"):
        path = MODEL_DIR / f"{version}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Model {version} not found at {path}")
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self.version = version
        return self

    def save(self, model, version: str):
        path = MODEL_DIR / f"{version}.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        self.model = model
        self.version = version
        return path

    def predict(self, features: list[float]):
        if self.model is None:
            raise RuntimeError("No model loaded")
        X = np.array(features).reshape(1, -1)
        prob = float(self.model.predict_proba(X)[0][1])
        pred = float(self.model.predict(X)[0])
        return pred, prob

    def retrain(self, version: str, n_samples: int = 1000) -> dict:
        """
        Retrain the model. Tries to fetch real feature data from the P2
        feature store first; falls back to synthetic data if unavailable.
        """
        from app.feature_store_client import fetch_training_data

        real_data = fetch_training_data()
        if real_data is not None:
            X, y = real_data
            data_source = "p2_feature_store"
        else:
            X, y = make_classification(
                n_samples=n_samples,
                n_features=8,
                random_state=np.random.randint(0, 9999),
            )
            data_source = "synthetic"

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = LogisticRegression(max_iter=500)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy":    round(accuracy_score(y_test, preds), 4),
            "auc_roc":     round(roc_auc_score(y_test, proba), 4),
            "data_source": data_source,
            "n_train":     len(X_train),
        }
        self.save(model, version)
        return metrics


# Global instance
store = ModelStore()
