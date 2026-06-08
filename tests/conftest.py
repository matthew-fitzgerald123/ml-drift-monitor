import os
import tempfile
import pytest


def pytest_configure(config):
    """Set MLflow to SQLite before any module import so no Postgres is touched."""
    _tmp = tempfile.mkdtemp()
    os.environ.setdefault("MLFLOW_TRACKING_URI", f"sqlite:///{_tmp}/mlflow.db")
