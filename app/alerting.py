"""
Webhook alerting for drift events.

Posts a JSON payload to ALERT_WEBHOOK_URL when drift is detected.
Compatible with Slack incoming webhooks and any generic HTTP receiver.
Set ALERT_WEBHOOK_URL in the environment to enable; if unset, alerts
are logged but not sent.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request
from datetime import datetime, timezone

log = logging.getLogger(__name__)

ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "")


def _build_payload(feature_name: str, event_count: int, model_version: str | None) -> dict:
    return {
        "text": (
            f"[ML Drift Monitor] Drift detected on feature '{feature_name}'. "
            f"{event_count} event(s) in current window. "
            f"Active model: {model_version or 'unknown'}."
        ),
        "feature": feature_name,
        "event_count": event_count,
        "model_version": model_version,
        "detected_at": datetime.now(timezone.utc).isoformat(),
    }


def send_drift_alert(feature_name: str, event_count: int, model_version: str | None = None) -> bool:
    """
    Send a drift alert to the configured webhook.
    Returns True if the request succeeded, False otherwise.
    Silently skips if ALERT_WEBHOOK_URL is not set.
    """
    if not ALERT_WEBHOOK_URL:
        log.debug("ALERT_WEBHOOK_URL not set -- skipping alert for '%s'", feature_name)
        return False

    payload = _build_payload(feature_name, event_count, model_version)
    data = json.dumps(payload).encode("utf-8")

    try:
        req = urllib.request.Request(
            ALERT_WEBHOOK_URL,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            ok = resp.status < 300
            if ok:
                log.info("Alert sent for drift on '%s'", feature_name)
            else:
                log.warning("Alert POST returned %d for '%s'", resp.status, feature_name)
            return ok
    except Exception as exc:
        log.warning("Alert POST failed for '%s': %s", feature_name, exc)
        return False


def send_retrain_alert(version: str, trigger: str, metrics: dict) -> bool:
    """Send a notification when a new model version is promoted."""
    if not ALERT_WEBHOOK_URL:
        return False

    payload = {
        "text": (
            f"[ML Drift Monitor] Retraining complete. "
            f"New model {version} promoted (trigger={trigger}). "
            f"Metrics: {metrics}."
        ),
        "version": version,
        "trigger": trigger,
        "metrics": metrics,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
    }
    data = json.dumps(payload).encode("utf-8")

    try:
        req = urllib.request.Request(
            ALERT_WEBHOOK_URL,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status < 300
    except Exception as exc:
        log.warning("Retrain alert POST failed: %s", exc)
        return False
