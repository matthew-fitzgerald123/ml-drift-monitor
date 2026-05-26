"""
Scheduled drift monitor using APScheduler.

Every DRIFT_CHECK_INTERVAL minutes it queries recent drift events. If the
count in that window exceeds DRIFT_RETRAIN_THRESHOLD, retraining is triggered
automatically and detectors are reset.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app.database import SessionLocal
from app.models import DriftEvent

log = logging.getLogger(__name__)

DRIFT_CHECK_INTERVAL = int(os.getenv("DRIFT_CHECK_INTERVAL", "15"))   # minutes
DRIFT_RETRAIN_THRESHOLD = int(os.getenv("DRIFT_RETRAIN_THRESHOLD", "5"))


class DriftScheduler:
    def __init__(self, model_store, feature_detector, prediction_detector):
        self.store = model_store
        self.fd = feature_detector
        self.pd = prediction_detector
        self._scheduler = AsyncIOScheduler()
        self._last_check: datetime | None = None
        self._check_count = 0
        self._retrain_count = 0

    def start(self):
        self._scheduler.add_job(
            self._check_drift,
            IntervalTrigger(minutes=DRIFT_CHECK_INTERVAL),
            id="drift_check",
            replace_existing=True,
            misfire_grace_time=60,
        )
        self._scheduler.start()
        log.info(
            "Drift scheduler started (interval=%dm, threshold=%d)",
            DRIFT_CHECK_INTERVAL,
            DRIFT_RETRAIN_THRESHOLD,
        )

    def stop(self):
        self._scheduler.shutdown(wait=False)
        log.info("Drift scheduler stopped")

    def status(self) -> dict:
        job = self._scheduler.get_job("drift_check")
        return {
            "running": self._scheduler.running,
            "interval_minutes": DRIFT_CHECK_INTERVAL,
            "retrain_threshold": DRIFT_RETRAIN_THRESHOLD,
            "last_check": self._last_check.isoformat() if self._last_check else None,
            "next_check": job.next_run_time.isoformat() if job and job.next_run_time else None,
            "checks_run": self._check_count,
            "auto_retrains": self._retrain_count,
        }

    async def _check_drift(self):
        log.info("Scheduled drift check running")
        self._check_count += 1
        self._last_check = datetime.now(timezone.utc)

        window_start = datetime.now(timezone.utc) - timedelta(minutes=DRIFT_CHECK_INTERVAL)

        db = SessionLocal()
        try:
            recent_events = (
                db.query(DriftEvent)
                .filter(
                    DriftEvent.detected == True,
                    DriftEvent.created_at >= window_start,
                )
                .count()
            )
        finally:
            db.close()

        log.info("Drift events in last %dm: %d (threshold=%d)",
                 DRIFT_CHECK_INTERVAL, recent_events, DRIFT_RETRAIN_THRESHOLD)

        if recent_events >= DRIFT_RETRAIN_THRESHOLD:
            log.warning(
                "Drift threshold exceeded (%d >= %d) — triggering auto-retrain",
                recent_events, DRIFT_RETRAIN_THRESHOLD,
            )
            await self._auto_retrain()

    async def _auto_retrain(self):
        import asyncio
        from app.retraining import retrain_and_promote
        from app.alerting import send_retrain_alert
        try:
            result = await asyncio.to_thread(retrain_and_promote, trigger="scheduled")
            self.store.load(result["version"])
            self.fd.reset_all()
            self.pd.reset()
            self._retrain_count += 1
            log.info("Auto-retrain complete, promoted %s", result["version"])
            send_retrain_alert(result["version"], "scheduled", result.get("metrics", {}))
        except Exception:
            log.exception("Auto-retrain failed")
