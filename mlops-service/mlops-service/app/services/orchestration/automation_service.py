from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Optional

from loguru import logger

from app.services.orchestration.training_service import create_job, run_pipeline_task
from app.services.monitoring.drift_detection import DriftDetectionService, DriftStatus
from app.services.monitoring.prometheus_metrics import (
    AUTOMATION_SCHEDULER_RUNNING,
    DRIFT_DETECTED,
    DRIFT_PSI_SCORE,
    sanitize_label_value,
)
from app.services.platform.automation_history import AutomationHistory
from app.services.registry.model_registry import ModelRegistryService
from app.services.monitoring.evidently_report import EvidentlyReportGenerator
from app.config.settings import Settings


class AutomationService:
    def __init__(self, artifacts_root, reports_dir):
        self._lock = threading.Lock()
        self._scheduler_thread: Optional[threading.Thread] = None
        self._running = False
        self._last_run: dict[str, str] = {}
        self._artifacts_root = artifacts_root
        self._reports_dir = reports_dir
        self._settings = Settings()
        self._registry = ModelRegistryService(artifacts_root / "model_registry")
        self._history = AutomationHistory(
            artifacts_root / "monitoring" / "automation_history.json"
        )
        self._drift_service = DriftDetectionService(
            artifacts_root=artifacts_root,
            reports_dir=reports_dir,
        )
        self._evidently = EvidentlyReportGenerator(reports_dir)

    def start_scheduler(self, interval_hours: int = 24) -> None:
        with self._lock:
            if self._running:
                logger.info("Automation scheduler already running")
                return
            self._running = True
            self._scheduler_thread = threading.Thread(
                target=self._run_loop,
                args=(interval_hours,),
                daemon=True,
            )
            self._scheduler_thread.start()
            AUTOMATION_SCHEDULER_RUNNING.set(1)
            logger.info(f"Automation scheduler started: interval={interval_hours}h")

    def stop_scheduler(self) -> None:
        with self._lock:
            self._running = False
            AUTOMATION_SCHEDULER_RUNNING.set(0)
            logger.info("Automation scheduler stopped")

    def _run_loop(self, interval_hours: int) -> None:
        while self._running:
            try:
                self._run_scheduled_jobs()
            except Exception as e:
                logger.warning(f"Automation scheduler error: {e}")
            time.sleep(interval_hours * 3600)

    def _run_scheduled_jobs(self) -> None:
        now = datetime.utcnow()
        last_run_time = self._last_run.get("scheduled")
        if last_run_time:
            last_run = datetime.fromisoformat(last_run_time)
            if now - last_run < timedelta(hours=self._settings.retrain_cooldown_hours):
                return

        if not self._passes_gate("imaging") or not self._passes_gate("clinical"):
            self._history.record(
                "scheduled_retrain_skipped",
                {"reason": "gate_failed"},
            )
            return

        logger.info("Running scheduled retraining job")
        job_id = create_job("both")
        threading.Thread(
            target=run_pipeline_task,
            args=(job_id, "both"),
            daemon=True,
        ).start()
        self._last_run["scheduled"] = now.isoformat()
        self._history.record(
            "scheduled_retrain",
            {"job_id": job_id, "pipeline": "both"},
        )

    def trigger_drift_retraining(
        self,
        reference_csv,
        current_csv,
        pipeline: str = "both",
        psi_threshold: float = 0.3,
    ) -> dict:
        report = self._drift_service.check_drift(
            reference_csv=reference_csv,
            current_csv=current_csv,
            pipeline=pipeline,
        )

        DRIFT_DETECTED.labels(pipeline=pipeline).set(1 if report.drift_detected else 0)
        DRIFT_PSI_SCORE.labels(pipeline=pipeline, feature="_overall").set(
            report.overall_psi
        )
        for feature in report.feature_results:
            DRIFT_PSI_SCORE.labels(
                pipeline=pipeline, feature=sanitize_label_value(feature.feature_name)
            ).set(feature.psi)

        try:
            self._evidently.run_drift_and_emit(
                pipeline=pipeline,
                reference_csv=reference_csv,
                current_csv=current_csv,
            )
        except Exception as e:
            logger.warning(f"Evidently drift check failed (non-fatal): {e}")

        if (
            report.status == DriftStatus.DRIFT_DETECTED
            and report.overall_psi >= psi_threshold
        ):
            now = datetime.utcnow()
            last_run_time = self._last_run.get("drift")
            if last_run_time:
                last_run = datetime.fromisoformat(last_run_time)
                if now - last_run < timedelta(
                    hours=self._settings.retrain_cooldown_hours
                ):
                    self._history.record(
                        "drift_retrain_skipped",
                        {
                            "pipeline": pipeline,
                            "psi": report.overall_psi,
                            "threshold": psi_threshold,
                            "reason": "cooldown",
                        },
                    )
                    return {"status": "no_retraining", "psi": report.overall_psi}

            if not self._passes_gate(pipeline):
                self._history.record(
                    "drift_retrain_skipped",
                    {
                        "pipeline": pipeline,
                        "psi": report.overall_psi,
                        "threshold": psi_threshold,
                        "reason": "gate_failed",
                    },
                )
                return {"status": "no_retraining", "psi": report.overall_psi}

            logger.info(
                f"Drift detected (psi={report.overall_psi:.4f}), triggering retraining"
            )
            job_id = create_job(pipeline)
            threading.Thread(
                target=run_pipeline_task,
                args=(job_id, pipeline),
                daemon=True,
            ).start()
            self._history.record(
                "drift_retrain_triggered",
                {
                    "job_id": job_id,
                    "pipeline": pipeline,
                    "psi": report.overall_psi,
                },
            )
            self._last_run["drift"] = now.isoformat()
            return {
                "status": "retraining_triggered",
                "job_id": job_id,
                "psi": report.overall_psi,
            }

        self._history.record(
            "drift_retrain_skipped",
            {
                "pipeline": pipeline,
                "psi": report.overall_psi,
                "threshold": psi_threshold,
            },
        )
        return {
            "status": "no_retraining",
            "psi": report.overall_psi,
        }

    def _passes_gate(self, pipeline: str) -> bool:
        metric = (
            self._settings.retrain_imaging_metric
            if pipeline == "imaging"
            else self._settings.retrain_clinical_metric
        )
        threshold = self._settings.retrain_min_improvement

        current_metrics = self._registry.get_production_metrics(pipeline)
        current_value = float(current_metrics.get(metric, 0.0))

        history_file = (
            self._artifacts_root / "monitoring" / "last_training_metrics.json"
        )
        if not history_file.exists():
            return True

        with open(history_file) as f:
            last_metrics = json.load(f)

        new_value = float(last_metrics.get(pipeline, {}).get(metric, 0.0))

        if new_value - current_value >= threshold:
            return True
        self._history.record(
            "retrain_gate_failed",
            {
                "pipeline": pipeline,
                "metric": metric,
                "current": current_value,
                "new": new_value,
                "threshold": threshold,
            },
        )
        return False


@lru_cache(maxsize=1)
def get_automation_service(artifacts_root: str, reports_dir: str) -> AutomationService:
    """Get automation service instance (cached singleton)."""
    from pathlib import Path

    return AutomationService(Path(artifacts_root), Path(reports_dir))
