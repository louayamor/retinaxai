import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

DRIFT_THRESHOLD_PSI = 0.3
FEATURE_DRIFT_THRESHOLD_PSI = 0.1


class DriftDetectionError(Exception):
    pass


class DriftStatus:
    NO_DRIFT = "no_drift"
    DRIFT_DETECTED = "drift_detected"
    UNKNOWN = "unknown"


class FeatureDriftResult:
    def __init__(
        self,
        feature_name: str,
        psi: float,
        drift_detected: bool,
        reference_mean: float,
        current_mean: float,
    ):
        self.feature_name = feature_name
        self.psi = psi
        self.drift_detected = drift_detected
        self.reference_mean = reference_mean
        self.current_mean = current_mean


class DriftReport:
    def __init__(
        self,
        pipeline: str,
        status: str,
        overall_psi: float,
        drift_detected: bool,
        feature_results: list[FeatureDriftResult],
        reference_samples: int,
        current_samples: int,
        generated_at: str,
    ):
        self.pipeline = pipeline
        self.status = status
        self.overall_psi = overall_psi
        self.drift_detected = drift_detected
        self.feature_results = feature_results
        self.reference_samples = reference_samples
        self.current_samples = current_samples
        self.generated_at = generated_at

    def to_dict(self) -> dict:
        return {
            "pipeline": self.pipeline,
            "status": self.status,
            "overall_psi": self.overall_psi,
            "drift_detected": self.drift_detected,
            "reference_samples": self.reference_samples,
            "current_samples": self.current_samples,
            "generated_at": self.generated_at,
            "features": [
                {
                    "feature_name": f.feature_name,
                    "psi": f.psi,
                    "drift_detected": f.drift_detected,
                    "reference_mean": f.reference_mean,
                    "current_mean": f.current_mean,
                }
                for f in self.feature_results
            ],
        }

    @property
    def num_features_drifted(self) -> int:
        return sum(1 for f in self.feature_results if f.drift_detected)


class DriftDetectionService:
    def __init__(self, artifacts_root: Path, reports_dir: Path):
        self.artifacts_root = artifacts_root
        self.reports_dir = reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self._load_history()

    def _load_history(self) -> None:
        history_file = self.reports_dir / "drift_history.json"
        if history_file.exists():
            with open(history_file) as f:
                self._history: list[dict] = json.load(f)
        else:
            self._history = []

    def _save_history(self) -> None:
        history_file = self.reports_dir / "drift_history.json"
        with open(history_file, "w") as f:
            json.dump(self._history, f, indent=2)

    def _calculate_psi(
        self, reference: np.ndarray, current: np.ndarray, n_bins: int = 10
    ) -> float:
        if len(reference) == 0 or len(current) == 0:
            return 0.0

        try:
            min_val = min(reference.min(), current.min())
            max_val = max(reference.max(), current.max())

            if min_val == max_val:
                return 0.0

            bins = np.linspace(min_val, max_val, n_bins + 1)
            ref_hist, _ = np.histogram(reference, bins=bins)
            curr_hist, _ = np.histogram(current, bins=bins)

            ref_dist = ref_hist / len(reference)
            curr_dist = curr_hist / len(current)

            ref_dist = np.where(ref_dist == 0, 1e-6, ref_dist)
            curr_dist = np.where(curr_dist == 0, 1e-6, curr_dist)

            psi = np.sum((curr_dist - ref_dist) * np.log(curr_dist / ref_dist))

            return float(psi)

        except Exception as e:
            logger.warning(f"PSI calculation error: {e}")
            return 0.0

    def check_drift(
        self,
        reference_csv: Path,
        current_csv: Path,
        pipeline: str = "imaging",
    ) -> DriftReport:
        if not reference_csv.exists():
            raise DriftDetectionError(f"Reference data not found: {reference_csv}")
        if not current_csv.exists():
            raise DriftDetectionError(f"Current data not found: {current_csv}")

        reference_df = pd.read_csv(reference_csv)
        current_df = pd.read_csv(current_csv)

        numeric_cols = reference_df.select_dtypes(include=[np.number]).columns.tolist()

        if "level" in numeric_cols:
            numeric_cols.remove("level")

        feature_results = []
        overall_psi_values = []

        for col in numeric_cols:
            if col in current_df.columns:
                ref_vals = reference_df[col].dropna().values
                curr_vals = current_df[col].dropna().values

                psi = self._calculate_psi(ref_vals, curr_vals)

                if len(ref_vals) > 0:
                    ref_mean = float(ref_vals.mean())
                else:
                    ref_mean = 0.0

                if len(curr_vals) > 0:
                    curr_mean = float(curr_vals.mean())
                else:
                    curr_mean = 0.0

                drift_detected = psi > FEATURE_DRIFT_THRESHOLD_PSI

                feature_results.append(
                    FeatureDriftResult(
                        feature_name=col,
                        psi=psi,
                        drift_detected=drift_detected,
                        reference_mean=ref_mean,
                        current_mean=curr_mean,
                    )
                )

                if drift_detected:
                    overall_psi_values.append(psi)

        overall_psi = float(np.mean(overall_psi_values)) if overall_psi_values else 0.0

        drift_detected = overall_psi > DRIFT_THRESHOLD_PSI

        status = DriftStatus.DRIFT_DETECTED if drift_detected else DriftStatus.NO_DRIFT

        report = DriftReport(
            pipeline=pipeline,
            status=status,
            overall_psi=overall_psi,
            drift_detected=drift_detected,
            feature_results=feature_results,
            reference_samples=len(reference_df),
            current_samples=len(current_df),
            generated_at=datetime.utcnow().isoformat(),
        )

        self._history.append(report.to_dict())
        if len(self._history) > 100:
            self._history = self._history[-100:]
        self._save_history()

        logger.info(
            f"Drift check complete: {pipeline} - status={status}, psi={overall_psi:.4f}"
        )

        return report

    def get_latest_drift(self, pipeline: str = "imaging") -> Optional[DriftReport]:
        for history in reversed(self._history):
            if history.get("pipeline") == pipeline:
                return self._history_entry_to_report(history)
        return None

    def get_drift_history(
        self, pipeline: Optional[str] = None, limit: int = 10
    ) -> list[dict]:
        if pipeline:
            return [h for h in self._history if h.get("pipeline") == pipeline][:limit]
        return self._history[:limit]

    def _history_entry_to_report(self, entry: dict) -> DriftReport:
        feature_results = []
        for f in entry.get("features", []):
            feature_results.append(
                FeatureDriftResult(
                    feature_name=f["feature_name"],
                    psi=f["psi"],
                    drift_detected=f["drift_detected"],
                    reference_mean=f["reference_mean"],
                    current_mean=f["current_mean"],
                )
            )
        return DriftReport(
            pipeline=entry["pipeline"],
            status=entry["status"],
            overall_psi=entry["overall_psi"],
            drift_detected=entry["drift_detected"],
            feature_results=feature_results,
            reference_samples=entry["reference_samples"],
            current_samples=entry["current_samples"],
            generated_at=entry["generated_at"],
        )
