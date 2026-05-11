import json
import pandas as pd
from pathlib import Path
from loguru import logger

from evidently import Report
from evidently.presets import DataDriftPreset

from app.services.monitoring.prometheus_metrics import (
    EVIDENTLY_DRIFT_DATASET_SHIFT,
    EVIDENTLY_DRIFT_FEATURES_DRIFTED,
)


class EvidentlyReportGenerator:
    def __init__(self, reports_dir: Path):
        self.reports_dir = reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _drop_non_numeric_cols(
        self, df: pd.DataFrame, keep: list | None = None
    ) -> pd.DataFrame:
        keep = keep or []
        exclude = {"image_path", "source", "level"}
        cols = [
            c
            for c in df.columns
            if c not in exclude
            and (c in keep or df[c].dtype in ["float64", "int64", "float32", "int32"])
        ]
        if not cols and "label" in df.columns:
            cols = ["label"]
        return df[cols] if cols else df

    def _extract_drift_values(self, snapshot) -> dict:
        """Parse Evidently snapshot.dict() metrics into flat dict."""
        result = snapshot.dict()
        metrics = {}
        for m in result.get("metrics", []):
            name = m.get("metric", m.get("name", ""))
            value = m.get("value", m.get("result", {}))
            metrics[name] = value
        return metrics

    def _safe_float(self, metrics: dict, key: str, default: float = 0.0) -> float:
        """Safely extract a float from metrics dict."""
        val = metrics.get(key, default)
        if isinstance(val, dict):
            val = val.get("value", val.get("drift_share", default))
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    def _safe_int(self, metrics: dict, key: str, default: int = 0) -> int:
        """Safely extract an int from metrics dict."""
        val = metrics.get(key, default)
        if isinstance(val, dict):
            val = val.get("count", val.get("value", default))
        try:
            return int(val)
        except (TypeError, ValueError):
            return default

    def imaging_data_drift(
        self,
        reference_csv: Path,
        current_csv: Path,
        output_path: Path,
    ) -> dict:
        logger.info("generating imaging data drift report")
        reference = self._drop_non_numeric_cols(pd.read_csv(reference_csv))
        current = self._drop_non_numeric_cols(pd.read_csv(current_csv))

        report = Report([DataDriftPreset()])
        snapshot = report.run(reference, current)
        snapshot.save_html(str(output_path))
        logger.info(f"imaging data drift report saved: {output_path}")
        return self._extract_drift_values(snapshot)

    def clinical_data_drift(
        self,
        reference_csv: Path,
        current_csv: Path,
        output_path: Path,
    ) -> dict:
        logger.info("generating clinical data drift report")
        reference = pd.read_csv(reference_csv)
        current = pd.read_csv(current_csv)

        report = Report([DataDriftPreset()])
        snapshot = report.run(reference, current)
        snapshot.save_html(str(output_path))
        logger.info(f"clinical data drift report saved: {output_path}")
        return self._extract_drift_values(snapshot)

    def imaging_classification_performance(
        self,
        reference_csv: Path,
        current_csv: Path,
        output_path: Path,
    ) -> None:
        logger.info("generating imaging classification performance report")
        reference = pd.read_csv(reference_csv)
        current = pd.read_csv(current_csv)

        report = Report([DataDriftPreset()])
        result = report.run(reference, current)
        result.save_html(str(output_path))
        logger.info(f"imaging classification report saved: {output_path}")

    def clinical_classification_performance(
        self,
        reference_csv: Path,
        current_csv: Path,
        output_path: Path,
    ) -> None:
        logger.info("generating clinical classification performance report")
        reference = pd.read_csv(reference_csv)
        current = pd.read_csv(current_csv)

        report = Report([DataDriftPreset()])
        result = report.run(reference, current)
        result.save_html(str(output_path))
        logger.info(f"clinical classification report saved: {output_path}")

    def domain_shift_report(
        self,
        eyepacs_csv: Path,
        samaya_csv: Path,
        output_path: Path,
    ) -> dict:
        logger.info("generating domain shift report: EyePACS vs Samaya")
        eyepacs = self._drop_non_numeric_cols(pd.read_csv(eyepacs_csv))
        samaya = self._drop_non_numeric_cols(pd.read_csv(samaya_csv))

        common_cols = list(set(eyepacs.columns) & set(samaya.columns))
        eyepacs = eyepacs[common_cols]
        samaya = samaya[common_cols]

        report = Report([DataDriftPreset()])
        snapshot = report.run(eyepacs, samaya)
        snapshot.save_html(str(output_path))
        logger.info(f"domain shift report saved: {output_path}")
        return self._extract_drift_values(snapshot)

    def _save_evidently_metrics(
        self, pipeline: str, dataset_drift: float, features_drifted: int
    ) -> None:
        """Persist Evidently drift metrics to JSON so the FastAPI process can serve them."""
        metrics_file = self.reports_dir / "evidently_metrics.json"
        try:
            existing = {}
            if metrics_file.exists():
                with open(metrics_file) as f:
                    existing = json.load(f)
            existing[pipeline] = {
                "dataset_drift": dataset_drift,
                "features_drifted": features_drifted,
            }
            with open(metrics_file, "w") as f:
                json.dump(existing, f, indent=2)
        except Exception as e:
            logger.warning(f"failed to persist evidently metrics: {e}")

    def run_drift_and_emit(
        self, pipeline: str, reference_csv: Path, current_csv: Path
    ) -> dict:
        """Run Evidently drift report and emit metrics to Prometheus.

        Gracefully handles cases where input DataFrames have no analyzable columns
        (e.g., imaging pipeline with only image_path and label columns).
        """
        try:
            reference_df = pd.read_csv(reference_csv)
            current_df = pd.read_csv(current_csv)

            ref_clean = self._drop_non_numeric_cols(reference_df)
            curr_clean = self._drop_non_numeric_cols(current_df)

            if ref_clean.empty or curr_clean.empty or len(ref_clean.columns) == 0:
                logger.info(
                    f"No numeric columns for Evidently drift check on {pipeline}; "
                    f"ref_cols={list(reference_df.columns)}, kept={list(ref_clean.columns)}"
                )
                self._save_evidently_metrics(pipeline, 0.0, 0)
                return {}

            output_path = self.reports_dir / f"{pipeline}_evidently_drift_report.html"

            if pipeline == "imaging":
                metrics = self.imaging_data_drift(
                    reference_csv, current_csv, output_path
                )
            else:
                metrics = self.clinical_data_drift(
                    reference_csv, current_csv, output_path
                )

            dataset_drift = self._safe_float(metrics, "DatasetDriftMetric")
            drifted_count = self._safe_int(metrics, "DriftedColumnsCount")

            EVIDENTLY_DRIFT_DATASET_SHIFT.labels(pipeline=pipeline).set(dataset_drift)
            EVIDENTLY_DRIFT_FEATURES_DRIFTED.labels(pipeline=pipeline).set(
                drifted_count
            )
            self._save_evidently_metrics(pipeline, dataset_drift, drifted_count)

            logger.info(
                f"Evidently drift check complete: {pipeline} - "
                f"dataset_drift={dataset_drift:.4f}, features_drifted={drifted_count}"
            )
            return metrics
        except Exception as e:
            logger.warning(
                f"Evidently drift check skipped for {pipeline} (non-fatal): {e}"
            )
            self._save_evidently_metrics(pipeline, 0.0, 0)
            return {}

    def run_all(
        self,
        imaging_train_csv: Path,
        imaging_test_csv: Path,
        imaging_samaya_csv: Path,
        clinical_train_csv: Path,
        clinical_test_csv: Path,
    ) -> None:
        logger.info("=" * 60)
        logger.info("running all evidently reports")
        logger.info("=" * 60)

        self.imaging_data_drift(
            reference_csv=imaging_train_csv,
            current_csv=imaging_test_csv,
            output_path=self.reports_dir / "imaging_drift_report.html",
        )

        self.clinical_data_drift(
            reference_csv=clinical_train_csv,
            current_csv=clinical_test_csv,
            output_path=self.reports_dir / "clinical_drift_report.html",
        )

        if imaging_samaya_csv.exists():
            self.domain_shift_report(
                eyepacs_csv=imaging_test_csv,
                samaya_csv=imaging_samaya_csv,
                output_path=self.reports_dir / "domain_shift_report.html",
            )

        logger.info("all evidently reports generated")
        logger.info("=" * 60)
