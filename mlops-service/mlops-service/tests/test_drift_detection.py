from __future__ import annotations

import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.services.monitoring.drift_detection import (
    DriftReport,
    FeatureDriftResult,
    DriftDetectionService,
    DriftStatus,
)
from app.services.monitoring.evidently_report import EvidentlyReportGenerator


class TestDriftReport:
    def test_num_features_drifted_property(self):
        feature_results = [
            FeatureDriftResult("feat1", 0.05, False, 0.0, 0.1),
            FeatureDriftResult("feat2", 0.15, True, 0.0, 0.2),
            FeatureDriftResult("feat3", 0.25, True, 0.0, 0.3),
            FeatureDriftResult("feat4", 0.02, False, 0.0, 0.05),
        ]
        report = DriftReport(
            pipeline="imaging",
            status=DriftStatus.DRIFT_DETECTED,
            overall_psi=0.2,
            drift_detected=True,
            feature_results=feature_results,
            reference_samples=100,
            current_samples=100,
            generated_at="2024-01-01T00:00:00",
        )
        assert report.num_features_drifted == 2

    def test_num_features_drifted_zero(self):
        feature_results = [
            FeatureDriftResult("feat1", 0.05, False, 0.0, 0.1),
            FeatureDriftResult("feat2", 0.02, False, 0.0, 0.05),
        ]
        report = DriftReport(
            pipeline="clinical",
            status=DriftStatus.NO_DRIFT,
            overall_psi=0.0,
            drift_detected=False,
            feature_results=feature_results,
            reference_samples=100,
            current_samples=100,
            generated_at="2024-01-01T00:00:00",
        )
        assert report.num_features_drifted == 0

    def test_num_features_drifted_all(self):
        feature_results = [
            FeatureDriftResult("feat1", 0.35, True, 0.0, 0.5),
            FeatureDriftResult("feat2", 0.45, True, 0.0, 0.6),
        ]
        report = DriftReport(
            pipeline="imaging",
            status=DriftStatus.DRIFT_DETECTED,
            overall_psi=0.4,
            drift_detected=True,
            feature_results=feature_results,
            reference_samples=100,
            current_samples=100,
            generated_at="2024-01-01T00:00:00",
        )
        assert report.num_features_drifted == 2


class TestEvidentlyReportGenerator:
    @pytest.fixture
    def temp_dir(self, tmp_path: Path) -> Path:
        return tmp_path / "evidently_reports"

    @pytest.fixture
    def sample_reference_csv(self, tmp_path: Path) -> Path:
        csv_path = tmp_path / "reference.csv"
        df = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0] * 20,
                "feature2": [10.0, 20.0, 30.0, 40.0, 50.0] * 20,
                "feature3": [0.1, 0.2, 0.3, 0.4, 0.5] * 20,
            }
        )
        df.to_csv(csv_path, index=False)
        return csv_path

    @pytest.fixture
    def sample_current_csv(self, tmp_path: Path) -> Path:
        csv_path = tmp_path / "current.csv"
        df = pd.DataFrame(
            {
                "feature1": [1.5, 2.5, 3.5, 4.5, 5.5] * 20,
                "feature2": [12.0, 22.0, 32.0, 42.0, 52.0] * 20,
                "feature3": [0.15, 0.25, 0.35, 0.45, 0.55] * 20,
            }
        )
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_init_creates_reports_dir(self, temp_dir: Path):
        generator = EvidentlyReportGenerator(temp_dir)
        assert temp_dir.exists()
        assert generator.reports_dir == temp_dir

    def test_imaging_data_drift_returns_metrics(
        self, temp_dir: Path, sample_reference_csv: Path, sample_current_csv: Path
    ):
        generator = EvidentlyReportGenerator(temp_dir)
        output_path = temp_dir / "test_drift.html"

        metrics = generator.imaging_data_drift(
            sample_reference_csv, sample_current_csv, output_path
        )

        assert output_path.exists()
        assert isinstance(metrics, dict)

    def test_extract_drift_values_handles_empty_metrics(self, temp_dir: Path):
        generator = EvidentlyReportGenerator(temp_dir)
        mock_snapshot = MagicMock()
        mock_snapshot.dict.return_value = {"metrics": []}

        result = generator._extract_drift_values(mock_snapshot)
        assert result == {}

    def test_safe_float_with_dict_value(self, temp_dir: Path):
        generator = EvidentlyReportGenerator(temp_dir)
        metrics = {"DatasetDriftMetric": {"value": 0.35, "drift_share": 0.4}}

        result = generator._safe_float(metrics, "DatasetDriftMetric")
        assert isinstance(result, float)

    def test_safe_int_with_dict_value(self, temp_dir: Path):
        generator = EvidentlyReportGenerator(temp_dir)
        metrics = {"DriftedColumnsCount": {"count": 3, "value": 5}}

        result = generator._safe_int(metrics, "DriftedColumnsCount")
        assert isinstance(result, int)

    def test_safe_float_with_invalid_value(self, temp_dir: Path):
        generator = EvidentlyReportGenerator(temp_dir)
        metrics = {"DatasetDriftMetric": "invalid"}

        result = generator._safe_float(metrics, "DatasetDriftMetric", default=0.0)
        assert result == 0.0

    def test_run_drift_and_emit_emits_metrics(
        self, temp_dir: Path, sample_reference_csv: Path, sample_current_csv: Path
    ):
        generator = EvidentlyReportGenerator(temp_dir)

        with (
            patch(
                "app.services.monitoring.evidently_report.EVIDENTLY_DRIFT_DATASET_SHIFT"
            ) as mock_shift,
            patch(
                "app.services.monitoring.evidently_report.EVIDENTLY_DRIFT_FEATURES_DRIFTED"
            ) as mock_features,
        ):
            metrics = generator.run_drift_and_emit(
                pipeline="imaging",
                reference_csv=sample_reference_csv,
                current_csv=sample_current_csv,
            )

            assert mock_shift.labels.called
            assert mock_features.labels.called
            assert isinstance(metrics, dict)


class TestDriftDetectionService:
    @pytest.fixture
    def temp_dir(self, tmp_path: Path) -> Path:
        return tmp_path / "drift_artifacts"

    @pytest.fixture
    def sample_reference_csv(self, tmp_path: Path) -> Path:
        csv_path = tmp_path / "reference.csv"
        df = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0] * 20,
                "feature2": [10.0, 20.0, 30.0, 40.0, 50.0] * 20,
            }
        )
        df.to_csv(csv_path, index=False)
        return csv_path

    @pytest.fixture
    def sample_current_csv(self, tmp_path: Path) -> Path:
        csv_path = tmp_path / "current.csv"
        df = pd.DataFrame(
            {
                "feature1": [1.5, 2.5, 3.5, 4.5, 5.5] * 20,
                "feature2": [12.0, 22.0, 32.0, 42.0, 52.0] * 20,
            }
        )
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_check_drift_returns_report(
        self, temp_dir: Path, sample_reference_csv: Path, sample_current_csv: Path
    ):
        service = DriftDetectionService(temp_dir, temp_dir / "reports")

        report = service.check_drift(
            reference_csv=sample_reference_csv,
            current_csv=sample_current_csv,
            pipeline="imaging",
        )

        assert isinstance(report, DriftReport)
        assert report.pipeline == "imaging"
        assert hasattr(report, "num_features_drifted")

    def test_check_drift_detects_no_drift_when_similar(
        self, temp_dir: Path, sample_reference_csv: Path
    ):
        service = DriftDetectionService(temp_dir, temp_dir / "reports")

        report = service.check_drift(
            reference_csv=sample_reference_csv,
            current_csv=sample_reference_csv,
            pipeline="imaging",
        )

        assert report.drift_detected is False
        assert report.overall_psi == 0.0

    def test_check_drift_saves_history(
        self, temp_dir: Path, sample_reference_csv: Path, sample_current_csv: Path
    ):
        service = DriftDetectionService(temp_dir, temp_dir / "reports")

        service.check_drift(
            reference_csv=sample_reference_csv,
            current_csv=sample_current_csv,
            pipeline="imaging",
        )

        history_file = temp_dir / "reports" / "drift_history.json"
        assert history_file.exists()
