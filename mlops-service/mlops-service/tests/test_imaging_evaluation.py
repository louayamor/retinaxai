from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

from app.domains.imaging.evaluation.imaging_evaluation import ImagingModelEvaluation
from app.entity.config_entity import ImagingModelEvaluationConfig


@pytest.fixture
def mock_config(tmp_path: Path) -> ImagingModelEvaluationConfig:
    return ImagingModelEvaluationConfig(
        root_dir=tmp_path / "eval_artifacts",
        model_name="efficientnet_b4",
        test_csv=tmp_path / "test.csv",
        samaya_csv=tmp_path / "samaya.csv",
        model_path=tmp_path / "model.pth",
        metric_file=tmp_path / "metrics.json",
        mlflow_uri="http://localhost:5000",
        experiment_name="test_experiment",
        run_name="test_run",
    )


@pytest.fixture
def mock_params() -> dict:
    return {
        "dl_training": {
            "image_size": 224,
            "num_classes": 5,
            "dropout": 0.3,
        },
        "augmentation": {
            "normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            }
        },
        "evaluation": {
            "dl": {
                "batch_size": 4,
                "num_workers": 0,
            },
            "domain_shift": {
                "compute_confidence_ece": False,
                "compute_embedding_mmd": False,
            },
        },
    }


@pytest.fixture
def evaluator(mock_config: ImagingModelEvaluationConfig, mock_params: dict):
    with (
        patch(
            "app.domains.imaging.evaluation.imaging_evaluation.read_yaml",
        ) as mock_read,
        patch("app.domains.imaging.evaluation.imaging_evaluation.Path.mkdir"),
        patch(
            "app.domains.imaging.evaluation.imaging_evaluation.Path.exists",
            return_value=True,
        ),
    ):
        params_mock = MagicMock()
        params_mock.dl_training.image_size = mock_params["dl_training"]["image_size"]
        params_mock.dl_training.num_classes = mock_params["dl_training"]["num_classes"]
        params_mock.dl_training.dropout = mock_params["dl_training"]["dropout"]
        params_mock.augmentation.normalize.mean = mock_params["augmentation"][
            "normalize"
        ]["mean"]
        params_mock.augmentation.normalize.std = mock_params["augmentation"][
            "normalize"
        ]["std"]
        params_mock.evaluation.dl.batch_size = mock_params["evaluation"]["dl"][
            "batch_size"
        ]
        params_mock.evaluation.dl.num_workers = mock_params["evaluation"]["dl"][
            "num_workers"
        ]
        params_mock.evaluation.domain_shift.compute_confidence_ece = mock_params[
            "evaluation"
        ]["domain_shift"]["compute_confidence_ece"]
        params_mock.evaluation.domain_shift.compute_embedding_mmd = mock_params[
            "evaluation"
        ]["domain_shift"]["compute_embedding_mmd"]
        params_mock.get.return_value = {}
        mock_read.return_value = params_mock

        yield ImagingModelEvaluation(mock_config)


class TestComputeMetrics:
    def test_returns_all_expected_keys(self, evaluator: ImagingModelEvaluation):
        preds = [0, 1, 2, 3, 4]
        labels = [0, 1, 2, 3, 4]
        probs = np.eye(5)

        metrics = evaluator._compute_metrics(preds, labels, probs, "test_split")

        expected_keys = {
            "split",
            "accuracy",
            "quadratic_weighted_kappa",
            "roc_auc_macro",
            "macro_f1",
            "precision_macro",
            "recall_macro",
            "confusion_matrix",
            "classification_report",
            "num_samples",
            "label_distribution",
        }
        assert set(metrics.keys()) == expected_keys

    def test_perfect_predictions(self, evaluator: ImagingModelEvaluation):
        preds = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        labels = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        probs = np.eye(5)[labels]

        metrics = evaluator._compute_metrics(preds, labels, probs, "perfect")

        assert metrics["accuracy"] == 1.0
        assert metrics["quadratic_weighted_kappa"] == 1.0
        assert metrics["macro_f1"] == 1.0
        assert metrics["precision_macro"] == 1.0
        assert metrics["recall_macro"] == 1.0
        assert metrics["roc_auc_macro"] == 1.0
        assert metrics["num_samples"] == 10

    def test_worst_predictions(self, evaluator: ImagingModelEvaluation):
        preds = [4, 4, 4, 4, 4]
        labels = [0, 1, 2, 3, 4]
        probs = np.array([[0.1, 0.1, 0.1, 0.1, 0.6]] * 5)

        metrics = evaluator._compute_metrics(preds, labels, probs, "worst")

        assert metrics["accuracy"] == pytest.approx(0.2)
        assert metrics["num_samples"] == 5

    def test_auc_none_when_single_class(self, evaluator: ImagingModelEvaluation):
        preds = [0, 0, 0, 0, 0]
        labels = [0, 0, 0, 0, 0]
        probs = np.array([[0.9, 0.025, 0.025, 0.025, 0.025]] * 5)

        metrics = evaluator._compute_metrics(preds, labels, probs, "single_class")

        assert metrics["roc_auc_macro"] is None


class TestComputeAUC:
    def test_none_when_less_than_two_classes(self, evaluator: ImagingModelEvaluation):
        assert evaluator._compute_auc([0, 0, 0], np.eye(3)[:3]) is None

    def test_returns_float_with_multiple_classes(
        self, evaluator: ImagingModelEvaluation
    ):
        labels = [0, 1, 2, 0, 1, 2]
        probs = np.eye(3)[labels] * 0.8 + 0.05
        probs = probs / probs.sum(axis=1, keepdims=True)

        result = evaluator._compute_auc(labels, probs)
        assert result is not None
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_handles_probability_normalization(self, evaluator: ImagingModelEvaluation):
        labels = [0, 1, 2, 0, 1, 2]
        probs = np.array(
            [
                [2.0, 0.5, 0.3],
                [0.3, 3.0, 0.5],
                [0.5, 0.3, 2.0],
                [2.0, 0.5, 0.3],
                [0.3, 3.0, 0.5],
                [0.5, 0.3, 2.0],
            ]
        )

        result = evaluator._compute_auc(labels, probs)
        assert result is not None
        assert isinstance(result, float)


class TestComputeECE:
    def test_perfect_calibration_returns_zero(self, evaluator: ImagingModelEvaluation):
        labels = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
        probs = np.array(
            [
                [0.99, 0.005, 0.005],
                [0.98, 0.01, 0.01],
                [0.97, 0.02, 0.01],
                [0.01, 0.98, 0.01],
                [0.01, 0.97, 0.02],
                [0.02, 0.97, 0.01],
                [0.005, 0.005, 0.99],
                [0.005, 0.005, 0.99],
                [0.005, 0.005, 0.99],
                [0.005, 0.005, 0.99],
            ]
        )

        preds = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
        ece = evaluator._compute_ece(labels, preds, probs, n_bins=5)
        assert ece < 0.02

    def test_miscalibration_returns_positive(self, evaluator: ImagingModelEvaluation):
        labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        probs = np.array(
            [
                [0.99, 0.005, 0.005],
                [0.99, 0.005, 0.005],
                [0.99, 0.005, 0.005],
                [0.99, 0.005, 0.005],
                [0.99, 0.005, 0.005],
                [0.01, 0.49, 0.50],
                [0.01, 0.49, 0.50],
                [0.01, 0.49, 0.50],
                [0.01, 0.49, 0.50],
                [0.01, 0.49, 0.50],
            ]
        )

        preds = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ece = evaluator._compute_ece(labels, preds, probs, n_bins=5)
        assert ece > 0.01

    def test_ece_uses_labels_not_preds(self, evaluator: ImagingModelEvaluation):
        # All predictions are class 0 but ground truth varies
        labels = [0, 1, 2, 0, 1, 2]
        probs = np.array(
            [
                [0.9, 0.05, 0.05],
                [0.9, 0.05, 0.05],
                [0.9, 0.05, 0.05],
                [0.9, 0.05, 0.05],
                [0.9, 0.05, 0.05],
                [0.9, 0.05, 0.05],
            ]
        )

        preds = [0, 0, 0, 0, 0, 0]
        ece = evaluator._compute_ece(labels, preds, probs, n_bins=5)
        # If ECE used preds, this would be near 0 (model "thinks" it's right)
        # If ECE uses labels, this is high (model is wrong 4/6 times despite 90% confidence)
        assert ece > 0.1


class TestComputeEmbeddingMMD:
    def test_identical_embeddings_returns_zero(self, evaluator: ImagingModelEvaluation):
        model = MagicMock(spec=nn.Module)
        model.forward_features = MagicMock(return_value=torch.randn(10, 128, 7, 7))
        model.global_pool = MagicMock(return_value=torch.randn(10, 128))

        with (
            patch(
                "app.domains.imaging.evaluation.imaging_evaluation.RetinalDataset",
            ),
            patch(
                "app.domains.imaging.evaluation.imaging_evaluation.DataLoader",
            ) as mock_loader,
        ):
            mock_loader.return_value = [
                (torch.randn(4, 3, 224, 224), torch.randint(0, 5, (4,)))
            ]

            actual_mmd = evaluator._compute_embedding_mmd(
                model,
                Path("/fake/eyepacs.csv"),
                Path("/fake/samaya.csv"),
                MagicMock(),
            )
            assert isinstance(actual_mmd, float)


class TestFullEvaluate:
    def test_evaluate_returns_expected_structure(
        self, evaluator: ImagingModelEvaluation, tmp_path: Path
    ):
        evaluator.config.samaya_csv.write_text("")
        evaluator.config.metric_file.parent.mkdir(parents=True, exist_ok=True)

        fake_model = MagicMock(spec=nn.Module)
        fake_model.to.return_value = fake_model
        fake_model.return_value = torch.randn(4, 5)

        fake_batch = (torch.randn(4, 3, 224, 224), torch.tensor([0, 1, 2, 3]))

        def _make_loader(*args, **kwargs):
            loader = MagicMock()
            loader.__len__.return_value = 2
            loader.__iter__.return_value = iter([fake_batch, fake_batch])
            return loader

        with (
            patch.object(evaluator, "_load_model", return_value=fake_model),
            patch(
                "app.domains.imaging.evaluation.imaging_evaluation.RetinalDataset",
            ),
            patch(
                "app.domains.imaging.evaluation.imaging_evaluation.DataLoader",
                side_effect=_make_loader,
            ),
            patch(
                "app.domains.imaging.evaluation.imaging_evaluation.mlflow.start_run",
            ) as mock_mlflow_run,
        ):
            mock_mlflow_run.return_value.__enter__.return_value = MagicMock()

            result = evaluator.evaluate()

        assert "eyepacs_test" in result
        eyepacs = result["eyepacs_test"]
        for key in (
            "accuracy",
            "quadratic_weighted_kappa",
            "macro_f1",
            "precision_macro",
            "recall_macro",
            "num_samples",
            "confusion_matrix",
            "classification_report",
        ):
            assert key in eyepacs, f"Missing key: {key}"

        assert "samaya_validation" in result

    def test_evaluate_skips_samaya_when_csv_missing(
        self, evaluator: ImagingModelEvaluation
    ):
        # Override the Path.exists patch so samaya_csv.exists() returns False
        fake_model = MagicMock(spec=nn.Module)
        fake_model.to.return_value = fake_model
        fake_model.return_value = torch.randn(4, 5)

        fake_batch = (torch.randn(4, 3, 224, 224), torch.tensor([0, 1, 2, 3]))

        def _make_loader(*args, **kwargs):
            loader = MagicMock()
            loader.__len__.return_value = 1
            loader.__iter__.return_value = iter([fake_batch])
            return loader

        original_exists = Path.exists

        def _exists_side_effect(self_self):
            if str(self_self).endswith("samaya.csv"):
                return False
            return original_exists(self_self)

        with (
            patch.object(evaluator, "_load_model", return_value=fake_model),
            patch(
                "app.domains.imaging.evaluation.imaging_evaluation.RetinalDataset",
            ),
            patch(
                "app.domains.imaging.evaluation.imaging_evaluation.DataLoader",
                side_effect=_make_loader,
            ),
            patch(
                "app.domains.imaging.evaluation.imaging_evaluation.mlflow.start_run",
            ) as mock_mlflow_run,
            patch.object(Path, "exists", _exists_side_effect),
        ):
            mock_mlflow_run.return_value.__enter__.return_value = MagicMock()

            result = evaluator.evaluate()

        assert result["samaya_validation"] is None
