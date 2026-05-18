from __future__ import annotations

import json
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import seaborn as sns
import torch
from loguru import logger

from app.utils.common import save_json
from app.training.components.epoch_metrics import EpochMetrics
from app.monitoring.prometheus_metrics import (
    ACTIVE_TRAINING_JOBS,
    BEST_VAL_ACCURACY,
    EPOCH_TRAIN_LOSS,
    GPU_MEMORY_USED_BYTES,
    GPU_UTILIZATION_PERCENT,
    TRAINING_BEST_F1,
    TRAINING_CURRENT_EPOCH,
    TRAINING_EPOCH_ACCURACY,
    TRAINING_EPOCH_DURATION,
    TRAINING_EPOCH_F1,
    TRAINING_EPOCH_QWK,
    TRAINING_LEARNING_RATE,
    TRAINING_PATIENCE_COUNTER,
    TRAINING_PER_CLASS_F1,
    TRAINING_PER_CLASS_RECALL,
    TRAINING_TOTAL_EPOCHS,
    TRAINING_VAL_LOSS,
    TRAINING_VAL_MAE,
    TRAINING_VAL_RMSE,
)

if TYPE_CHECKING:
    from app.training.components.model_trainer import ImagingModelTrainer


def _safe_mlflow(log_call, *args, **kwargs):
    try:
        return log_call(*args, **kwargs)
    except Exception as e:
        logger.warning(f"mlflow call failed (non-fatal, continuing): {e}")
        return None


class TrainingLogger:
    def __init__(self, trainer: ImagingModelTrainer):
        self._trainer = trainer
        self.epoch_log: list[dict] = []
        self._mlflow_run: mlflow.ActiveRun | None = None

    def __enter__(self) -> TrainingLogger:
        t = self._trainer
        run_name = (
            t.params.get("mlflow", {}).get("imaging_run_name", "efficientnet_b3")
            + f"_{int(time.time()) % 1000:03d}"
        )
        self._mlflow_run = mlflow.start_run(run_name=run_name)

        _safe_mlflow(
            mlflow.log_params,
            {
                "model": t.config.model_name,
                "pretrained": t.config.pretrained,
                "epochs": t.phase_epochs,
                "batch_size": t._training_batch_size,
                "lr": t.phase_lr,
                "weight_decay": t._training_weight_decay,
                "scheduler": t._training_scheduler,
                "num_classes": t._global_num_classes,
                "dropout": getattr(
                    t, "_phase_dropout", getattr(t, "_training_dropout", 0.5)
                ),
                "seed": t._global_seed,
                "loss": getattr(t, "_loss_type", "ordinal_cross_entropy"),
                "focal_gamma": getattr(t, "_focal_loss_gamma", None),
                "class_weights": (
                    t._class_weights_tensor.tolist()
                    if t._class_weights_tensor is not None
                    else None
                ),
                "freeze_backbone": t.freeze_backbone,
                "unfreeze_last_blocks": getattr(t, "unfreeze_last_blocks", False),
                "freeze_blocks": getattr(t, "_freeze_blocks", 3),
                "device": str(t.device),
                "phase": t.phase,
                "mixup_enabled": t._use_mixup,
                "mixup_alpha": t._mixup_alpha if t._use_mixup else 0.0,
                "fda_enabled": t._fda_augment is not None,
            },
        )
        TRAINING_TOTAL_EPOCHS.labels(pipeline="imaging").set(t.phase_epochs)
        ACTIVE_TRAINING_JOBS.inc()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ACTIVE_TRAINING_JOBS.dec()
        if self._mlflow_run is not None:
            try:
                mlflow.end_run()
            except Exception as e:
                logger.warning(f"mlflow end_run failed (non-fatal): {e}")
        return False

    def log_epoch(self, metrics: EpochMetrics, patience_counter: int) -> None:
        epoch = metrics.epoch

        TRAINING_CURRENT_EPOCH.labels(pipeline="imaging").set(epoch)
        TRAINING_EPOCH_ACCURACY.labels(pipeline="imaging", split="train").set(
            metrics.train_acc
        )
        TRAINING_EPOCH_ACCURACY.labels(pipeline="imaging", split="val").set(
            metrics.val_acc
        )
        TRAINING_EPOCH_F1.labels(pipeline="imaging", split="train").set(
            metrics.train_f1
        )
        TRAINING_EPOCH_F1.labels(pipeline="imaging", split="val").set(metrics.val_f1)
        TRAINING_LEARNING_RATE.labels(pipeline="imaging").set(metrics.lr)
        TRAINING_EPOCH_DURATION.labels(pipeline="imaging").set(metrics.duration_s)
        TRAINING_PATIENCE_COUNTER.labels(pipeline="imaging").set(patience_counter)
        TRAINING_VAL_LOSS.labels(pipeline="imaging").set(metrics.loss)
        TRAINING_VAL_MAE.labels(pipeline="imaging").set(metrics.mae)
        TRAINING_VAL_RMSE.labels(pipeline="imaging").set(metrics.rmse)
        TRAINING_EPOCH_QWK.labels(pipeline="imaging").set(metrics.qwk)
        EPOCH_TRAIN_LOSS.labels(pipeline="imaging").observe(metrics.loss)

        for cls_idx, cls_f1 in enumerate(metrics.class_f1):
            TRAINING_PER_CLASS_F1.labels(pipeline="imaging", dr_grade=str(cls_idx)).set(
                float(cls_f1)
            )
        for cls_idx, cls_recall in enumerate(metrics.class_recall):
            TRAINING_PER_CLASS_RECALL.labels(
                pipeline="imaging", dr_grade=str(cls_idx)
            ).set(float(cls_recall))

        if torch.cuda.is_available():
            GPU_MEMORY_USED_BYTES.labels(device="0").set(torch.cuda.memory_allocated(0))
            try:
                GPU_UTILIZATION_PERCENT.labels(device="0").set(
                    torch.cuda.utilization(0)
                )
            except Exception:
                pass

        _safe_mlflow(
            mlflow.log_metrics,
            {
                "train_loss": metrics.loss,
                "train_acc": metrics.train_acc,
                "val_acc": metrics.val_acc,
                "val_macro_f1": metrics.val_f1,
                "train_f1": metrics.train_f1,
                "lr": metrics.lr,
                "epoch_duration_s": metrics.duration_s,
                "val_mae": metrics.mae,
                "val_rmse": metrics.rmse,
                "val_qwk": metrics.qwk,
            },
            step=epoch - 1,
        )

        for cls_idx, cls_f1 in enumerate(metrics.class_f1):
            _safe_mlflow(
                mlflow.log_metric,
                f"val_f1_class_{cls_idx}",
                float(cls_f1),
                step=epoch - 1,
            )
        for cls_idx, cls_recall in enumerate(metrics.class_recall):
            _safe_mlflow(
                mlflow.log_metric,
                f"val_recall_class_{cls_idx}",
                float(cls_recall),
                step=epoch - 1,
            )

        self.epoch_log.append(
            {
                "epoch": epoch,
                "phase": self._trainer.phase,
                "loss": metrics.loss,
                "train_acc": metrics.train_acc,
                "val_acc": metrics.val_acc,
                "train_f1": metrics.train_f1,
                "val_f1": metrics.val_f1,
                "val_qwk": metrics.qwk,
                "val_mae": metrics.mae,
                "val_rmse": metrics.rmse,
                "train_mae": metrics.train_mae,
                "train_rmse": metrics.train_rmse,
                "lr": metrics.lr,
                "duration_s": metrics.duration_s,
                "class_recall": metrics.class_recall,
                "class_f1": metrics.class_f1,
                "class_precision": metrics.class_precision,
            }
        )

    def log_artifact(self, metrics: EpochMetrics) -> None:
        dr_labels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            metrics.confusion_matrix,
            annot=True,
            fmt="d",
            ax=ax,
            xticklabels=dr_labels,
            yticklabels=dr_labels,
            cmap="Blues",
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix \u2014 Epoch {metrics.epoch}")
        plt.tight_layout()
        cm_path = Path(f"/tmp/cm_epoch_{metrics.epoch}.png")
        fig.savefig(cm_path)
        _safe_mlflow(mlflow.log_artifact, str(cm_path), "confusion_matrices")
        plt.close(fig)

    def mark_checkpoint(self, qwk: float, f1: float) -> None:
        BEST_VAL_ACCURACY.labels(pipeline="imaging").set(qwk)
        TRAINING_BEST_F1.labels(pipeline="imaging").set(f1)

    def summarize(self, best_epoch_idx: int, best_val_acc: float) -> None:
        best_entry = self.epoch_log[best_epoch_idx] if best_epoch_idx >= 0 else {}
        best_val_qwk = best_entry.get("val_qwk", 0.0)
        macro_f1 = best_entry.get("val_f1", 0.0)

        _safe_mlflow(mlflow.log_metric, "best_val_acc", float(best_val_acc))
        _safe_mlflow(mlflow.log_metric, "best_val_qwk", float(best_val_qwk))
        _safe_mlflow(mlflow.log_metric, "best_val_macro_f1", float(macro_f1))

        checkpoint_dir = self._trainer.config.checkpoint_path.parent
        summary = {
            "phase": self._trainer.phase,
            "total_epochs": self._trainer.phase_epochs,
            "best_epoch": best_epoch_idx + 1,
            "best_val_acc": float(best_val_acc),
            "best_val_qwk": float(best_val_qwk),
            "best_val_f1": float(best_entry.get("val_f1", 0.0)),
            "best_val_mae": float(best_entry.get("val_mae", 0.0)),
            "best_val_rmse": float(best_entry.get("val_rmse", 0.0)),
            "epoch_log": self.epoch_log,
        }
        save_json(checkpoint_dir / "training_summary.json", summary)

        history_path = checkpoint_dir / "training_history.jsonl"
        mlflow_run_id = "no-run"
        try:
            active = mlflow.active_run()
            if active:
                mlflow_run_id = active.info.run_id
        except Exception:
            pass

        history_entry = {
            "run_id": mlflow_run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phase": self._trainer.phase,
            "best_epoch": best_epoch_idx + 1,
            "total_epochs_trained": len(self.epoch_log),
            "best_val_qwk": float(best_val_qwk),
            "best_val_acc": float(best_val_acc),
            "best_val_f1": float(best_entry.get("val_f1", 0.0)),
            "best_val_mae": float(best_entry.get("val_mae", 0.0)),
            "best_val_rmse": float(best_entry.get("val_rmse", 0.0)),
        }
        with open(history_path, "a") as f:
            f.write(json.dumps(history_entry) + "\n")
        logger.info(f"training history appended: {history_path}")

    def log_model(
        self,
        checkpoint_path: Path,
        model_name: str,
        num_classes: int,
        dropout_rate: float,
    ) -> None:
        if not checkpoint_path.exists():
            logger.warning("best checkpoint missing; skipping mlflow model log")
            return
        logger.info(f"logging best model to mlflow: {checkpoint_path}")

        timeout_seconds = int(
            self._trainer.params.get("mlflow", {}).get("model_log_timeout_seconds", 600)
        )

        class _LogTimeout(Exception):
            pass

        def _timeout_handler(signum, frame):
            raise _LogTimeout("model logging timed out")

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_seconds)

        try:
            _safe_mlflow(mlflow.log_param, "checkpoint_path", str(checkpoint_path))
            import timm

            model = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=num_classes,
                drop_rate=dropout_rate,
            )
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state_dict)
            model.eval()
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="imaging_model",
                registered_model_name="efficientnet_b3",
            )
            logger.info("model logged to mlflow model registry")
        except _LogTimeout:
            logger.warning(f"mlflow model logging timed out after {timeout_seconds}s")
        except Exception as e:
            logger.warning(f"failed to log model to mlflow registry: {e}")
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
