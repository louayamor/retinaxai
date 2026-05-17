from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    phase: str
    loss: float
    train_acc: float
    val_acc: float
    train_f1: float
    val_f1: float
    qwk: float
    mae: float
    rmse: float
    train_mae: float
    train_rmse: float
    lr: float
    duration_s: float
    class_recall: list[float] = field(default_factory=list)
    class_f1: list[float] = field(default_factory=list)
    class_precision: list[float] = field(default_factory=list)
    confusion_matrix: list[list[int]] = field(default_factory=list)

    @staticmethod
    def from_predictions(
        epoch: int,
        phase: str,
        train_preds: list[int],
        train_labels: list[int],
        val_preds: list[int],
        val_labels: list[int],
        avg_loss: float,
        lr: float,
        duration_s: float,
        num_classes: int = 5,
    ) -> EpochMetrics:
        train_acc = float(
            sum(1 for p, t in zip(train_preds, train_labels) if p == t)
            / max(len(train_labels), 1)
        )
        train_f1 = float(
            f1_score(train_labels, train_preds, average="macro", zero_division="warn")
        )
        train_mae = float(
            np.mean(np.abs(np.array(train_preds) - np.array(train_labels)))
        )
        train_rmse = float(
            np.sqrt(
                np.mean(
                    (
                        np.array(train_preds).astype(float)
                        - np.array(train_labels).astype(float)
                    )
                    ** 2
                )
            )
        )

        val_acc = float(
            sum(1 for p, t in zip(val_preds, val_labels) if p == t)
            / max(len(val_labels), 1)
        )
        val_f1 = float(
            f1_score(val_labels, val_preds, average="macro", zero_division="warn")
        )
        per_class_f1 = f1_score(val_labels, val_preds, average=None, zero_division=0)
        per_class_recall = recall_score(
            val_labels, val_preds, average=None, zero_division=0
        )
        per_class_precision = precision_score(
            val_labels, val_preds, average=None, zero_division=0
        )
        cm = confusion_matrix(val_labels, val_preds)
        mae = float(np.mean(np.abs(np.array(val_preds) - np.array(val_labels))))
        rmse = float(
            np.sqrt(
                np.mean(
                    (
                        np.array(val_preds).astype(float)
                        - np.array(val_labels).astype(float)
                    )
                    ** 2
                )
            )
        )
        qwk = float(cohen_kappa_score(val_labels, val_preds, weights="quadratic"))

        return EpochMetrics(
            epoch=epoch,
            phase=phase,
            loss=float(avg_loss),
            train_acc=train_acc,
            val_acc=val_acc,
            train_f1=train_f1,
            val_f1=val_f1,
            qwk=qwk,
            mae=mae,
            rmse=rmse,
            train_mae=train_mae,
            train_rmse=train_rmse,
            lr=float(lr),
            duration_s=duration_s,
            class_recall=[float(r) for r in per_class_recall[:num_classes]],
            class_f1=[float(f) for f in per_class_f1[:num_classes]],
            class_precision=[float(p) for p in per_class_precision[:num_classes]],
            confusion_matrix=cm.tolist(),
        )
