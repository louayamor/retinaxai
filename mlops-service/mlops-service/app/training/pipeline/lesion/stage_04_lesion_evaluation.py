from __future__ import annotations

import sys
from pathlib import Path

import dagshub
import mlflow
import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader

from app.constants import PARAMS_FILE_PATH
from app.training.components.ddr_lesion_dataset import DDRLesionDataset
from app.training.components.lesion_model import LesionSegmentationModel
from app.utils.common import read_yaml

logger.remove()
logger.add(sys.stdout, serialize=True)

ARTIFACTS_DIR = Path("artifacts")
MANIFESTS_DIR = ARTIFACTS_DIR / "lesion" / "manifests"
LESION_CHECKPOINT = ARTIFACTS_DIR / "model" / "lesion" / "model.pth"

CLASS_NAMES = ("ma", "he", "ex", "se")


def compute_dice_per_class(
    probs: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for i, cls_name in enumerate(CLASS_NAMES):
        p = probs[:, i]
        t = targets[:, i]
        intersection = (p * t).sum().item()
        total = p.sum().item() + t.sum().item()
        dice = (2.0 * intersection + smooth) / (total + smooth)
        scores[cls_name] = round(dice, 4)
    return scores


def run() -> dict:
    logger.info(">>> stage 04: lesion model evaluation started")

    params = read_yaml(PARAMS_FILE_PATH)
    lesion_cfg = params.get("lesion_model", {})
    image_size = int(lesion_cfg.get("image_size", 384))
    batch_size = int(lesion_cfg.get("batch_size", 8))
    num_workers = int(lesion_cfg.get("num_workers", 4))
    encoder_name = str(lesion_cfg.get("encoder_name", "timm-efficientnet-b3"))
    checkpoint_path = Path(
        str(lesion_cfg.get("checkpoint_path", LESION_CHECKPOINT))
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Lesion checkpoint not found: {checkpoint_path}")

    from app.utils.common import require_cuda

    device = require_cuda()

    val_csv = MANIFESTS_DIR / "val.csv"
    if not val_csv.exists():
        raise FileNotFoundError(f"Val manifest not found: {val_csv}")

    val_dataset = DDRLesionDataset(val_csv, image_size=image_size)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = LesionSegmentationModel(encoder_name=encoder_name)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device)
    )
    model.to(device)
    model.eval()

    all_probs: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    with torch.inference_mode():
        for images, masks in val_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu()
            all_probs.append(probs)
            all_targets.append(masks)

    all_probs = torch.cat(all_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    per_class_dice = compute_dice_per_class(all_probs, all_targets)
    mean_dice = round(
        sum(per_class_dice.values()) / len(per_class_dice), 4
    )

    logger.info(f"Per-class Dice: {per_class_dice}")
    logger.info(f"Mean Dice: {mean_dice}")

    dagshub.init(repo_owner="louayamor", repo_name="retinaxai", mlflow=True)

    with mlflow.start_run(run_name="stage_04_lesion_evaluation"):
        mlflow.log_metrics({
            **{f"dice_{k}": v for k, v in per_class_dice.items()},
            "dice_mean": mean_dice,
        })
        mlflow.log_param("val_samples", len(val_dataset))
        mlflow.log_artifact(str(checkpoint_path), artifact_path="model")

        metrics = {"per_class_dice": per_class_dice, "dice_mean": mean_dice}
        metrics_path = checkpoint_path.parent / "metrics.json"
        import json

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")

    logger.info(f"stage 04 complete: dice_mean={mean_dice}")
    return metrics


if __name__ == "__main__":
    run()
