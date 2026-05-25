from __future__ import annotations

import sys
from pathlib import Path

import dagshub
import mlflow
import torch
from loguru import logger
from torch.utils.data import DataLoader

from app.constants import PARAMS_FILE_PATH
from app.training.components.ddr_lesion_dataset import DDRLesionDataset
from app.training.components.lesion_model import LesionSegmentationModel
from app.training.components.lesion_trainer import LesionTrainer
from app.utils.common import read_yaml

logger.remove()
logger.add(sys.stdout, serialize=True)

ARTIFACTS_DIR = Path("artifacts")
MANIFESTS_DIR = ARTIFACTS_DIR / "lesion" / "manifests"
LESION_CHECKPOINT = ARTIFACTS_DIR / "model" / "lesion" / "model.pth"
GRADER_CHECKPOINT = ARTIFACTS_DIR / "model" / "imaging" / "model.pth"


def run() -> None:
    logger.info(">>> stage 03: lesion model training started")

    params = read_yaml(PARAMS_FILE_PATH)
    lesion_cfg = params.get("lesion_model", {})
    image_size = int(lesion_cfg.get("image_size", 384))
    batch_size = int(lesion_cfg.get("batch_size", 8))
    num_workers = int(lesion_cfg.get("num_workers", 4))
    lr = float(lesion_cfg.get("lr", 0.0001))
    epochs = int(lesion_cfg.get("epochs", 100))
    patience = int(lesion_cfg.get("patience", 15))
    encoder_name = str(lesion_cfg.get("encoder_name", "timm-efficientnet-b3"))
    grader_ckpt = Path(str(lesion_cfg.get("grader_checkpoint", GRADER_CHECKPOINT)))
    output_ckpt = Path(str(lesion_cfg.get("checkpoint_path", LESION_CHECKPOINT)))

    from app.utils.common import require_cuda

    device = require_cuda()

    train_csv = MANIFESTS_DIR / "train.csv"
    val_csv = MANIFESTS_DIR / "val.csv"

    if not train_csv.exists():
        raise FileNotFoundError(f"Train manifest not found: {train_csv}")
    if not val_csv.exists():
        raise FileNotFoundError(f"Val manifest not found: {val_csv}")

    train_dataset = DDRLesionDataset(train_csv, image_size=image_size)
    val_dataset = DDRLesionDataset(val_csv, image_size=image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(
        f"Dataset: {len(train_dataset)} train, {len(val_dataset)} val"
    )

    model = LesionSegmentationModel(encoder_name=encoder_name)

    if grader_ckpt.exists():
        logger.info(f"Transferring grader weights from {grader_ckpt}")
        LesionTrainer.transfer_grader_weights(grader_ckpt, model)
    else:
        logger.warning(
            f"Grader checkpoint not found at {grader_ckpt} — "
            "training from scratch"
        )

    LesionTrainer.freeze_encoder(model)

    trainer = LesionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=lr,
        epochs=epochs,
        patience=patience,
        device=device,
        checkpoint_path=output_ckpt,
    )

    dagshub.init(repo_owner="louayamor", repo_name="retinaxai", mlflow=True)

    with mlflow.start_run(run_name="stage_03_lesion_training"):
        mlflow.log_params({
            "encoder_name": encoder_name,
            "image_size": image_size,
            "batch_size": batch_size,
            "lr": lr,
            "epochs": epochs,
            "patience": patience,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
        })

        best_ckpt = trainer.train()
        mlflow.log_artifact(str(best_ckpt), artifact_path="model")

        logger.info(f"stage 03 complete: best model at {best_ckpt}")


if __name__ == "__main__":
    run()
