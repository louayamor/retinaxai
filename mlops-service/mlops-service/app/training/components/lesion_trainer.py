from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader

from app.training.components.dice_focal_loss import DiceFocalLoss
from app.training.components.lesion_model import LesionSegmentationModel


class LesionTrainer:
    MIN_MATCH_RATIO: float = 0.9

    def __init__(
        self,
        model: LesionSegmentationModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-4,
        epochs: int = 100,
        patience: int = 15,
        device: torch.device | None = None,
        checkpoint_path: Path | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        if device is not None:
            self.device = device
        else:
            from app.utils.common import require_cuda

            self.device = require_cuda()
        self.checkpoint_path = checkpoint_path or Path("artifacts/model/lesion/model.pth")

        self.model.to(self.device)
        self.criterion = DiceFocalLoss()
        self.optimizer = optim.Adam(
            self.model.model.decoder.parameters(), lr=self.lr
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

    @staticmethod
    def transfer_grader_weights(
        grader_checkpoint: Path, model: LesionSegmentationModel
    ) -> int:
        grader_sd = torch.load(grader_checkpoint, map_location="cpu")

        encoder_sd = {}
        matched = 0
        total = 0
        for k, v in grader_sd.items():
            if k.startswith("classifier"):
                continue
            remapped_key = f"model.{k}"
            if remapped_key in model.model.encoder.state_dict():
                encoder_sd[remapped_key] = v
                matched += 1
            total += 1

        match_ratio = matched / total if total > 0 else 0.0
        logger.info(
            f"Encoder weight transfer: {matched}/{total} keys matched "
            f"({match_ratio:.2%})"
        )

        if match_ratio < LesionTrainer.MIN_MATCH_RATIO:
            raise RuntimeError(
                f"Encoder weight transfer failed: match ratio {match_ratio:.2%} "
                f"below threshold {LesionTrainer.MIN_MATCH_RATIO:.0%}"
            )

        model.model.encoder.load_state_dict(encoder_sd, strict=False)
        logger.info("Grader weights transferred to lesion model encoder")
        return matched

    @staticmethod
    def freeze_encoder(model: LesionSegmentationModel) -> None:
        for param in model.model.encoder.parameters():
            param.requires_grad_(False)
        logger.info("Encoder frozen")

    def train(self) -> Path:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch()
            val_loss = self._validate_epoch()

            self.scheduler.step(val_loss)

            logger.info(
                f"epoch {epoch}/{self.epochs}: "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
                logger.info(f"checkpoint saved: {self.checkpoint_path}")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(
                        f"early stopping triggered after {epoch} epochs"
                    )
                    break

        logger.info(
            f"training complete: best_val_loss={best_val_loss:.4f}"
        )
        return self.checkpoint_path

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for images, masks in self.train_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, masks)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.inference_mode()
    def _validate_epoch(self) -> float:
        self.model.eval()
        total_loss = 0.0
        for images, masks in self.val_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, masks)
            total_loss += loss.item()

        return total_loss / len(self.val_loader)
