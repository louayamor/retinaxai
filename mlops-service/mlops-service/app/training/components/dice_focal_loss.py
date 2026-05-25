from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceFocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        dice_loss = self._dice_loss(probs, targets)
        focal_loss = self._focal_loss(probs, targets)

        return self.dice_weight * dice_loss + self.focal_weight * focal_loss

    def _dice_loss(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return (1.0 - dice).mean()

    def _focal_loss(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(probs, targets, reduction="none")
        pt = torch.where(targets == 1, probs, 1.0 - probs)
        focal_weight = (1.0 - pt) ** self.gamma
        return (focal_weight * bce).mean()
