from __future__ import annotations

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class LesionSegmentationModel(nn.Module):
    def __init__(
        self,
        encoder_name: str = "timm-efficientnet-b3",
        encoder_weights: str | None = None,
        num_classes: int = 4,
    ):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
