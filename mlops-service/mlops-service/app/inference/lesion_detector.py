from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from app.training.components.lesion_model import LesionSegmentationModel

CLASS_NAMES = ("ma", "he", "ex", "se")


class LesionDetector:
    def __init__(
        self,
        checkpoint_path: Path,
        class_names: tuple[str, ...] = CLASS_NAMES,
        thresholds: dict[str, float] | None = None,
        encoder_name: str = "timm-efficientnet-b3",
        device: torch.device | None = None,
    ) -> None:
        self.class_names = class_names
        self.thresholds = thresholds or {
            "ma": 0.3,
            "he": 0.4,
            "ex": 0.4,
            "se": 0.35,
        }
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = LesionSegmentationModel(encoder_name=encoder_name)
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, tensor: torch.Tensor) -> dict[str, np.ndarray]:
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)

        tensor = tensor.to(self.device)
        logits = self.model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

        masks: dict[str, np.ndarray] = {}
        for i, cls_name in enumerate(self.class_names):
            threshold = self.thresholds.get(cls_name, 0.3)
            masks[cls_name] = (probs[i] > threshold).astype(np.uint8)

        return masks

    def extract_connected_components(
        self, masks: dict[str, np.ndarray]
    ) -> list[dict]:
        clusters: list[dict] = []
        for cls_name, mask in masks.items():
            num_labels, labels, stats, centroids = (
                cv2.connectedComponentsWithStats(
                    mask, connectivity=8
                )
            )
            for label_id in range(1, num_labels):
                area = int(stats[label_id, cv2.CC_STAT_AREA])
                cx = int(centroids[label_id, 0])
                cy = int(centroids[label_id, 1])
                clusters.append({
                    "class": cls_name,
                    "centroid_x": cx,
                    "centroid_y": cy,
                    "area": area,
                })

        return clusters
