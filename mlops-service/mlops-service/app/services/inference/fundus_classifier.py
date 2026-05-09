from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import timm
import torch
import torch.nn as nn
from loguru import logger
from PIL import Image
from torchvision import transforms

from app.constants import PARAMS_FILE_PATH
from app.domains.imaging.preprocessing import preprocess_fundus_image
from app.utils.common import read_yaml


class FundusClassifierService:
    """Binary classifier: is this image a valid fundus photograph?"""

    def __init__(
        self,
        model_path: Path,
        model_name: str,
        image_size: int,
        device: torch.device,
        threshold: float = 0.3,
    ):
        self.model_path = model_path
        self.model_name = model_name
        self.image_size = image_size
        self.device = device
        self.threshold = threshold
        self._model: Optional[nn.Module] = None
        self._transform: Optional[transforms.Compose] = None

        params = read_yaml(PARAMS_FILE_PATH)
        norm = params.augmentation.normalize
        self._transform = transforms.Compose(
            [
                transforms.Lambda(
                    lambda img: preprocess_fundus_image(img, image_size=image_size)
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm.mean, std=norm.std),
            ]
        )

    def _load_model(self) -> nn.Module:
        if self._model is not None:
            return self._model

        if not self.model_path.exists():
            raise FileNotFoundError(f"fundus classifier not found: {self.model_path}")

        model = timm.create_model(
            self.model_name,
            pretrained=False,
            num_classes=2,
            drop_rate=0.1,
        )

        state_dict = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        self._model = model
        logger.info(f"[FUNDUS] classifier loaded from {self.model_path}")
        return model

    def is_fundus(self, image_bytes: bytes) -> tuple[bool, float, str]:
        """
        Check if image is a valid fundus photograph.

        Returns:
            (is_valid, fundus_score, message)
            - is_valid: True if fundus_score >= threshold
            - fundus_score: probability that image is fundus (0.0 - 1.0)
            - message: human-readable explanation
        """
        model = self._load_model()

        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            original_size = img.size
        except Exception as e:
            raise ValueError(f"Failed to open image: {e}") from e

        tensor = self._transform(img).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        fundus_score = float(probs[1])
        is_valid = fundus_score >= self.threshold

        if is_valid:
            message = (
                f"fundus image accepted (score={fundus_score:.3f} >= {self.threshold})"
            )
        else:
            message = (
                f"not a fundus image (score={fundus_score:.3f} < {self.threshold})"
            )

        logger.info(
            f"[FUNDUS] validation: original_size={original_size} "
            f"→ resized={self.image_size}x{self.image_size} "
            f"→ fundus_score={fundus_score:.3f} threshold={self.threshold} "
            f"→ {'ACCEPTED' if is_valid else 'REJECTED'}"
        )

        return is_valid, fundus_score, message
