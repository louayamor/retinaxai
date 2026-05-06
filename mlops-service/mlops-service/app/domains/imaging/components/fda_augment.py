from __future__ import annotations

from pathlib import Path

import torch
import numpy as np
from loguru import logger
from PIL import Image


class FDAAugment:
    """
    Fourier Domain Adaptation augmentation.

    Extracts the mean amplitude spectrum from target domain images once,
    then during training blends source image amplitude with the pre-computed
    target amplitude spectrum. Low-frequency amplitude carries style
    (illumination, contrast, color balance); phase carries structure
    (vessel topology, lesions).

    No target labels or target image content is used — only the aggregate
    frequency-domain amplitude statistics.

    Reference: Yang & Soatto, "FDA: Fourier Domain Adaptation for Semantic
    Segmentation", CVPR 2020.
    """

    def __init__(
        self,
        target_images_dir: Path,
        beta: float = 0.15,
        probability: float = 0.5,
        cache_path: Path | None = None,
        source_amp_cache_path: Path | None = None,
    ):
        if beta < 0.0 or beta > 1.0:
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        if probability < 0.0 or probability > 1.0:
            raise ValueError(f"probability must be in [0, 1], got {probability}")

        self.beta = beta
        self.probability = probability
        self.target_images_dir = Path(target_images_dir)
        self.cache_path = (
            Path(cache_path)
            if cache_path
            else self.target_images_dir / "fda_amplitude_target.pt"
        )
        self.source_amp_cache_path = source_amp_cache_path

        self._target_amp: torch.Tensor | None = None
        self._source_amp: torch.Tensor | None = None

    @property
    def target_amplitude(self) -> torch.Tensor:
        if self._target_amp is None:
            self._target_amp = self._load_or_compute_amplitude(
                self.target_images_dir, self.cache_path, label="target"
            )
        return self._target_amp

    @property
    def source_amplitude(self) -> torch.Tensor | None:
        if self._source_amp is None and self.source_amp_cache_path:
            if Path(self.source_amp_cache_path).exists():
                self._source_amp = torch.load(self.source_amp_cache_path)
                logger.info(
                    f"loaded source amplitude from {self.source_amp_cache_path}"
                )
        return self._source_amp

    def set_source_amplitude(
        self, source_dir: Path, cache_path: Path | None = None
    ) -> None:
        save_path = (
            Path(cache_path) if cache_path else source_dir / "fda_amplitude_source.pt"
        )
        self._source_amp = self._load_or_compute_amplitude(
            source_dir, save_path, label="source"
        )
        self.source_amp_cache_path = save_path

    def _load_or_compute_amplitude(
        self, images_dir: Path, cache_path: Path, label: str = ""
    ) -> torch.Tensor:
        if cache_path.exists():
            tensor = torch.load(cache_path)
            logger.info(f"loaded {label} FDA amplitude from cache: {cache_path}")
            return tensor

        logger.info(f"computing {label} FDA amplitude from: {images_dir}")
        image_paths = sorted(images_dir.rglob("*.png"))
        if not image_paths:
            raise FileNotFoundError(f"no PNG images found in {images_dir}")

        amplitudes: list[torch.Tensor] = []
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((224, 224))
                tensor = (
                    torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
                )
                fft = torch.fft.fft2(tensor)
                amp = torch.abs(fft)
                amplitudes.append(amp)
            except Exception as e:
                logger.warning(f"skipping {img_path}: {e}")

        if not amplitudes:
            raise RuntimeError(f"no valid images processed from {images_dir}")

        mean_amp = torch.stack(amplitudes).mean(dim=0)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(mean_amp, cache_path)
        logger.info(
            f"{label} FDA amplitude cached ({len(amplitudes)} images): {cache_path}"
        )
        return mean_amp

    def __call__(self, img_tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.probability:
            return img_tensor
        return self._fda_adapt(img_tensor, self.target_amplitude.to(img_tensor.device))

    def inverse(self, img_tensor: torch.Tensor) -> torch.Tensor:
        if self._source_amp is None:
            logger.warning(
                "FDA inverse called but no source amplitude set; returning original"
            )
            return img_tensor
        return self._fda_adapt(img_tensor, self._source_amp.to(img_tensor.device))

    def _fda_adapt(
        self, img_tensor: torch.Tensor, target_amp: torch.Tensor
    ) -> torch.Tensor:
        fft = torch.fft.fft2(img_tensor)
        src_amp = torch.abs(fft)
        phase = torch.angle(fft)

        blended_amp = (1.0 - self.beta) * src_amp + self.beta * target_amp
        adapted_fft = blended_amp * torch.exp(1j * phase)

        return torch.fft.ifft2(adapted_fft).real.clamp(0.0, 1.0)
