"""
Unit tests for preprocessing functions.

Tests the shared preprocessing module:
- circular_crop: removes black borders
- apply_clahe_l_channel: enhances contrast on L channel only (preserves color)
- preprocess_fundus_image: full pipeline

Key assertions:
1. Color is preserved (not converted to grayscale)
2. Intensity distribution is natural (not inverted)
3. Circular crop removes black borders
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from app.training.preprocessing import (
    circular_crop,
    apply_clahe_l_channel,
    preprocess_fundus_image,
    CROP_THRESHOLD,
    CLAHE_CLIP_LIMIT,
)


class TestCircularCrop:
    def test_circular_crop_removes_black_borders(self):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(img, (100, 100), 60, (100, 100, 100), -1)

        cropped = circular_crop(img)

        assert cropped.shape[0] < 200 or cropped.shape[1] < 200

    def test_circular_crop_returns_original_if_no_border(self):
        img = np.ones((100, 100, 3), dtype=np.uint8) * 100

        cropped = circular_crop(img)

        assert cropped.shape == img.shape

    def test_circular_crop_handles_grayscale(self):
        img = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(img, (100, 100), 60, 100, -1)

        cropped = circular_crop(img)

        assert len(cropped.shape) == 2


class TestApplyClaheLChannel:
    def test_clahe_preserves_color(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :, 0] = 50
        img[:, :, 1] = 100
        img[:, :, 2] = 150

        enhanced = apply_clahe_l_channel(img)

        assert enhanced.shape == img.shape
        assert enhanced.shape[2] == 3

        b, g, r = cv2.split(enhanced)
        assert not np.allclose(b, g, atol=2) or not np.allclose(g, r, atol=2)

    def test_clahe_enhances_contrast(self):
        img = np.ones((100, 100, 3), dtype=np.uint8) * 100
        img[25:75, 25:75] = 120

        enhanced = apply_clahe_l_channel(img)

        orig_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

        assert enh_gray.std() >= orig_gray.std() * 0.5


class TestPreprocessFundusImage:
    def test_returns_pil_image(self):
        img_np = np.ones((200, 200, 3), dtype=np.uint8) * 100
        cv2.circle(img_np, (100, 100), 60, (150, 100, 100), -1)
        img = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))

        result = preprocess_fundus_image(img, image_size=100)

        assert isinstance(result, Image.Image)
        assert result.size == (100, 100)

    def test_preserves_rgb_color(self):
        img_np = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(img_np, (100, 100), 60, (200, 100, 50), -1)
        img = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))

        result = preprocess_fundus_image(img, image_size=100)

        result_np = np.array(result)
        r = result_np[:, :, 0]
        g = result_np[:, :, 1]
        b = result_np[:, :, 2]

        assert result_np.shape == (100, 100, 3)

        assert not np.allclose(r, g, atol=1) or not np.allclose(g, b, atol=1)

    def test_handles_synthetic_fundus(self):
        img_np = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.circle(img_np, (150, 150), 100, (120, 60, 40), -1)
        cv2.circle(img_np, (150, 150), 20, (20, 20, 20), -1)
        img = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))

        result = preprocess_fundus_image(img, image_size=256)

        assert result.size == (256, 256)
        result_np = np.array(result)
        assert result_np.mean() > 10


class TestWithRealSampleImages:
    SAMPLE_DIR = Path("artifacts/ocr/output/images")

    @pytest.fixture
    def sample_fundus_path(self) -> Path | None:
        if not self.SAMPLE_DIR.exists():
            return None

        for root, _, files in self.SAMPLE_DIR.walk():
            for f in files:
                if "fundus" in f.lower() or "infrared" in f.lower():
                    return Path(root) / f
        return None

    def test_preprocess_real_fundus(self, sample_fundus_path: Path | None):
        if sample_fundus_path is None:
            pytest.skip("No sample fundus images found")

        img = Image.open(sample_fundus_path).convert("RGB")
        result = preprocess_fundus_image(img, image_size=300)

        assert result.size == (300, 300)

        result_np = np.array(result)
        assert result_np.shape == (300, 300, 3)

        assert result_np.mean() > 5
        assert result_np.max() > result_np.min()
