"""
Shared preprocessing functions for fundus images.

This module provides standardized preprocessing that is used consistently
across training, evaluation, and inference.

Preprocessing pipeline:
1. Circular crop (remove black borders)
2. CLAHE (contrast enhancement on L channel, preserves color)
3. Resize to target size

Note: Histogram matching was REMOVED because:
1. The original implementation was buggy and inverted intensities
2. CLAHE alone is sufficient for contrast normalization
3. It caused domain mismatch issues
"""

from __future__ import annotations

import io

import cv2
import numpy as np
from pathlib import Path
from PIL import Image


CROP_THRESHOLD = 10
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)


def circular_crop(img_np: np.ndarray) -> np.ndarray:
    """
    Remove black borders from fundus image.

    Uses thresholding to find the retinal region and crops to the bounding box.

    Args:
        img_np: Input image in BGR format (OpenCV) or grayscale

    Returns:
        Cropped image, or original if no border detected
    """
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_np if len(img_np.shape) == 2 else img_np[:, :, 0]
    _, mask = cv2.threshold(gray, CROP_THRESHOLD, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        if w > 10 and h > 10:
            return img_np[y : y + h, x : x + w]
    return img_np


def apply_clahe_l_channel(img_np: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to the L channel only (preserves RGB color information).

    This is the standard approach for fundus image enhancement:
    - Convert to LAB color space
    - Apply CLAHE to L (lightness) channel
    - Convert back to BGR

    Args:
        img_np: Input image in BGR format

    Returns:
        Contrast-enhanced image in BGR format
    """
    lab = cv2.cvtColor(img_np, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE
    )
    l_enhanced = clahe.apply(l_channel)
    lab[:, :, 0] = l_enhanced
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess_fundus_image(
    img: Image.Image,
    image_size: int = 384,
    apply_crop: bool = True,
    apply_clahe: bool = True,
) -> Image.Image:
    """
    Apply standardized preprocessing to a fundus image.

    Pipeline:
    1. Convert to RGB (if not already)
    2. Convert to numpy array in BGR format (for OpenCV)
    3. Circular crop (remove black borders) - if apply_crop=True
    4. CLAHE on L channel (preserve color) - if apply_clahe=True
    5. Convert back to RGB PIL Image
    6. Resize to target size

    Args:
        img: Input PIL Image
        image_size: Target size for resizing (default: 384)
        apply_crop: Whether to apply circular crop
        apply_clahe: Whether to apply CLAHE contrast enhancement

    Returns:
        Preprocessed PIL Image
    """
    img = img.convert("RGB")
    img_np = np.array(img)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    if apply_crop:
        img_np = circular_crop(img_np)

    if apply_clahe:
        img_np = apply_clahe_l_channel(img_np)

    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_np)

    img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
    return img


def preprocess_image_bytes(
    image_bytes: bytes,
    image_size: int = 384,
    apply_crop: bool = True,
    apply_clahe: bool = True,
) -> Image.Image:
    """
    Preprocess fundus image from raw bytes.

    Convenience wrapper for preprocessing uploaded images during inference.

    Args:
        image_bytes: Raw image bytes (e.g., from uploaded file)
        image_size: Target size for resizing
        apply_crop: Whether to apply circular crop
        apply_clahe: Whether to apply CLAHE

    Returns:
        Preprocessed PIL Image
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return preprocess_fundus_image(img, image_size, apply_crop, apply_clahe)
