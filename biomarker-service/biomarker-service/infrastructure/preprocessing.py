"""
Image preprocessing utilities for the Biomarker Service.

This module handles decoding raw image bytes and normalizing them for the VascX model.
"""

import io
from typing import Any

from PIL import Image
from loguru import logger


def decode_and_normalize_image(image_bytes: bytes) -> Any:
    """
    Decode raw image bytes and normalize for VascX model.
    """
    if not image_bytes:
        raise ValueError("empty image payload")

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        logger.debug("Image decoded successfully with size {}", image.size)
        return image
    except Exception as exc:
        logger.exception("Failed to parse image payload")
        raise ValueError(f"invalid image payload: {exc}") from exc
