import cv2
import numpy as np
from pathlib import Path
from loguru import logger


def load_and_preprocess(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    scaled = cv2.resize(binary, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    logger.debug(f"Preprocessed: {image_path.name} shape={scaled.shape}")
    return scaled  # type: ignore[return-value]


def load_color_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")
    return image  # type: ignore[return-value]