import numpy as np
from loguru import logger


def extract_regions(image_bgr: np.ndarray, regions_config: dict) -> dict[str, np.ndarray]:
    h, w = image_bgr.shape[:2]
    crops = {}

    for name, spec in regions_config.items():
        x1_rel, y1_rel, x2_rel, y2_rel = spec["rel"]
        x1, y1 = int(x1_rel * w), int(y1_rel * h)
        x2, y2 = int(x2_rel * w), int(y2_rel * h)
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            logger.warning(f"Empty crop: {name}")
            continue
        crops[name] = crop
        logger.debug(f"Region extracted: {name} shape={crop.shape}")

    return crops
