import pytesseract
import numpy as np
from PIL import Image
from loguru import logger

TESSERACT_CONFIG = "--oem 3 --psm 6"


def run_ocr(image: np.ndarray) -> str:
    pil_image = Image.fromarray(image)
    text = pytesseract.image_to_string(pil_image, config=TESSERACT_CONFIG)
    logger.debug(f"OCR extracted {len(text)} characters")
    return text