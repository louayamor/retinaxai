import base64
import cv2
import numpy as np
from pathlib import Path
from loguru import logger


def encode_to_base64(image_bgr: np.ndarray) -> str:
    success, buffer = cv2.imencode(".png", image_bgr)  # type: ignore[call-overload]
    if not success:
        raise RuntimeError("Failed to encode image to PNG")
    return base64.b64encode(buffer).decode("utf-8")


def save_region_png(
    image_bgr: np.ndarray,
    images_dir: Path,
    patient_id: str,
    report_key: str,
    region_name: str,
    eye: str = "",
) -> Path:
    patient_dir = images_dir / patient_id
    if eye:
        patient_dir = patient_dir / eye
    patient_dir = patient_dir / report_key
    patient_dir.mkdir(parents=True, exist_ok=True)
    out_path = patient_dir / f"{region_name}.png"
    cv2.imwrite(str(out_path), image_bgr)  # type: ignore[call-overload]
    logger.debug(f"Saved: {out_path}")
    return out_path


def export_regions(
    crops: dict[str, np.ndarray],
    images_dir: Path,
    patient_id: str,
    report_key: str,
    eye: str = "",
) -> dict[str, dict]:
    result = {}
    for region_name, image_bgr in crops.items():
        try:
            png_path = save_region_png(image_bgr, images_dir, patient_id, report_key, region_name, eye)
            b64 = encode_to_base64(image_bgr)
            result[region_name] = {"png_path": str(png_path), "base64_png": b64}
        except Exception as e:
            logger.error(f"Failed export {region_name} for {patient_id}: {e}")
    return result
