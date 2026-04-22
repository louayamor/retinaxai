from __future__ import annotations

import io
from dataclasses import dataclass
from math import atan2, degrees

import cv2
import numpy as np
from PIL import Image

from app.schemas import VascularBiomarkers
from loguru import logger


class BiomarkerExtractionError(ValueError):
    pass


@dataclass(slots=True)
class BiomarkerService:
    service_name: str = "biomarker-service"
    service_version: str = "0.1.0"

    def extract_biomarkers(self, _image_bytes: bytes) -> VascularBiomarkers:
        logger.info("starting biomarker feature extraction")
        image = self._load_image(_image_bytes)
        logger.debug("loaded image shape={} dtype={}", image.shape, image.dtype)
        vessel_mask = self._segment_vessels(image)
        logger.debug(
            "segmented vessels mask_shape={} vessel_pixels={}",
            vessel_mask.shape,
            int(cv2.countNonZero(vessel_mask)),
        )
        skeleton = self._skeletonize(vessel_mask)
        logger.debug(
            "skeletonized vessel tree skeleton_pixels={}",
            int(cv2.countNonZero(skeleton)),
        )

        vessel_density = self._vessel_density(vessel_mask)
        tortuosity = self._tortuosity(skeleton)
        bifurcation_count, bifurcation_angles = self._bifurcations(skeleton)
        avr, cre = self._calibrate_vessel_widths(vessel_mask, skeleton)
        fractal_dimension = self._fractal_dimension(vessel_mask)

        logger.info(
            "computed biomarkers vessel_density={:.4f} tortuosity={:.4f} bifurcations={} avr={} fractal_dimension={:.4f}",
            vessel_density,
            tortuosity,
            bifurcation_count,
            avr,
            fractal_dimension,
        )
        logger.debug(
            "biomarker detail bifurcation_angles={} cre={}",
            bifurcation_angles,
            cre,
        )

        return VascularBiomarkers(
            tortuosity=self._clamp01(tortuosity),
            avr=avr,
            fractal_dimension=fractal_dimension,
            vessel_density=self._clamp01(vessel_density),
            bifurcation_count=bifurcation_count,
            bifurcation_angles=bifurcation_angles,
            cre=cre,
            raw_feature_vector=[
                float(tortuosity),
                float(avr or 0.0),
                float(fractal_dimension),
                float(vessel_density),
                float(bifurcation_count),
            ],
        )

    def _load_image(self, image_bytes: bytes) -> np.ndarray:
        if not image_bytes:
            logger.warning("received empty image payload")
            raise BiomarkerExtractionError("empty image payload")

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            logger.exception("failed to parse image payload")
            raise BiomarkerExtractionError(f"invalid image payload: {exc}") from exc

        return np.array(image)

    def _segment_vessels(self, image: np.ndarray) -> np.ndarray:
        logger.debug("starting vessel segmentation")
        green = image[:, :, 1]
        blurred = cv2.GaussianBlur(green, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)

        _, vessel_mask = cv2.threshold(
            enhanced,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        vessel_mask = cv2.morphologyEx(vessel_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        vessel_mask = cv2.morphologyEx(vessel_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        logger.debug("completed vessel segmentation")
        return vessel_mask

    def _skeletonize(self, mask: np.ndarray) -> np.ndarray:
        logger.debug("starting skeletonization")
        skeleton = np.zeros(mask.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        working = mask.copy()

        while True:
            eroded = cv2.erode(working, element)
            opened = cv2.dilate(eroded, element)
            temp = cv2.subtract(working, opened)
            skeleton = cv2.bitwise_or(skeleton, temp)
            working = eroded.copy()
            if cv2.countNonZero(working) == 0:
                break

        logger.debug("completed skeletonization")
        return skeleton

    def _vessel_density(self, vessel_mask: np.ndarray) -> float:
        total = vessel_mask.size
        if total == 0:
            return 0.0
        return float(cv2.countNonZero(vessel_mask) / total)

    def _tortuosity(self, skeleton: np.ndarray) -> float:
        logger.debug("computing tortuosity")
        contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        ratios: list[float] = []
        for contour in contours:
            if len(contour) < 5:
                continue
            path_len = float(cv2.arcLength(contour, False))
            start = contour[0][0]
            end = contour[-1][0]
            straight = float(np.linalg.norm(start - end))
            if straight > 0:
                ratios.append(path_len / straight)
        return float(np.mean(ratios)) if ratios else 0.0

    def _bifurcations(self, skeleton: np.ndarray) -> tuple[int, list[float]]:
        logger.debug("computing bifurcations")
        padded = np.pad((skeleton > 0).astype(np.uint8), 1)
        bifurcation_points = np.zeros_like(skeleton, dtype=bool)

        for y in range(1, padded.shape[0] - 1):
            for x in range(1, padded.shape[1] - 1):
                if padded[y, x] == 0:
                    continue
                neighborhood = padded[y - 1 : y + 2, x - 1 : x + 2]
                neighbors = int(neighborhood.sum() - 1)
                if neighbors >= 3:
                    bifurcation_points[y - 1, x - 1] = True

        points = np.argwhere(bifurcation_points)
        angles: list[float] = []
        for y, x in points[:20]:
            patch = padded[y : y + 3, x : x + 3]
            coords = np.argwhere(patch > 0) - 1
            if len(coords) >= 3:
                vecs = coords[:3].astype(float)
                base = vecs[0]
                for vec in vecs[1:3]:
                    angle = abs(degrees(atan2(vec[0], vec[1]) - atan2(base[0], base[1])))
                    angles.append(round(angle % 180.0, 2))

        return int(len(points)), angles[:10]

    def _calibrate_vessel_widths(self, vessel_mask: np.ndarray, skeleton: np.ndarray) -> tuple[float | None, dict]:
        logger.debug("computing vessel width calibration")
        if cv2.countNonZero(vessel_mask) == 0 or cv2.countNonZero(skeleton) == 0:
            return None, {"artery_cre": None, "vein_cre": None}

        distance = cv2.distanceTransform(vessel_mask, cv2.DIST_L2, 5)
        skeleton_points = skeleton > 0
        widths = (distance[skeleton_points] * 2.0).astype(float)
        widths = widths[widths > 0]
        if widths.size == 0:
            return None, {"artery_cre": None, "vein_cre": None}

        artery_width = float(np.percentile(widths, 25))
        vein_width = float(np.percentile(widths, 75))
        avr = float(artery_width / vein_width) if vein_width > 0 else None
        cre = {
            "artery_cre": round(artery_width, 4),
            "vein_cre": round(vein_width, 4),
            "width_samples": int(widths.size),
        }
        return avr, cre

    def _fractal_dimension(self, vessel_mask: np.ndarray) -> float:
        logger.debug("computing fractal dimension")
        binary = vessel_mask > 0
        if not binary.any():
            return 0.0

        def boxcount(arr: np.ndarray, k: int) -> int:
            S = np.add.reduceat(
                np.add.reduceat(arr, np.arange(0, arr.shape[0], k), axis=0),
                np.arange(0, arr.shape[1], k),
                axis=1,
            )
            return int(((S > 0) & (S < k * k)).sum())

        p = min(binary.shape)
        n = 2 ** int(np.floor(np.log2(p)))
        if n < 4:
            return 0.0

        sizes = 2 ** np.arange(1, int(np.log2(n)))
        counts = []
        for size in sizes:
            counts.append(boxcount(binary[:n, :n], int(size)))

        valid = [(s, c) for s, c in zip(sizes, counts) if c > 0]
        if len(valid) < 2:
            return 0.0

        sizes_arr = np.array([v[0] for v in valid], dtype=float)
        counts_arr = np.array([v[1] for v in valid], dtype=float)
        coeffs = np.polyfit(np.log(1.0 / sizes_arr), np.log(counts_arr), 1)
        return float(max(0.0, coeffs[0]))

    def _clamp01(self, value: float) -> float:
        return float(max(0.0, min(1.0, value)))
