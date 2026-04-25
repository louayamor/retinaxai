from __future__ import annotations

import io

import numpy as np
from PIL import Image

from app.service import BiomarkerExtractionError, BiomarkerService
from app.schemas import VascularBiomarkers


def _make_image_bytes(size: tuple[int, int] = (64, 64), fill: int = 180) -> bytes:
    image = Image.fromarray(np.full((*size, 3), fill, dtype=np.uint8), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_extract_biomarkers_returns_typed_result():
    service = BiomarkerService()

    class DummyAdapter:
        def predict(self, _image_bytes):
            return {
                "tortuosity": 0.2,
                "avr": None,
                "fractal_dimension": 1.2,
                "vessel_density": 0.5,
                "bifurcation_count": 2,
                "bifurcation_angles": None,
                "cre": None,
                "raw_feature_vector": None,
            }

    service._registry.load = lambda: DummyAdapter()

    biomarkers = service.extract_biomarkers(_make_image_bytes())

    assert isinstance(biomarkers, VascularBiomarkers)
    assert biomarkers.tortuosity == 0.2
    assert biomarkers.vessel_density == 0.5
    assert biomarkers.raw_feature_vector is None


def test_extract_biomarkers_rejects_invalid_bytes():
    service = BiomarkerService()

    class DummyAdapter:
        def predict(self, _image_bytes):
            raise BiomarkerExtractionError("invalid image payload")

    service._registry.load = lambda: DummyAdapter()

    try:
        service.extract_biomarkers(b"not-an-image")
    except BiomarkerExtractionError as exc:
        assert "invalid image payload" in str(exc)
    else:
        raise AssertionError("Expected BiomarkerExtractionError")
