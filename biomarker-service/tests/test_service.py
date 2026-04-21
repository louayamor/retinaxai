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

    biomarkers = service.extract_biomarkers(_make_image_bytes())

    assert isinstance(biomarkers, VascularBiomarkers)
    assert biomarkers.tortuosity is not None
    assert biomarkers.vessel_density is not None
    assert biomarkers.raw_feature_vector


def test_extract_biomarkers_rejects_invalid_bytes():
    service = BiomarkerService()

    try:
        service.extract_biomarkers(b"not-an-image")
    except BiomarkerExtractionError as exc:
        assert "invalid image payload" in str(exc)
    else:
        raise AssertionError("Expected BiomarkerExtractionError")
