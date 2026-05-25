from __future__ import annotations

import numpy as np

from app.inference.lesion_detector import LesionDetector


def test_extract_connected_components() -> None:
    detector = LesionDetector.__new__(LesionDetector)
    detector.class_names = ("ma", "he", "ex", "se")
    detector.thresholds = {"ma": 0.3, "he": 0.3, "ex": 0.3, "se": 0.3}

    masks = {
        "ma": np.zeros((64, 64), dtype=np.uint8),
        "he": np.zeros((64, 64), dtype=np.uint8),
        "ex": np.zeros((64, 64), dtype=np.uint8),
        "se": np.zeros((64, 64), dtype=np.uint8),
    }
    masks["ma"][10:20, 10:20] = 1
    masks["he"][30:40, 30:40] = 1

    clusters = detector.extract_connected_components(masks)

    assert len(clusters) == 2
    cls_names = {c["class"] for c in clusters}
    assert "ma" in cls_names
    assert "he" in cls_names
    for c in clusters:
        assert c["area"] == 100
        assert isinstance(c["centroid_x"], int)
        assert isinstance(c["centroid_y"], int)


def test_extract_connected_components_empty() -> None:
    detector = LesionDetector.__new__(LesionDetector)
    detector.class_names = ("ma", "he", "ex", "se")
    detector.thresholds = {}

    masks = {
        cls_name: np.zeros((64, 64), dtype=np.uint8)
        for cls_name in ("ma", "he", "ex", "se")
    }
    clusters = detector.extract_connected_components(masks)
    assert clusters == []
