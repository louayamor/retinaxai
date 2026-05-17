from __future__ import annotations

import numpy as np
import torch

from app.inference.gradcam_service import GradCAMService


class DummyModel:
    def eval(self):
        return self


class DummyTargetLayer:
    def register_forward_hook(self, hook):
        self.forward_hook = hook
        return type("H", (), {"remove": lambda self: None})()

    def register_full_backward_hook(self, hook):
        self.backward_hook = hook
        return type("H", (), {"remove": lambda self: None})()


def test_extract_regions_numeric_preserves_duplicate_labels(monkeypatch):
    service = GradCAMService(DummyModel(), target_layer=DummyTargetLayer())
    cam = np.zeros((224, 224), dtype=np.float32)
    cam[20:35, 20:35] = 0.92
    cam[160:175, 160:175] = 0.91
    img = np.zeros((224, 224, 3), dtype=np.uint8)

    monkeypatch.setattr(service, "_map_to_anatomical_region", lambda *args, **kwargs: "macula_center")

    regions = service.extract_regions_numeric(cam, torch.tensor(img))

    assert len(regions) == 2
    assert regions[0]["name"] == "macula_center"
    assert regions[1]["name"] == "macula_center"
    assert regions[0]["intensity"] >= regions[1]["intensity"]


def test_extract_regions_numeric_is_deterministic_for_ties(monkeypatch):
    service = GradCAMService(DummyModel(), target_layer=DummyTargetLayer())
    cam = np.zeros((224, 224), dtype=np.float32)
    cam[10:20, 10:20] = 0.90
    cam[50:60, 50:60] = 0.90
    img = np.zeros((224, 224, 3), dtype=np.uint8)

    labels = iter(["region_b", "region_a"])
    monkeypatch.setattr(service, "_map_to_anatomical_region", lambda *args, **kwargs: next(labels))

    regions = service.extract_regions_numeric(cam, torch.tensor(img))

    assert [r["name"] for r in regions] == ["region_a", "region_b"]


def test_get_top_hotspots_uses_full_region_list():
    service = GradCAMService(DummyModel(), target_layer=DummyTargetLayer())
    regions = [
        {"name": "r1", "intensity": 0.2, "area": 5, "center_x": 1, "center_y": 1, "saliency_score": 0.01},
        {"name": "r2", "intensity": 0.9, "area": 10, "center_x": 2, "center_y": 2, "saliency_score": 0.03},
        {"name": "r3", "intensity": 0.5, "area": 7, "center_x": 3, "center_y": 3, "saliency_score": 0.02},
        {"name": "r4", "intensity": 0.8, "area": 9, "center_x": 4, "center_y": 4, "saliency_score": 0.04},
        {"name": "r5", "intensity": 0.7, "area": 8, "center_x": 5, "center_y": 5, "saliency_score": 0.05},
        {"name": "r6", "intensity": 0.6, "area": 6, "center_x": 6, "center_y": 6, "saliency_score": 0.06},
    ]

    hotspots = service._get_top_hotspots(regions)

    assert [h["region"] for h in hotspots] == ["r2", "r4", "r5", "r6", "r3"]
    assert [h["rank"] for h in hotspots] == [1, 2, 3, 4, 5]
