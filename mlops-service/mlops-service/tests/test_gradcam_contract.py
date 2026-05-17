from __future__ import annotations

from types import SimpleNamespace

import torch

from app.inference.gradcam_service import GradCAMService


class DummyModel:
    def named_modules(self):
        return []

    def eval(self):
        return self

    def zero_grad(self):
        return None


class DummyTargetLayer:
    def register_forward_hook(self, hook):
        self.forward_hook = hook
        return SimpleNamespace(remove=lambda: None)

    def register_full_backward_hook(self, hook):
        self.backward_hook = hook
        return SimpleNamespace(remove=lambda: None)


def test_generate_with_regions_numeric_returns_stable_contract(monkeypatch):
    model = DummyModel()
    service = GradCAMService(model, target_layer=DummyTargetLayer())

    cam = torch.zeros((224, 224), dtype=torch.float32)
    cam[100:120, 80:100] = 1.0

    class DummyImage:
        def resize(self, *_args, **_kwargs):
            return self

    monkeypatch.setattr(
        service, "_build_cam_data", lambda *_args: (cam.numpy(), DummyImage(), "base64")
    )

    gradcam_base64, regions, hotspots = service.generate_with_regions_numeric(
        b"image-bytes",
        torch.zeros((1, 3, 224, 224), dtype=torch.float32),
        1,
    )

    assert gradcam_base64 == "base64"
    assert isinstance(regions, list)
    assert isinstance(hotspots, list)
    assert regions
    assert hotspots
    assert set(regions[0].keys()) == {
        "name",
        "intensity",
        "area",
        "center_x",
        "center_y",
        "saliency_score",
    }
    assert hotspots[0]["rank"] == 1


def test_generate_with_regions_numeric_falls_back_when_no_contours(monkeypatch):
    model = DummyModel()
    service = GradCAMService(model, target_layer=DummyTargetLayer())

    cam = torch.zeros((224, 224), dtype=torch.float32)

    class DummyImage:
        def resize(self, *_args, **_kwargs):
            return self

    monkeypatch.setattr(
        service, "_build_cam_data", lambda *_args: (cam.numpy(), DummyImage(), "base64")
    )

    _, regions, hotspots = service.generate_with_regions_numeric(
        b"image-bytes",
        torch.zeros((1, 3, 224, 224), dtype=torch.float32),
        1,
    )

    assert len(regions) == 1
    assert len(hotspots) == 1
    assert regions[0]["area"] == 1
