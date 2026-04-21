from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch

import app.services.inference.inference_service as module


class DummyImage:
    def convert(self, *_args, **_kwargs):
        return self


class DummyTensor:
    def unsqueeze(self, *_args, **_kwargs):
        return self

    def to(self, *_args, **_kwargs):
        return self


class DummyModel:
    def __init__(self):
        self.device = torch.device("cuda")
        self.calls = 0

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def __call__(self, _tensor):
        self.calls += 1
        if self.device.type == "cuda":
            raise RuntimeError("CUDA out of memory. Tried to allocate 442.00 MiB.")
        return torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]], dtype=torch.float32)


@contextmanager
def _noop_context(*_args, **_kwargs):
    yield


def test_predict_imaging_retries_on_cuda_oom(monkeypatch):
    service = module.InferenceService.__new__(module.InferenceService)
    model = DummyModel()

    service.device = torch.device("cuda")
    service.settings = SimpleNamespace(imaging_model_path=Path("/tmp/model.pth"))
    service.params = SimpleNamespace(
        dl_training=SimpleNamespace(num_classes=5, dropout=0.3),
        augmentation=SimpleNamespace(
            normalize=SimpleNamespace(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ),
    )
    service.schema = {}
    service._imaging_model = model
    service._clinical_model = None
    service._feature_meta = None
    service._clinical_encoders = None
    service._clinical_numeric_medians = None

    monkeypatch.setattr(service, "_load_imaging_model", lambda: model)
    monkeypatch.setattr(service, "_build_transform", lambda: lambda _img: DummyTensor())
    monkeypatch.setattr(module.Image, "open", lambda *_args, **_kwargs: DummyImage())
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(module.torch.cuda.amp, "autocast", _noop_context)
    monkeypatch.setattr(
        module,
        "INFERENCE_LATENCY",
        SimpleNamespace(
            labels=lambda **_kwargs: SimpleNamespace(observe=lambda *_args: None)
        ),
    )

    result = module.InferenceService.predict_imaging(service, b"fake-image-bytes")

    assert result["predicted_grade"] == 4
    assert result["predicted_label"] == "Proliferative DR"
    assert service.device.type == "cpu"
    assert model.calls == 2
