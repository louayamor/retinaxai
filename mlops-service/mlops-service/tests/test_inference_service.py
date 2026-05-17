from __future__ import annotations

import asyncio
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest
from fastapi import HTTPException

from app.api.schemas import MLPredictHttpRequest
import app.inference.inference_service as module
from app.api.routes import predict as predict_route


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


def test_get_embedding_returns_1536_dim_vector(monkeypatch):
    service = module.InferenceService.__new__(module.InferenceService)
    service.settings = SimpleNamespace()
    service.params = SimpleNamespace()

    class DummyModel:
        def __init__(self):
            self.module = self

        def forward_features(self, image_tensor):
            return torch.ones((1, 1536, 7, 7), dtype=torch.float32)

        def global_pool(self, features):
            return torch.ones((1, 1536, 1, 1), dtype=torch.float32)

    service._load_imaging_model = lambda: DummyModel()

    embedding = module.InferenceService.get_embedding(
        service, torch.ones((1, 3, 224, 224), dtype=torch.float32)
    )

    assert len(embedding) == 1536
    assert all(isinstance(value, float) for value in embedding)


def test_predict_route_emits_failure_event(monkeypatch):
    emitted = {}

    class DummyService:
        def predict_imaging_with_gradcam(self, _image_bytes):
            raise RuntimeError("boom")

    async def fake_log(*_args, **_kwargs):
        return None

    async def fake_event(**kwargs):
        emitted.update(kwargs)

    monkeypatch.setattr(predict_route, "get_inference_service", lambda: DummyService())
    monkeypatch.setattr(predict_route, "send_prediction_log", fake_log)
    monkeypatch.setattr(predict_route, "send_prediction_event", fake_event)
    monkeypatch.setattr(
        predict_route, "_decode_base64_image", lambda value: b"fake-bytes"
    )
    monkeypatch.setattr(predict_route, "_validate_image_bytes", lambda value: None)

    request = MLPredictHttpRequest(
        model_name="efficientnet_b3",
        model_version="1.0.0",
        patient_id="patient-1",
        patient_age=70,
        patient_gender="M",
        left_scan="ZmFrZQ==",
        right_scan="ZmFrZQ==",
        features={},
    )

    async def run():
        with pytest.raises(HTTPException):
            await predict_route.predict(request, service=DummyService())

    asyncio.run(run())

    assert emitted["error"] == "boom"


def test_predict_route_includes_embedding(monkeypatch):
    class DummyService:
        def predict_imaging_with_gradcam(self, image_bytes):
            return {
                "predicted_grade": 2,
                "predicted_label": "Moderate",
                "severity": "moderate",
                "confidence": 0.91,
                "probabilities": {"Moderate": 0.91},
                "gradcam_heatmap": "heatmap",
                "regions": [],
                "top_hotspots": [],
                "embedding": [0.1, 0.2, 0.3],
            }

        def predict_clinical(self, features):
            return {}

    async def fake_log(*_args, **_kwargs):
        return None

    async def fake_event(*_args, **_kwargs):
        return None

    monkeypatch.setattr(predict_route, "send_prediction_log", fake_log)
    monkeypatch.setattr(predict_route, "send_prediction_event", fake_event)
    monkeypatch.setattr(
        predict_route, "_decode_base64_image", lambda value: b"fake-bytes"
    )
    monkeypatch.setattr(predict_route, "_validate_image_bytes", lambda value: None)

    request = MLPredictHttpRequest(
        model_name="efficientnet_b3",
        model_version="1.0.0",
        patient_id="patient-1",
        patient_age=70,
        patient_gender="M",
        left_scan="ZmFrZQ==",
        right_scan="ZmFrZQ==",
        features={},
    )

    async def run():
        response = await predict_route.predict(request, service=DummyService())
        assert response.embedding == [0.1, 0.2, 0.3]

    asyncio.run(run())
