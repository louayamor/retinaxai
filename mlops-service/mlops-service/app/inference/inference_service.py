from __future__ import annotations

import io
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from loguru import logger
from PIL import Image
from torchvision import transforms


from app.config.settings import Settings
from app.training.preprocessing import preprocess_fundus_image
from app.inference.gradcam_service import GradCAMService
from app.inference.fundus_classifier import FundusClassifierService
from app.utils.common import load_json, read_yaml
from app.constants import PARAMS_FILE_PATH, SCHEMA_FILE_PATH
from app.monitoring.prometheus_metrics import (
    INFERENCE_LATENCY,
    INFERENCE_OOM_KILLS,
    GPU_MEMORY_USED_BYTES,
    GPU_UTILIZATION_PERCENT,
)
from app.registry.model_registry import ModelRegistryService
from app.registry.model_registry import (
    ModelNotFoundError as ModelRegistryNotFoundError,
)
from app.platform.feature_store import get_feature_store


DR_CLASSES = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}
DR_SEVERITY = {0: "none", 1: "low", 2: "moderate", 3: "high", 4: "critical"}


def _emit_gpu_metrics() -> None:
    """Emit GPU memory and utilization metrics if CUDA is available."""
    if not torch.cuda.is_available():
        return
    try:
        import pynvml  # type: ignore[import-untyped]

        pynvml.nvmlInit()
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            GPU_MEMORY_USED_BYTES.labels(device=str(i)).set(mem.used)
            GPU_UTILIZATION_PERCENT.labels(device=str(i)).set(util.gpu)
    except Exception:
        pass  # nvidia-smi not available — skip


class InferenceService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.params = read_yaml(PARAMS_FILE_PATH)
        self.schema = read_yaml(SCHEMA_FILE_PATH)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info(f"[INFERENCE] Using CUDA: {torch.cuda.get_device_name(0)}")
            logger.info(
                f"[INFERENCE] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            )
        else:
            self.device = torch.device("cpu")
            logger.info("[INFERENCE] CUDA not available, using CPU")

        self._imaging_model = None
        self._feature_meta = None
        self._fundus_classifier = None

        # Add model registry service for loading models
        self._registry_service = ModelRegistryService(
            self.settings.artifacts_root / "model_registry"
        )

        global_cfg = self.params.get("global", {}) or {}
        training_cfg = self.params.get("training", {}) or {}

        self._global_num_classes = int(global_cfg.get("num_classes", 5))
        self._global_image_size = int(global_cfg.get("image_size", 384))

        phase1_cfg = training_cfg.get("phase1", {}) or {}
        self._training_dropout = float(phase1_cfg.get("dropout", 0.5))

        inference_cfg = self.params.get("inference", {}) or {}
        self._confidence_threshold = float(
            inference_cfg.get("confidence_threshold", 0.0)
        )
        self._apply_fda_inverse = inference_cfg.get("apply_fda_inverse", False)

    def _get_current_production_model_path(self, pipeline: str) -> Optional[Path]:
        """Get current production model path from registry, or fall back to settings paths."""
        try:
            current_model = self._registry_service.get_current_production(pipeline)
            if current_model:
                model_path = self.settings.artifacts_root / current_model.artifact_path
                if model_path.exists():
                    logger.info(
                        f"Using registry model for {pipeline}: {current_model.version}"
                    )
                    return model_path
                else:
                    logger.warning(f"Registry model path doesn't exist: {model_path}")
        except ModelRegistryNotFoundError:
            logger.info(
                f"No production model found in registry for {pipeline}, using default paths"
            )
        except Exception as e:
            logger.warning(f"Failed to load from registry for {pipeline}: {e}")

        # Fall back to settings paths
        return self.settings.imaging_model_path

    def _move_to_cpu(self) -> None:
        self.device = torch.device("cpu")
        if self._imaging_model is not None:
            self._imaging_model.to(self.device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_imaging_model(self) -> nn.Module:
        if self._imaging_model is not None:
            return self._imaging_model
        logger.info(
            f"[IMAGING MODEL] path={self.settings.imaging_model_path} exists={self.settings.imaging_model_path.exists()}"
        )
        if not self.settings.imaging_model_path.exists():
            raise FileNotFoundError(
                f"imaging model not found: {self.settings.imaging_model_path}"
            )

        if self.settings.imaging_model_path.stat().st_size == 0:
            raise ValueError(
                f"imaging model file is empty: {self.settings.imaging_model_path}"
            )

        model_name = self.params.get("mlflow", {}).get(
            "imaging_run_name", "efficientnet_b3"
        )
        logger.info(
            f"[IMAGING MODEL] creating {model_name} num_classes={self._global_num_classes} drop={self._training_dropout}"
        )
        model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=self._global_num_classes,
            drop_rate=self._training_dropout,
        )
        try:
            model_path = self._get_current_production_model_path("imaging")
            logger.info(f"[IMAGING MODEL] loading state dict from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        except Exception as e:
            if self.device.type == "cuda" and "out of memory" in str(e).lower():
                logger.warning(
                    "[IMAGING MODEL] CUDA OOM while loading; retrying on CPU"
                )
                self._move_to_cpu()
                model = self._load_imaging_model()
                return model
            raise RuntimeError(f"Failed to load imaging model state dict: {e}") from e

        model.to(self.device)
        model.eval()
        self._imaging_model = model
        logger.info("[IMAGING MODEL] loaded successfully")
        return model

    def _build_transform(self) -> transforms.Compose:
        aug = self.params.get("augmentation", {}) or {}
        norm = aug.get("normalize", {}) or {}
        return transforms.Compose([
            transforms.Lambda(
                lambda img: preprocess_fundus_image(
                    img, image_size=self._global_image_size
                )
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=norm.get("mean", [0.485, 0.456, 0.406]),
                std=norm.get("std", [0.229, 0.224, 0.225]),
            ),
        ])

    def _validate_fundus(self, image_bytes: bytes, eye_side: str) -> float:
        if self._fundus_classifier is None:
            fc_cfg = self.params.get("fundus_classifier", {}) or {}
            fc_path = self.settings.artifacts_root / "fundus_classifier.pth"
            if not fc_path.exists():
                logger.warning(f"[FUNDUS] classifier model not found: {fc_path} — skipping validation")
                return 1.0
            self._fundus_classifier = FundusClassifierService(
                model_path=fc_path,
                model_name=fc_cfg.get("model_name", "mobilenetv3_small_100"),
                image_size=self._global_image_size,
                device=self.device,
                threshold=float(fc_cfg.get("threshold", 0.3)),
            )
        _, score, msg = self._fundus_classifier.is_fundus(image_bytes)
        logger.info(f"[FUNDUS] {eye_side} eye: {msg}")
        return score

    def get_embedding(self, tensor: torch.Tensor) -> list[float]:
        model = self._load_imaging_model()
        with torch.inference_mode():
            feats = model.forward_features(tensor)
            pooled = model.global_pool(feats)
            embedding = pooled.flatten(1).cpu().numpy()[0].tolist()
        return embedding

    def predict_imaging_with_gradcam(
        self, image_bytes: bytes, eye_side: str = "unknown"
    ) -> dict:
        start = time.time()

        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            original_size = img.size
        except Exception as e:
            raise ValueError(f"Failed to open image: {e}") from e

        logger.info(
            f"[IMAGING] {eye_side} eye: original_size={original_size} "
            f"→ resize={self._global_image_size}x{self._global_image_size}"
        )

        fundus_score = self._validate_fundus(image_bytes, eye_side)

        model = self._load_imaging_model()

        tf: nn.Module = self._build_transform()  # type: ignore[assignment]
        tensor = tf(img).unsqueeze(0).to(self.device)  # type: ignore[operator]

        try:
            with torch.inference_mode():
                with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                    outputs = model(tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        except RuntimeError as e:
            if self.device.type == "cuda" and "out of memory" in str(e).lower():
                logger.warning("[IMAGING] CUDA OOM during inference; retrying on CPU")
                INFERENCE_OOM_KILLS.labels(model="imaging").inc()
                self._move_to_cpu()
                model = self._load_imaging_model()
                tensor = tensor.to(self.device)
                with torch.inference_mode():
                    outputs = model(tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            else:
                raise

        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        embedding = self.get_embedding(tensor)

        gradcam_service = GradCAMService(model)
        gradcam_base64, regions, top_hotspots = (
            gradcam_service.generate_with_regions_numeric(
                image_bytes, tensor, pred_class
            )
        )

        low_confidence = confidence < self._confidence_threshold
        logger.info(
            f"[IMAGING] {eye_side} eye: fundus={fundus_score:.3f} -> "
            f"DR grade={pred_class} ({DR_CLASSES[pred_class]}) "
            f"conf={confidence:.4f} "
            f"{'LOW_CONFIDENCE' if low_confidence else ''}"
        )

        INFERENCE_LATENCY.labels(model="imaging").observe(time.time() - start)
        _emit_gpu_metrics()

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        del tensor, outputs

        return {
            "predicted_grade": pred_class,
            "predicted_label": DR_CLASSES[pred_class],
            "severity": DR_SEVERITY.get(pred_class, "unknown"),
            "confidence": round(confidence, 4),
            "probabilities": {DR_CLASSES[i]: float(p) for i, p in enumerate(probs)},
            "embedding": embedding,
            "gradcam_heatmap": gradcam_base64,
            "regions": regions,
            "top_hotspots": top_hotspots,
            "fundus_score": fundus_score,
            "confidence_threshold": self._confidence_threshold,
            "low_confidence": low_confidence,
        }
