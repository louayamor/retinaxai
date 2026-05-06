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

from app.api.schemas import ClinicalFeatures
from app.config.settings import Settings
from app.services.inference.gradcam_service import GradCAMService
from app.utils.common import load_json, read_yaml
from app.constants import PARAMS_FILE_PATH, SCHEMA_FILE_PATH
from app.services.monitoring.prometheus_metrics import (
    INFERENCE_LATENCY,
    INFERENCE_OOM_KILLS,
    GPU_MEMORY_USED_BYTES,
    GPU_UTILIZATION_PERCENT,
)
from app.services.registry.model_registry import ModelRegistryService
from app.services.registry.model_registry import (
    ModelNotFoundError as ModelRegistryNotFoundError,
)
from app.services.platform.feature_store import get_feature_store


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
        self._clinical_model = None
        self._feature_meta = None
        self._clinical_encoders = None
        self._clinical_numeric_medians = None

        # Add model registry service for loading models
        self._registry_service = ModelRegistryService(
            self.settings.artifacts_root / "model_registry"
        )

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
        return (
            self.settings.imaging_model_path
            if pipeline == "imaging"
            else self.settings.clinical_model_path
        )

    def _move_to_cpu(self) -> None:
        self.device = torch.device("cpu")
        if self._imaging_model is not None:
            self._imaging_model.to(self.device)
        if self._clinical_model is not None and hasattr(self._clinical_model, "to"):
            self._clinical_model.to(self.device)
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

        p = self.params.dl_training
        logger.info(
            f"[IMAGING MODEL] creating efficientnet_b3 num_classes={p.num_classes} drop={p.dropout}"
        )
        model = timm.create_model(
            "efficientnet_b3",
            pretrained=False,
            num_classes=p.num_classes,
            drop_rate=p.dropout,
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

    def _load_clinical_model(self):
        if self._clinical_model is not None:
            return self._clinical_model, self._feature_meta
        logger.info(
            f"[CLINICAL MODEL] model_path={self.settings.clinical_model_path} exists={self.settings.clinical_model_path.exists()}"
        )
        logger.info(
            f"[CLINICAL MODEL] feature_meta_path={self.settings.clinical_feature_importance_path} exists={self.settings.clinical_feature_importance_path.exists()}"
        )

        if not self.settings.clinical_model_path.exists():
            raise FileNotFoundError(
                f"clinical model not found: {self.settings.clinical_model_path}"
            )
        if not self.settings.clinical_feature_importance_path.exists():
            raise FileNotFoundError(
                f"clinical feature metadata not found: {self.settings.clinical_feature_importance_path}"
            )

        if self.settings.clinical_model_path.stat().st_size == 0:
            raise ValueError(
                f"clinical model file is empty: {self.settings.clinical_model_path}"
            )

        logger.info("[CLINICAL MODEL] loading pickle model")
        try:
            model_path = self._get_current_production_model_path("clinical")
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load clinical model: {e}") from e

        try:
            feature_meta = load_json(self.settings.clinical_feature_importance_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load clinical feature metadata: {e}") from e

        self._clinical_model = model
        self._feature_meta = feature_meta
        self._clinical_encoders = feature_meta.get("categorical_encoders", {}) or {}
        self._clinical_numeric_medians = feature_meta.get("numeric_medians", {}) or {}
        logger.info("[CLINICAL MODEL] loaded successfully")
        return model, feature_meta

    def _build_transform(self):
        norm = self.params.augmentation.normalize
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm.mean, std=norm.std),
            ]
        )

    def get_embedding(self, image_tensor: torch.Tensor) -> list[float]:
        """Extract the 1536-dim EfficientNet-B3 embedding before classification."""
        model = self._load_imaging_model()

        if hasattr(model, "module"):
            model = model.module

        with torch.inference_mode():
            features = model.forward_features(image_tensor)
            embedding = model.global_pool(features)
            if embedding.ndim > 2:
                embedding = embedding.flatten(start_dim=1)

        return embedding.squeeze(0).detach().cpu().numpy().astype(float).tolist()

    def predict_imaging(self, image_bytes: bytes) -> dict:
        start = time.time()
        model = self._load_imaging_model()

        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to open image: {e}") from e

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
                INFERENCE_OOM_KILLS.labels(model="efficientnet_b3").inc()
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

        INFERENCE_LATENCY.labels(model="efficientnet_b3").observe(time.time() - start)
        _emit_gpu_metrics()

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        del tensor, outputs

        threshold = 0.0
        try:
            threshold = float(
                self.params.get("inference", {}).get("confidence_threshold", 0.0)
            )
        except (AttributeError, TypeError):
            pass
        if confidence < threshold:
            return {
                "predicted_grade": -1,
                "predicted_label": "Uncertain",
                "severity": "unknown",
                "confidence": round(confidence, 4),
                "uncertain": True,
                "probabilities": {DR_CLASSES[i]: float(p) for i, p in enumerate(probs)},
            }

        return {
            "predicted_grade": pred_class,
            "predicted_label": DR_CLASSES[pred_class],
            "severity": DR_SEVERITY.get(pred_class, "unknown"),
            "confidence": round(confidence, 4),
            "probabilities": {DR_CLASSES[i]: float(p) for i, p in enumerate(probs)},
        }

    def predict_clinical(self, features: ClinicalFeatures) -> dict:
        start = time.time()
        model, feature_meta = self._load_clinical_model()
        feature_meta = feature_meta or {}
        label_offset = feature_meta.get("label_offset", 0)
        feature_cols = feature_meta.get("feature_cols", [])
        if not feature_cols:
            raise ValueError("clinical feature metadata is missing feature_cols")
        categorical_encoders = self._clinical_encoders or {}
        numeric_medians = self._clinical_numeric_medians or {}

        feature_dict = features.model_dump()

        store = get_feature_store()
        cache_key = f"clinical:{hash(frozenset(feature_dict.items()))}"

        if store.exists(cache_key):
            cached = store.get(cache_key)
            logger.info(f"Clinical prediction cache hit: {cache_key}")
            return cached

        row = {}
        for col in feature_cols:
            val = feature_dict.get(col)
            if col in categorical_encoders:
                classes = categorical_encoders[col]
                normalized = "unknown" if val is None else str(val)
                if normalized not in classes:
                    normalized = "unknown" if "unknown" in classes else classes[0]
                row[col] = classes.index(normalized)
            else:
                if val is None:
                    row[col] = float(numeric_medians.get(col, 0.0))
                else:
                    row[col] = float(val)

        X = pd.DataFrame([row])[feature_cols].values
        pred_0indexed = int(model.predict(X)[0])
        pred_original = pred_0indexed + label_offset
        probs = model.predict_proba(X)[0]

        risk_labels = {
            0: "No DR",
            1: "Mild NPDR",
            2: "Moderate NPDR",
            3: "Severe NPDR",
            4: "Proliferative DR",
        }

        INFERENCE_LATENCY.labels(model="xgboost_clinical").observe(time.time() - start)

        result = {
            "predicted_grade": pred_original,
            "predicted_label": risk_labels.get(pred_original, "Unknown"),
            "severity": DR_SEVERITY.get(pred_original, "unknown"),
            "risk_score": round(float(max(probs)), 4),
            "probabilities": {
                risk_labels.get(i + label_offset, str(i + label_offset)): float(p)
                for i, p in enumerate(probs)
            },
        }

        store.set(cache_key, result, ttl_seconds=3600)

        return result

    def predict_imaging_with_gradcam(self, image_bytes: bytes) -> dict:
        start = time.time()
        model = self._load_imaging_model()

        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to open image: {e}") from e

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
                INFERENCE_OOM_KILLS.labels(model="efficientnet_b3").inc()
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

        INFERENCE_LATENCY.labels(model="efficientnet_b3").observe(time.time() - start)
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
        }
