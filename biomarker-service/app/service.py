from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from app.schemas import VascularBiomarkers


class BiomarkerExtractionError(ValueError):
    pass


@dataclass(slots=True)
class VascXAdapter:
    model: Any

    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        if not image_bytes:
            raise BiomarkerExtractionError("empty image payload")
        try:
            return self.model.run(image_bytes)
        except Exception as exc:
            raise BiomarkerExtractionError(f"vascx inference failed: {exc}") from exc


class VascXRegistry:
    _instance: "VascXRegistry | None" = None

    def __new__(cls) -> "VascXRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._adapter: VascXAdapter | None = None
        self._initialized = True

    def load(self) -> VascXAdapter:
        if self._adapter is None:
            from rtnls_inference import VascX

            logger.info("loading vascx model")
            model = VascX()
            self._adapter = VascXAdapter(model=model)
            logger.info("vascx model loaded")
        return self._adapter

    def get(self) -> VascXAdapter:
        return self.load()


def get_vascx_registry() -> VascXRegistry:
    return VascXRegistry()


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [float(v) for v in value if v is not None]
    return None


@dataclass(slots=True)
class BiomarkerService:
    service_name: str = "biomarker-service"
    service_version: str = "0.1.0"

    def __post_init__(self) -> None:
        self._registry = get_vascx_registry()

    def warm(self) -> None:
        self._registry.load()

    def extract_biomarkers(self, image_bytes: bytes) -> VascularBiomarkers:
        logger.info("starting biomarker feature extraction")
        adapter = self._registry.get()
        raw = adapter.predict(image_bytes)
        logger.debug("vascx raw output keys={}", sorted(raw.keys()))

        biomarkers = VascularBiomarkers(
            tortuosity=_coerce_float(raw.get("tortuosity")),
            avr=_coerce_float(raw.get("avr")),
            fractal_dimension=_coerce_float(raw.get("fractal_dimension")),
            vessel_density=_coerce_float(raw.get("vessel_density")),
            bifurcation_count=_coerce_int(raw.get("bifurcation_count")),
            bifurcation_angles=_coerce_list(raw.get("bifurcation_angles")),
            cre=raw.get("cre") if raw.get("cre") is not None else None,
            raw_feature_vector=(
                raw.get("raw_feature_vector")
                if raw.get("raw_feature_vector") is not None
                else None
            ),
        )
        logger.info(
            "computed biomarkers vessel_density={} tortuosity={} avr={} fractal_dimension={}",
            biomarkers.vessel_density,
            biomarkers.tortuosity,
            biomarkers.avr,
            biomarkers.fractal_dimension,
        )
        logger.debug(
            "biomarker payload={} extracted_at={}",
            biomarkers.model_dump(),
            datetime.now(timezone.utc),
        )
        return biomarkers
