from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class BiomarkerService:
    service_name: str = "biomarker-service"
    service_version: str = "0.1.0"

    def extract_biomarkers(self, _image_bytes: bytes) -> dict:
        return {
            "tortuosity": None,
            "avr": None,
            "fractal_dimension": None,
            "vessel_density": None,
            "bifurcation_count": None,
            "bifurcation_angles": [],
            "cre": {},
            "raw_feature_vector": [],
        }
