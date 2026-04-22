from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


BIOMARKER_CONTRACT_VERSION = "1.0"


class BiomarkerExtractionRequest(BaseModel):
    contract_version: str = Field(default=BIOMARKER_CONTRACT_VERSION, description="Request contract version")
    prediction_id: str = Field(description="Prediction identifier")
    patient_id: str = Field(description="Patient identifier")
    eye_side: str | None = Field(default=None, description="Eye side if applicable")
    model_version: str | None = Field(default=None, description="Prediction model version")
    image_name: str | None = Field(default=None, description="Original uploaded image name")


class VascularBiomarkers(BaseModel):
    tortuosity: float | None = Field(default=None, description="Average vessel tortuosity")
    avr: float | None = Field(default=None, description="Artery-to-vein ratio")
    fractal_dimension: float | None = Field(default=None, description="Fractal dimension of the vascular tree")
    vessel_density: float | None = Field(default=None, description="Vessel area ratio")
    bifurcation_count: int | None = Field(default=None, description="Number of vessel bifurcations")
    bifurcation_angles: list[float] = Field(default_factory=list, description="Angles at bifurcation points")
    cre: dict[str, Any] = Field(default_factory=dict, description="Central retinal equivalent metrics")
    raw_feature_vector: list[float] = Field(default_factory=list, description="Flattened feature vector for downstream use")


class BiomarkerExtractionResponse(BaseModel):
    contract_version: str = Field(default=BIOMARKER_CONTRACT_VERSION, description="Response contract version")
    prediction_id: str
    patient_id: str
    eye_side: str | None = None
    model_version: str | None = None
    status: str
    service_name: str
    service_version: str
    extracted_at: datetime | None = None
    biomarkers: VascularBiomarkers
    error: str | None = None
