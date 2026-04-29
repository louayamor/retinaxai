"""
API-facing Pydantic models for the Biomarker Service.

These models define the request and response contracts for the external API.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class BiomarkerExtractionRequest(BaseModel):
    """
    Request schema for the /biomarkers/extract endpoint.
    """

    contract_version: str = Field(
        default="1.0",
        description="Request contract version",
    )
    prediction_id: str = Field(description="Prediction identifier")
    patient_id: str = Field(description="Patient identifier")
    eye_side: Optional[str] = Field(default=None, description="Eye side if applicable")
    model_version: Optional[str] = Field(
        default=None, description="Prediction model version"
    )
    image_name: Optional[str] = Field(
        default=None, description="Original uploaded image name"
    )


class VascularBiomarkers(BaseModel):
    """
    Response schema for the biomarker extraction results.
    """

    tortuosity: Optional[float] = Field(
        default=None, description="Average vessel tortuosity"
    )
    avr: Optional[float] = Field(default=None, description="Artery-to-vein ratio")
    fractal_dimension: Optional[float] = Field(
        default=None, description="Fractal dimension of the vascular tree"
    )
    vessel_density: Optional[float] = Field(
        default=None, description="Vessel area ratio"
    )
    bifurcation_count: Optional[int] = Field(
        default=None, description="Number of vessel bifurcations"
    )
    bifurcation_angles: Optional[list[float]] = Field(
        default=None, description="Angles at bifurcation points"
    )
    cre: Optional[dict[str, Any]] = Field(
        default=None, description="Central retinal equivalent metrics"
    )
    raw_feature_vector: Optional[list[float]] = Field(
        default=None, description="Flattened feature vector for downstream use"
    )


class BiomarkerExtractionResponse(BaseModel):
    """
    Response schema for the /biomarkers/extract endpoint.
    """

    contract_version: str = Field(
        default="1.0",
        description="Response contract version",
    )
    prediction_id: str
    patient_id: str
    eye_side: Optional[str] = None
    model_version: Optional[str] = None
    status: str
    service_name: str
    service_version: str
    extracted_at: Optional[datetime] = None
    biomarkers: VascularBiomarkers
    error: Optional[str] = None
