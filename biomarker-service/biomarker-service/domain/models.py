"""
Internal domain models and value objects for the Biomarker Service.

These models represent the business domain and are used internally by the application layer.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BiomarkerResult:
    """
    Internal representation of a successful biomarker extraction result.
    """

    prediction_id: str
    patient_id: str
    eye_side: Optional[str]
    model_version: Optional[str]
    biomarkers: dict[str, any]
    service_name: str
    service_version: str
    contract_version: str
    extracted_at: str


@dataclass
class BiomarkerFailure:
    """
    Internal representation of a failed biomarker extraction.
    """

    prediction_id: str
    patient_id: str
    eye_side: Optional[str]
    model_version: Optional[str]
    error_code: str
    error_message: str
    service_name: str
    service_version: str
    contract_version: str
