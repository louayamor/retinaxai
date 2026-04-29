"""
Domain-level validation logic for the Biomarker Service.

This module contains validation rules that go beyond simple type checking.
"""

from typing import Optional

from loguru import logger


def validate_extraction_request(
    prediction_id: str,
    patient_id: str,
    eye_side: Optional[str],
    model_version: Optional[str],
) -> None:
    """
    Validate the extraction request parameters.
    """
    if not prediction_id:
        raise ValueError("prediction_id is required")
    if not patient_id:
        raise ValueError("patient_id is required")
    if eye_side and eye_side not in ["left", "right", "both"]:
        raise ValueError("eye_side must be 'left', 'right', or 'both'")
    logger.debug(
        "Validation passed for prediction_id={}, patient_id={}, eye_side={}, model_version={}",
        prediction_id,
        patient_id,
        eye_side,
        model_version,
    )
