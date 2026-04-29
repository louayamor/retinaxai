"""
Orchestrator for the biomarker extraction workflow.

This module coordinates the flow from raw image bytes to normalized biomarker output.
"""

import asyncio
from typing import Any, Optional

from loguru import logger

from biomarker_service.domain.contracts import (
    BiomarkerExtractionResponse,
    VascularBiomarkers,
)
from biomarker_service.domain.models import BiomarkerResult, BiomarkerFailure
from biomarker_service.infrastructure.preprocessing import decode_and_normalize_image
from biomarker_service.infrastructure.vascx_adapter import VascXAdapter


class BiomarkerOrchestrator:
    """
    Orchestrates the biomarker extraction workflow.
    """

    def __init__(self):
        self.vascx_adapter = VascXAdapter()

    async def extract_biomarkers(
        self,
        prediction_id: str,
        patient_id: str,
        eye_side: Optional[str],
        model_version: Optional[str],
        image_bytes: bytes,
    ) -> BiomarkerResult:
        """
        Execute the full biomarker extraction workflow.
        """
        logger.info(
            "Starting biomarker extraction for prediction_id={}",
            prediction_id,
        )

        # 1. Preprocess the image
        normalized_image = await decode_and_normalize_image(image_bytes)

        # 2. Run inference
        raw_output = await self.vascx_adapter.predict(normalized_image)

        # 3. Normalize the output
        biomarkers = VascularBiomarkers(
            tortuosity=raw_output.get("tortuosity"),
            avr=raw_output.get("avr"),
            fractal_dimension=raw_output.get("fractal_dimension"),
            vessel_density=raw_output.get("vessel_density"),
            bifurcation_count=raw_output.get("bifurcation_count"),
            bifurcation_angles=raw_output.get("bifurcation_angles"),
            cre=raw_output.get("cre"),
            raw_feature_vector=raw_output.get("raw_feature_vector"),
        )

        # 4. Construct result
        result = BiomarkerResult(
            prediction_id=prediction_id,
            patient_id=patient_id,
            eye_side=eye_side,
            model_version=model_version,
            biomarkers=biomarkers.model_dump(),
            service_name="biomarker-service",
            service_version="0.1.0",
            contract_version="1.0",
            extracted_at="",
        )

        logger.info(
            "Biomarker extraction completed successfully for prediction_id={}",
            prediction_id,
        )

        return result

    async def handle_failure(
        self,
        prediction_id: str,
        patient_id: str,
        eye_side: Optional[str],
        model_version: Optional[str],
        error_code: str,
        error_message: str,
    ) -> BiomarkerFailure:
        """
        Handle a failed biomarker extraction.
        """
        logger.error(
            "Biomarker extraction failed for prediction_id={} with error_code={}",
            prediction_id,
            error_code,
        )

        return BiomarkerFailure(
            prediction_id=prediction_id,
            patient_id=patient_id,
            eye_side=eye_side,
            model_version=model_version,
            error_code=error_code,
            error_message=error_message,
            service_name="biomarker-service",
            service_version="0.1.0",
            contract_version="1.0",
        )
