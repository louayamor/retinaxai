from __future__ import annotations

from datetime import datetime, timezone

from app.schemas.prediction_schema import PredictionRead


def test_prediction_read_includes_biomarker_fields() -> None:
    payload = {
        "id": "d3fc6d95-b2fa-4cb0-a41f-f78dbce57d89",
        "patient_id": "d3fc6d95-b2fa-4cb0-a41f-f78dbce57d88",
        "mri_scan_id": "d3fc6d95-b2fa-4cb0-a41f-f78dbce57d87",
        "requested_by": "d3fc6d95-b2fa-4cb0-a41f-f78dbce57d86",
        "model_name": "efficientnet_b4",
        "model_version": "v1.0.0",
        "input_payload": {},
        "output_payload": {},
        "confidence_score": 0.9,
        "status": "SUCCESS",
        "biomarker_status": "COMPLETED",
        "biomarker_error_code": None,
        "biomarker_error_message": None,
        "error_message": None,
        "created_at": datetime.now(timezone.utc),
    }

    model = PredictionRead.model_validate(payload)

    assert model.biomarker_status == "COMPLETED"
    assert model.biomarker_error_message is None
