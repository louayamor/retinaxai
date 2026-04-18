from pydantic import BaseModel


class MLPredictRequest(BaseModel):
    model_name: str
    model_version: str
    patient_id: str
    patient_age: int
    patient_gender: str
    left_scan_path: str
    right_scan_path: str
    features: dict


class MLPredictResponse(BaseModel):
    prediction: dict
    confidence_score: float
    model_name: str
    model_version: str
    gradcam_left: str | None = None
    gradcam_right: str | None = None
    regions_left: list[str] | None = None
    regions_right: list[str] | None = None
