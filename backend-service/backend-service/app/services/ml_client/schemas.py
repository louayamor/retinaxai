from pydantic import BaseModel


class RegionNumeric(BaseModel):
    name: str
    intensity: float
    area: int
    center_x: int
    center_y: int
    saliency_score: float


class TopHotspot(BaseModel):
    region: str
    intensity: float
    rank: int


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
    embedding: list[float] | None = None
    gradcam_left: str | None = None
    gradcam_right: str | None = None
    regions_left: list[RegionNumeric] | None = None
    regions_right: list[RegionNumeric] | None = None
    top_hotspots_left: list[TopHotspot] | None = None
    top_hotspots_right: list[TopHotspot] | None = None

