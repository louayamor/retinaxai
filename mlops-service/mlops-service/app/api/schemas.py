from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
from datetime import datetime
from pydantic import ConfigDict


class ModelStage(str, Enum):
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ModelVersion(BaseModel):
    model_config = ConfigDict(extra="allow")

    version: str
    pipeline: Optional[str] = None
    stage: ModelStage
    model_path: Optional[str] = None
    artifact_path: Optional[str] = None
    hash: Optional[str] = None
    metrics: Optional[dict] = None
    metadata: dict = {}
    created_at: Optional[datetime] = None
    promoted_at: Optional[datetime] = None


class ModelRegisterResponse(BaseModel):
    model: ModelVersion
    message: str
    next_action: str


class ModelPromotionRequest(BaseModel):
    target_stage: ModelStage


class ModelPromotionResponse(BaseModel):
    success: bool
    previous_version: Optional[str]
    new_version: str
    promotion_time: Optional[datetime]
    notes: str


class ModelRollbackRequest(BaseModel):
    version: str


class ModelDetailResponse(BaseModel):
    model: ModelVersion
    is_current_production: bool
    can_promote: bool
    can_rollback: bool
    promotion_history: list[dict]


class ModelListResponse(BaseModel):
    models: list[ModelDetailResponse]
    total: int
    staging_count: int
    production_count: int
    archived_count: int


class CurrentProductionResponse(BaseModel):
    imaging: Optional[ModelVersion]
    clinical: Optional[ModelVersion]
    promoted_at: Optional[str]


class PipelineType(str, Enum):
    both = "both"
    imaging = "imaging"
    clinical = "clinical"


class TrainRequest(BaseModel):
    pipeline: PipelineType = PipelineType.both


class TrainResponse(BaseModel):
    job_id: str
    pipeline: str
    status: str
    message: str


class StatusResponse(BaseModel):
    job_id: Optional[str]
    pipeline: Optional[str]
    status: str
    started_at: Optional[str]
    completed_at: Optional[str]
    error: Optional[str]


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class ImagingMetrics(BaseModel):
    accuracy: Optional[float]
    quadratic_weighted_kappa: Optional[float]
    roc_auc_macro: Optional[float]
    precision_macro: Optional[float]
    recall_macro: Optional[float]
    num_samples: Optional[int]


class ClinicalMetrics(BaseModel):
    accuracy: Optional[float]
    quadratic_weighted_kappa: Optional[float]
    roc_auc_macro: Optional[float]
    precision_macro: Optional[float]
    recall_macro: Optional[float]
    num_samples: Optional[int]


class MetricsResponse(BaseModel):
    imaging: Optional[ImagingMetrics]
    clinical: Optional[ClinicalMetrics]


class ClinicalFeatures(BaseModel):
    patient_age: Optional[float] = None
    patient_gender: Optional[str] = None
    meta_eye: Optional[str] = None
    thickness_center_fovea: Optional[float] = None
    thickness_average_thickness: Optional[float] = None
    thickness_total_volume_mm3: Optional[float] = None
    thickness_inner_superior: Optional[float] = None
    thickness_inner_nasal: Optional[float] = None
    thickness_inner_inferior: Optional[float] = None
    thickness_inner_temporal: Optional[float] = None
    thickness_outer_superior: Optional[float] = None
    thickness_outer_nasal: Optional[float] = None
    thickness_outer_inferior: Optional[float] = None
    thickness_outer_temporal: Optional[float] = None
    clinical_edema: Optional[str] = None
    clinical_erm_status: Optional[str] = None
    meta_image_quality: Optional[str] = None


class MLPredictHttpRequest(BaseModel):
    model_name: str
    model_version: str
    patient_id: str
    patient_age: int
    patient_gender: str
    left_scan_path: Optional[str] = None
    right_scan_path: Optional[str] = None
    left_scan: str
    right_scan: str
    features: dict


class RegionNumeric(BaseModel):
    name: str = Field(description="Anatomical region name")
    intensity: float = Field(description="Activation strength (0-1)")
    area: int = Field(description="Pixel count of region")
    center_x: int = Field(description="Centroid X coordinate")
    center_y: int = Field(description="Centroid Y coordinate")
    saliency_score: float = Field(description="Weighted importance score (0-1)")


class TopHotspot(BaseModel):
    region: str = Field(description="Region name")
    intensity: float = Field(description="Activation strength (0-1)")
    rank: int = Field(description="Rank by intensity (1-5)")


class PredictResponse(BaseModel):
    prediction: dict = Field(
        description="Primary imaging prediction with clinical metadata"
    )
    confidence_score: float = Field(
        description="Confidence score from imaging model (0-1)"
    )
    model_name: str = Field(description="Model used for primary prediction")
    model_version: str = Field(description="Model version")
    gradcam_left: Optional[str] = Field(
        None, description="GradCAM heatmap for left eye (base64 PNG)"
    )
    gradcam_right: Optional[str] = Field(
        None, description="GradCAM heatmap for right eye (base64 PNG)"
    )
    regions_left: Optional[list[RegionNumeric]] = Field(
        None, description="Anatomical regions with numeric values for left eye"
    )
    regions_right: Optional[list[RegionNumeric]] = Field(
        None, description="Anatomical regions with numeric values for right eye"
    )
    top_hotspots_left: Optional[list[TopHotspot]] = Field(
        None, description="Top 5 hotspots ranked by intensity for left eye"
    )
    top_hotspots_right: Optional[list[TopHotspot]] = Field(
        None, description="Top 5 hotspots ranked by intensity for right eye"
    )
    embedding: Optional[list[float]] = Field(
        None, description="Global image embedding from EfficientNet-B3 features"
    )
    shap_explanation: Optional[dict] = Field(
        None, description="SHAP feature explanations for clinical model"
    )
