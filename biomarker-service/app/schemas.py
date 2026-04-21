from pydantic import BaseModel, Field


class BiomarkerExtractionRequest(BaseModel):
    prediction_id: str = Field(description="Prediction identifier")
    patient_id: str = Field(description="Patient identifier")
    eye_side: str | None = Field(default=None, description="Eye side if applicable")
    model_version: str | None = Field(default=None, description="Prediction model version")


class BiomarkerExtractionResponse(BaseModel):
    prediction_id: str
    status: str
    service_name: str
    service_version: str
    biomarkers: dict
    error: str | None = None
