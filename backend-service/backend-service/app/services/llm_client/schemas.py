from __future__ import annotations
from pydantic import BaseModel


class LLMReportRequest(BaseModel):
    patient: dict
    prediction: dict
    cleaned_summary: str
    raw_ocr_text: str
    report_type: str = "report"
    language: str = "en"
    tone: str = "clinical"
    model_name: str
    model_version: str
    prediction_output: dict
    confidence_score: float


class LLMReportResponse(BaseModel):
    content: str
    summary: str
    model_used: str
