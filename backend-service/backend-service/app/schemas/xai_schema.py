from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class XAIExplainResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    content: str = ""
    summary: str | None = None
    model_used: str = "unknown"


class XAIGradCAMResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    left_eye_explanation: str = ""
    right_eye_explanation: str = ""
    highlighted_regions: dict[str, Any] = {}
    model_used: str = "unknown"


class XAISeverityResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    content: str = ""
    summary: str | None = None
    risk_level: str = "moderate"
    recommendations: list[str] = []
    model_used: str = "unknown"
