from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AnalyticsQueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)


class ChartSpec(BaseModel):
    type: Literal["bar", "pie", "line", "area", "radar", "table"]
    title: str
    description: str = ""
    data: list[dict]
    config: dict = Field(default_factory=dict)


class SourceInfo(BaseModel):
    artifact_id: str
    snippet: str


class AnalyticsQueryResponse(BaseModel):
    question: str
    summary: str
    chart: ChartSpec | None = None
    sources: list[SourceInfo] = Field(default_factory=list)
    error: str | None = None
