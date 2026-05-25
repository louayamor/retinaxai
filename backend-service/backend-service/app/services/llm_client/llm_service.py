from __future__ import annotations
import json

import httpx

from app.core.config import settings
from app.core.exceptions import ServiceUnavailableException
from app.services.llm_client.schemas import LLMReportRequest, LLMReportResponse

OLLAMA_GENERATE_ENDPOINT = "/api/generate"


class LLMServiceClient:
    def __init__(self):
        self.base_url = settings.LLM_SERVICE_URL
        self.timeout = settings.LLM_SERVICE_TIMEOUT
        self.model = settings.LLM_MODEL

    async def generate_report(self, request: LLMReportRequest) -> LLMReportResponse:
        payload = {
            "model": self.model,
            "patient": request.patient,
            "prediction": request.prediction,
            "cleaned_summary": request.cleaned_summary,
            "raw_ocr_text": request.raw_ocr_text,
            "report_type": request.report_type,
            "language": request.language,
            "tone": request.tone,
            "stream": False,
            "format": "json",
        }

        headers = {
            "X-API-Key": settings.LLM_SERVICE_API_KEY,
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}{OLLAMA_GENERATE_ENDPOINT}",
                    json=payload,
                    headers=headers,
                )
                if response.status_code == 401:
                    raise ServiceUnavailableException("llm-service")
                response.raise_for_status()
                try:
                    data: dict = response.json()
                    parsed = json.loads(data.get("response", "{}"))
                except (json.JSONDecodeError, KeyError, Exception):
                    data = {}
                    response_text = (
                        data.get("response", "") if isinstance(data, dict) else ""
                    )
                    parsed = {
                        "content": response_text,
                        "summary": "",
                        "model_used": self.model,
                    }
                return LLMReportResponse(
                    content=parsed.get("content", ""),
                    summary=parsed.get("summary", ""),
                    model_used=parsed.get("model_used", self.model),
                )
        except httpx.TimeoutException:
            raise ServiceUnavailableException("llm-service")
        except httpx.ConnectError:
            raise ServiceUnavailableException("llm-service")
        except ServiceUnavailableException:
            raise
        except (KeyError, ValueError):
            raise ServiceUnavailableException("llm-service")


llm_client = LLMServiceClient()
