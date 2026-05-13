from __future__ import annotations

from typing import Any

import httpx

from app.core.config import settings
from app.core.exceptions import ServiceUnavailableException


class ChatServiceClient:
    def __init__(self) -> None:
        self.base_url = settings.LLM_SERVICE_URL
        self.timeout = 120.0
        self.api_key = settings.LLM_SERVICE_API_KEY

    async def send_chat(
        self, messages: list[dict[str, str]], question: str
    ) -> dict[str, Any]:
        headers: dict[str, str] = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "messages": messages,
            "question": question,
            "top_k": 5,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                return response.json()
        except httpx.TimeoutException:
            raise ServiceUnavailableException("llm-service")
        except httpx.ConnectError:
            raise ServiceUnavailableException("llm-service")
        except ServiceUnavailableException:
            raise
        except Exception:
            raise ServiceUnavailableException("llm-service")


chat_client = ChatServiceClient()
