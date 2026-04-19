from abc import ABC, abstractmethod
import time
from typing import Optional

from loguru import logger


class LLMClient(ABC):
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        pass


class GitHubLLMClient(LLMClient):
    def __init__(
        self,
        token: str,
        model: str,
        endpoint: str = "https://models.github.ai/inference",
        timeout_seconds: int = 120,
        max_tokens: int = 4000,
    ):
        self.token = token
        self.model = model
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self._client = None
        self._retry_until: float = 0.0

    def _get_client(self):
        if self._client is None:
            from azure.ai.inference import ChatCompletionsClient
            from azure.core.credentials import AzureKeyCredential
            from azure.core.pipeline.transport import RequestsTransport

            self._client = ChatCompletionsClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.token),
                transport=RequestsTransport(
                    connection_timeout=self.timeout_seconds,
                    read_timeout=self.timeout_seconds,
                ),
            )
        return self._client

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        import asyncio
        from azure.ai.inference.models import SystemMessage, UserMessage
        from azure.core.exceptions import AzureError

        # Bug 2 fix: Check retry timestamp instead of counter
        remaining = self._retry_until - time.monotonic()
        if remaining > 0:
            logger.warning(f"Rate limited, waiting {remaining:.1f}s")
            await asyncio.sleep(remaining)
            self._retry_until = 0.0

        messages = []
        if system_prompt:
            messages.append(SystemMessage(system_prompt))
        messages.append(UserMessage(prompt))

        try:
            response = self._get_client().complete(
                messages=messages,
                temperature=0.3,
                top_p=0.9,
                model=model or self.model,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content
        except AzureError as e:
            # Bug 4 fix: Check status code safely
            status_code = getattr(getattr(e, "response", None), "status_code", None)
            if status_code == 429:
                retry_after = getattr(e, "retry_after", 60)
                logger.warning(f"Rate limited by GitHub API, retry after {retry_after}s")
                self._retry_until = time.monotonic() + retry_after
                raise Exception(f"Rate limited, retry after {retry_after}s")
            logger.error(f"GitHub API error: {e}")
            raise Exception(f"LLM API error: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in GitHub LLM client: {e}")
            raise Exception(f"LLM generation failed: {e}") from e


class OllamaLLMClient(LLMClient):
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout_seconds: int = 120,
        max_tokens: int = 4000,
    ):
        self.model = model
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        if self._client is None:
            import httpx
            from langchain_ollama import ChatOllama

            self._client = ChatOllama(
                model=self.model,
                base_url=self.base_url,
                http_client=httpx.Client(timeout=self.timeout_seconds),
            )
        return self._client

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        try:
            messages = []
            if system_prompt:
                messages.append(("system", system_prompt))
            messages.append(("user", prompt))

            response = self._get_client().invoke(messages)
            return response.content  # type: ignore[return-value]
        except Exception as e:
            logger.error(f"Ollama LLM error: {e}")
            raise Exception(f"Ollama generation failed: {e}") from e


class MockLLMClient(LLMClient):
    def __init__(self, **kwargs) -> None:
        pass

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        return f"[Mock response for: {prompt[:50]}...]"


def get_llm_client(provider: str, **kwargs) -> LLMClient:
    providers = {
        "github": GitHubLLMClient,
        "ollama": OllamaLLMClient,
        "mock": MockLLMClient,
    }

    client_class = providers.get(provider.lower(), MockLLMClient)
    logger.info(f"Initializing LLM client: {provider}")
    return client_class(**kwargs)
