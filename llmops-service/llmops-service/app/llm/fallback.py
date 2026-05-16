from __future__ import annotations

import asyncio
from typing import Optional

from loguru import logger

from app.core.config import settings
from app.llm.client import LLMClient, get_llm_client


class InferenceResult:
    def __init__(
        self, content: str, model_name: str, provider: str, switched: str = ""
    ):
        self.content = content
        self.model_name = model_name
        self.provider = provider
        self.switched = switched


_GITHUB_FALLBACK_CLIENT: LLMClient | None = None
_NVIDIA_FALLBACK_CLIENT: LLMClient | None = None
_OLLAMA_FALLBACK_CLIENT: LLMClient | None = None


def _get_github_fallback() -> LLMClient:
    global _GITHUB_FALLBACK_CLIENT
    if _GITHUB_FALLBACK_CLIENT is None:
        _GITHUB_FALLBACK_CLIENT = get_llm_client(
            "github",
            token=settings.github_token or "",
            model="gpt-4o",
            endpoint=settings.github_endpoint,
            timeout_seconds=45,
            max_tokens=min(settings.max_tokens, 1024),
        )
    return _GITHUB_FALLBACK_CLIENT


def _get_nvidia_fallback() -> LLMClient:
    global _NVIDIA_FALLBACK_CLIENT
    if _NVIDIA_FALLBACK_CLIENT is None:
        _NVIDIA_FALLBACK_CLIENT = get_llm_client(
            "nvidia",
            api_key=settings.nvidia_api_key or "",
            model="meta/llama-3.1-8b-instruct",
            base_url=settings.nvidia_base_url,
            timeout_seconds=45,
            max_tokens=min(settings.max_tokens, 1024),
        )
    return _NVIDIA_FALLBACK_CLIENT


def _get_ollama_fallback() -> LLMClient:
    global _OLLAMA_FALLBACK_CLIENT
    if _OLLAMA_FALLBACK_CLIENT is None:
        _OLLAMA_FALLBACK_CLIENT = get_llm_client(
            "ollama",
            model=settings.ollama_fallback_model,
            base_url=settings.ollama_base_url,
            timeout_seconds=90,
            max_tokens=min(settings.max_tokens, 1024),
        )
    return _OLLAMA_FALLBACK_CLIENT


async def generate_with_fallback(
    primary_client: LLMClient,
    prompt: str,
    system_prompt: Optional[str] = None,
    *,
    primary_provider: str = settings.llm_provider.value,
    primary_model: str = settings.resolved_model,
) -> InferenceResult:
    providers_to_try = [
        ("github", "gpt-4o", "GitHub"),
        ("nvidia", "meta/llama-3.1-8b-instruct", "NVIDIA NIM"),
        ("ollama", settings.ollama_fallback_model, "local Ollama"),
    ]

    start_idx = 0
    for i, (p, _, _) in enumerate(providers_to_try):
        if p == primary_provider:
            start_idx = i
            break

    attempts: list[str] = []

    for i in range(start_idx, len(providers_to_try)):
        provider_key, model, display = providers_to_try[i]
        client = primary_client if i == start_idx else None

        try:
            if i == start_idx:
                raw = await primary_client.generate(prompt, system_prompt=system_prompt)
            else:
                if provider_key == "github":
                    client = _get_github_fallback()
                elif provider_key == "nvidia":
                    client = _get_nvidia_fallback()
                elif provider_key == "ollama":
                    client = _get_ollama_fallback()

                if client is None:
                    continue

                logger.info(f"fallback: trying {display}/{model}")
                raw = await client.generate(prompt, system_prompt=system_prompt)

            switch_note = ""
            if i > start_idx:
                switch_note = f"[Switched to {display} ({model})]\n\n"
            return InferenceResult(
                content=raw,
                model_name=model,
                provider=display,
                switched=switch_note,
            )
        except Exception as e:
            attempts.append(f"{display}: {str(e)[:80]}")
            logger.warning(f"fallback_failed: {display} — {e}")
            if i < len(providers_to_try) - 1:
                await asyncio.sleep(1)

    raise Exception(
        "Sorry, inference unavailable. "
        + "All providers failed: "
        + "; ".join(attempts)
    )
