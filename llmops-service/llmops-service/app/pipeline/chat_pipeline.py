from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from loguru import logger
from pydantic import BaseModel

from app.api.analytics_schemas import (
    AnalyticsQueryResponse,
    ChartSpec,
    SourceInfo,
)
from app.core.config import settings
from app.llm.client import get_llm_client
from app.prompts.chat import CHAT_SYSTEM_PROMPT, CHAT_USER_PROMPT
from app.vectorstore.chroma_store import ChromaStore


class ChatRequest(BaseModel):
    messages: list[dict[str, str]]
    question: str
    top_k: int = 5


class ChatPipeline:
    def __init__(self) -> None:
        self.store = ChromaStore(
            settings.rag_chroma_persist_directory,
            settings.rag_chroma_collection_name,
            settings.rag_embedding_model,
        )
        self._semaphore = asyncio.Semaphore(1)

        provider = (
            settings.llm_provider.value
            if hasattr(settings.llm_provider, "value")
            else str(settings.llm_provider)
        )
        token = settings.github_token if provider == "github" else settings.llm_api_key
        base_url = (
            settings.github_endpoint if provider == "github" else settings.llm_base_url
        )
        client_kwargs: dict[str, str | int] = {
            "model": settings.llm_model,
            "timeout_seconds": 60,
            "max_tokens": min(settings.max_tokens, 1024),
        }
        if provider == "github":
            client_kwargs["token"] = token if token is not None else ""
            client_kwargs["endpoint"] = base_url if base_url is not None else ""
        elif provider == "ollama":
            client_kwargs["base_url"] = (
                base_url if base_url is not None else settings.ollama_base_url
            )
        else:
            client_kwargs["token"] = token if token is not None else ""
            client_kwargs["base_url"] = base_url if base_url is not None else ""

        self.client = get_llm_client(provider, **client_kwargs)

    async def run(
        self, messages: list[dict], question: str, top_k: int = 5
    ) -> AnalyticsQueryResponse:
        logger.info(f"chat_query: {question[:120]}...")

        rag_context, sources = self._retrieve_context(question, top_k)
        logger.info(f"chat_context: {len(rag_context)} chars, {len(sources)} sources")

        recent = messages[-10:]
        history_text = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in recent)

        prompt = CHAT_USER_PROMPT.format(
            history=history_text,
            context=rag_context
            if rag_context.strip()
            else "No indexed data available.",
            question=question,
        )

        try:
            async with self._semaphore:
                raw = await self.client.generate(
                    prompt, system_prompt=CHAT_SYSTEM_PROMPT
                )
        except Exception as e:
            logger.error(f"chat_generation_failed: {e}")
            return AnalyticsQueryResponse(
                question=question,
                summary="I'm unable to process your question right now. The AI service may be temporarily unavailable.",
                sources=sources,
                error=str(e)[:500],
            )

        return self._parse_response(question, raw, sources)

    def _retrieve_context(
        self, question: str, top_k: int
    ) -> tuple[str, list[SourceInfo]]:
        try:
            results = self.store.query(question, top_k=top_k)
        except Exception as e:
            logger.warning(f"chat_retrieval_failed: {e}")
            return "", []

        if not results:
            return "", []

        snippets: list[str] = []
        sources: list[SourceInfo] = []
        for doc, _score in results:
            text = getattr(doc, "page_content", str(doc)).strip()
            if not text:
                continue
            metadata = getattr(doc, "metadata", {}) or {}
            artifact_id = str(metadata.get("artifact_id", "unknown"))
            snippets.append(f"[source: {artifact_id}]\n{text}")
            sources.append(
                SourceInfo(
                    artifact_id=artifact_id,
                    snippet=text[:200],
                )
            )

        context = "\n\n---\n\n".join(snippets)
        max_chars = 3000
        if len(context) > max_chars:
            context = context[:max_chars]
        return context, sources[:4]

    def _parse_response(
        self, question: str, raw: str, sources: list[SourceInfo]
    ) -> AnalyticsQueryResponse:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return AnalyticsQueryResponse(
                question=question,
                summary=text[:2000],
                sources=sources,
            )

        chart: ChartSpec | None = None
        if data.get("chart"):
            try:
                chart = ChartSpec(**data["chart"])
            except Exception:
                pass

        return AnalyticsQueryResponse(
            question=question,
            summary=str(data.get("summary", text[:2000])),
            chart=chart,
            sources=sources,
        )
