from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any

import httpx
from loguru import logger

from app.api.analytics_schemas import (
    AnalyticsQueryRequest,
    AnalyticsQueryResponse,
    ChartSpec,
    SourceInfo,
)
from app.core.config import settings
from app.llm.client import get_llm_client
from app.llm.fallback import generate_with_fallback
from app.prompts.analytics import ANALYTICS_SYSTEM_PROMPT, ANALYTICS_USER_PROMPT
from app.vectorstore.chroma_store import ChromaStore

_MAX_CONCURRENT_LLM_CALLS = 1
_RETRY_DELAY_SECONDS = 20
_MAX_RETRIES = 1
_CACHE_TTL_SECONDS = 300


class AnalyticsPipeline:
    def __init__(self) -> None:
        self.store = ChromaStore(
            settings.rag_chroma_persist_directory,
            settings.rag_chroma_collection_name,
            settings.rag_embedding_model,
        )
        self._llm_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_LLM_CALLS)
        self._cache: dict[str, tuple[float, AnalyticsQueryResponse]] = {}

        provider = (
            settings.llm_provider.value
            if hasattr(settings.llm_provider, "value")
            else str(settings.llm_provider)
        )

        client_kwargs: dict[str, str | int] = {
            "model": settings.resolved_model,
            "timeout_seconds": 60,
            "max_tokens": min(settings.max_tokens, 1024),
        }
        if provider == "github":
            client_kwargs["token"] = settings.github_token or ""
            client_kwargs["endpoint"] = settings.github_endpoint
        elif provider == "nvidia":
            client_kwargs["api_key"] = settings.nvidia_api_key or ""
            client_kwargs["base_url"] = settings.nvidia_base_url
        elif provider == "ollama":
            client_kwargs["base_url"] = settings.ollama_base_url
        else:
            client_kwargs["token"] = settings.llm_api_key or ""
            client_kwargs["base_url"] = settings.llm_base_url or ""

        self.client = get_llm_client(provider, **client_kwargs)

    async def run(self, request: AnalyticsQueryRequest) -> AnalyticsQueryResponse:
        question = request.question
        cache_key = hashlib.sha256(question.encode()).hexdigest()

        now = time.monotonic()
        if cache_key in self._cache:
            cached_at, cached_response = self._cache[cache_key]
            if now - cached_at < _CACHE_TTL_SECONDS:
                logger.info(f"analytics_cache_hit: {question[:80]}...")
                return cached_response

        logger.info(f"analytics_query: question={question[:120]}...")

        retrieval_start = time.time()
        rag_context, sources = self._retrieve_context(question, top_k=request.top_k)
        retrieval_ms = (time.time() - retrieval_start) * 1000
        logger.info(
            f"analytics_context_retrieved: {len(rag_context)} chars, "
            f"{len(sources)} sources in {retrieval_ms:.0f}ms"
        )

        pg_context = await self._fetch_dashboard_stats()
        if pg_context:
            logger.info(f"analytics_pg_context: {len(pg_context)} chars from backend")

        full_context_parts = []
        if pg_context:
            full_context_parts.append(f"LIVE PATIENT DATA:\n{pg_context}")
        if rag_context.strip():
            full_context_parts.append(f"KNOWLEDGE BASE CONTEXT:\n{rag_context}")
        if not full_context_parts:
            return AnalyticsQueryResponse(
                question=question,
                summary="No data available. Run training and indexing to populate metrics and patient data.",
                sources=[],
            )

        full_context = "\n\n".join(full_context_parts)

        prompt = ANALYTICS_USER_PROMPT.format(question=question, context=full_context)

        last_error: str | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                raw = await self._generate(prompt)
                response = self._parse_response(question, raw, sources)
                self._cache[cache_key] = (time.monotonic(), response)
                return response
            except Exception as e:
                msg = str(e)
                if (
                    "rate" in msg.lower() or "disconnect" in msg.lower()
                ) and attempt < _MAX_RETRIES:
                    logger.warning(
                        f"analytics_retry: attempt {attempt + 1}/{_MAX_RETRIES} "
                        f"after {_RETRY_DELAY_SECONDS}s ({msg[:80]})"
                    )
                    await asyncio.sleep(_RETRY_DELAY_SECONDS)
                    last_error = msg
                    continue
                logger.error(f"analytics_generation_failed: {e}")
                last_error = msg
                break

        return AnalyticsQueryResponse(
            question=question,
            summary="Failed to generate analysis. The analytics engine is currently overloaded. Please try again.",
            sources=sources,
            error=last_error[:500] if last_error else "Unknown error",
        )

    async def _fetch_dashboard_stats(self) -> str | None:
        try:
            backend_url = settings.backend_service_url.rstrip("/")
            url = f"{backend_url}/api/v1/dashboard/stats"
            headers: dict[str, str] = {}
            if settings.backend_api_key:
                headers["x-api-key"] = settings.backend_api_key

            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                data = resp.json()

            essential = {
                "totals": data.get("totals", {}),
                "severity_distribution": data.get("severity_distribution", {}),
                "gender_distribution": data.get("gender_distribution", {}),
                "age_distribution": data.get("age_distribution", {}),
                "recent_activity": data.get("recent_activity", {}),
                "avg_confidence": data.get("avg_confidence"),
            }

            return json.dumps(essential, indent=2, default=str)
        except Exception as e:
            logger.warning(f"analytics_backend_stats_failed: {e}")
            return None

    def _retrieve_context(
        self, question: str, top_k: int = 8
    ) -> tuple[str, list[SourceInfo]]:
        try:
            results = self.store.query(question, top_k=top_k)
        except Exception as e:
            logger.warning(f"analytics_retrieval_failed: {e}")
            return "", []

        if not results:
            return "", []

        snippets: list[str] = []
        sources: list[SourceInfo] = []
        for doc, _score in results:
            text = getattr(doc, "page_content", str(doc)).strip()
            metadata = getattr(doc, "metadata", {}) or {}
            if not text:
                continue
            artifact_id = str(metadata.get("artifact_id", "unknown"))
            snippets.append(f"[source: {artifact_id}]\n{text}")
            sources.append(
                SourceInfo(
                    artifact_id=artifact_id,
                    snippet=text[:100],
                )
            )

        context = "\n\n---\n\n".join(snippets)
        max_chars = 2000
        if len(context) > max_chars:
            context = context[:max_chars]
            logger.warning(f"analytics_context_truncated: {max_chars} chars")

        return context, sources[:3]

    async def _generate(self, prompt: str) -> str:
        async with self._llm_semaphore:
            result = await generate_with_fallback(
                self.client, prompt, system_prompt=ANALYTICS_SYSTEM_PROMPT
            )
            return result.content

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
            logger.warning("analytics_json_parse_failed, using raw text as summary")
            return AnalyticsQueryResponse(
                question=question,
                summary=text[:2000],
                sources=sources,
                error="Failed to parse structured response",
            )

        chart: ChartSpec | None = None
        if data.get("chart"):
            try:
                chart = ChartSpec(**data["chart"])
            except Exception as e:
                logger.warning(f"analytics_chart_parse_failed: {e}")

        response_sources = sources
        raw_sources = data.get("sources", [])
        if isinstance(raw_sources, list) and raw_sources:
            parsed = []
            for s in raw_sources:
                if isinstance(s, dict):
                    parsed.append(
                        SourceInfo(
                            artifact_id=str(s.get("artifact_id", "unknown")),
                            snippet=str(s.get("snippet", "")),
                        )
                    )
            if parsed:
                response_sources = parsed

        return AnalyticsQueryResponse(
            question=question,
            summary=str(data.get("summary", text[:2000])),
            chart=chart,
            sources=response_sources,
        )
