from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Awaitable, Callable

from loguru import logger
from pydantic import BaseModel

from app.api.analytics_schemas import (
    AnalyticsQueryResponse,
    ChartSpec,
    SourceInfo,
)
from app.core.config import settings
from app.llm.client import get_llm_client
from app.llm.fallback import generate_with_fallback
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
            settings.resolved_rag_embedding_model,
        )
        self._semaphore = asyncio.Semaphore(1)

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

    async def run(
        self,
        messages: list[dict],
        question: str,
        top_k: int = 5,
        thinking_callback: Callable[[str, str], Awaitable[None]] | None = None,
    ) -> AnalyticsQueryResponse:
        logger.info(f"chat_query: {question[:120]}...")

        if thinking_callback:
            await thinking_callback(
                "retrieving", "Searching knowledge base for relevant documents..."
            )
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

        if thinking_callback:
            model_name = settings.resolved_model
            await thinking_callback(
                "generating", f"Generating response via {model_name}..."
            )
        try:
            async with self._semaphore:
                combined = f"{CHAT_SYSTEM_PROMPT}\n\n{prompt}"
                result = await generate_with_fallback(self.client, combined)
                raw = result.content
        except Exception as e:
            logger.error(f"chat_generation_failed: {e}")
            return AnalyticsQueryResponse(
                question=question,
                summary="Sorry, inference unavailable. All AI providers are currently unreachable. Please try again later.",
                sources=sources,
                error=str(e)[:500],
            )

        return self._parse_response(question, raw, sources)

    def _artifact_filter_for_question(self, question: str) -> dict | None:
        q = question.lower()
        metrics_keywords = {
            "accuracy",
            "qwk",
            "auc",
            "f1",
            "performance",
            "metric",
            "kappa",
            "roc",
            "precision",
            "recall",
        }
        feature_keywords = {
            "feature importance",
            "important feature",
            "what matters",
            "key factor",
            "top feature",
        }
        prediction_keywords = {
            "prediction",
            "predict",
            "dr grade",
            "grade distribution",
            "severity",
            "stage",
        }
        explanation_keywords = {
            "explain",
            "explanation",
            "xai",
            "why",
            "gradcam",
            "reason",
            "attribution",
        }
        patient_keywords = {"patient", "demographic", "age", "gender", "population"}
        ocr_keywords = {
            "ocr",
            "report",
            "scan",
            "oct",
            "fundus",
            "clinical finding",
            "edema",
            "thickness",
        }

        matched: list[str] = []
        if any(kw in q for kw in metrics_keywords):
            matched.extend(["clinical_metrics", "imaging_metrics"])
        if any(kw in q for kw in feature_keywords):
            matched.append("clinical_feature_importance")
        if any(kw in q for kw in prediction_keywords):
            matched.append("db_predictions")
        if any(kw in q for kw in explanation_keywords):
            matched.append("db_explanations")
        if any(kw in q for kw in patient_keywords):
            matched.append("db_patients")
        if any(kw in q for kw in ocr_keywords):
            matched.append("ocr_reports")

        if matched:
            return {"artifact_id": {"$in": matched}}
        return None

    def _retrieve_context(
        self, question: str, top_k: int
    ) -> tuple[str, list[SourceInfo]]:
        try:
            metadata_filter = self._artifact_filter_for_question(question)
            results = self.store.query(
                question, top_k=top_k, metadata_filter=metadata_filter
            )
        except Exception as e:
            logger.warning(f"chat_retrieval_failed: {e}")
            return "", []

        if not results:
            return "", []

        seen_docs: set[tuple[str, str]] = set()
        seen_hashes: set[str] = set()
        reassembled: list[tuple[float, str, SourceInfo]] = []

        for doc, score in results:
            metadata = getattr(doc, "metadata", {}) or {}
            artifact_id = str(metadata.get("artifact_id", "unknown"))
            content_hash = str(metadata.get("content_hash", ""))
            doc_key = (artifact_id, content_hash)
            if doc_key in seen_docs:
                continue
            seen_docs.add(doc_key)

            try:
                chunks = self.store.get_document_chunks(artifact_id, content_hash)
                full_text = "\n".join(chunks)
                first_chunk = chunks[0] if chunks else ""
                first_hash = content_hash[:12] if content_hash else "unknown"
            except Exception:
                full_text = getattr(doc, "page_content", str(doc)).strip()
                first_chunk = full_text
                first_hash = content_hash[:12] if content_hash else "unknown"

            h = content_hash[:12] if content_hash else "unknown"
            if h not in seen_hashes:
                seen_hashes.add(h)
            else:
                continue

            reassembled.append(
                (
                    score,
                    f"[source: {artifact_id} / {first_hash}]\n{full_text}",
                    SourceInfo(
                        artifact_id=artifact_id,
                        snippet=first_chunk[:200],
                    ),
                )
            )

        reassembled.sort(key=lambda x: x[0])
        context = "\n\n---\n\n".join(text for _, text, _ in reassembled)
        max_chars = 10000
        if len(context) > max_chars:
            context = context[:max_chars]
        return context, [s for _, _, s in reassembled][:4]

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
