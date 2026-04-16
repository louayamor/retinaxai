from __future__ import annotations

import json
import time

from loguru import logger

from app.core.config import settings
from app.llm.client import get_llm_client
from app.prompts.templates import REPORT_SYSTEM_PROMPT, REPORT_USER_PROMPT
from app.services.operation_state import set_operation
from app.utils.helpers import dump_compact
from app.vectorstore.chroma_store import ChromaStore


class InferencePipeline:
    _instance: InferencePipeline | None = None

    def __new__(cls) -> InferencePipeline:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        provider = (
            settings.llm_provider.value
            if hasattr(settings.llm_provider, "value")
            else str(settings.llm_provider)
        )
        token = settings.github_token if provider == "github" else settings.llm_api_key
        base_url = (
            settings.github_endpoint if provider == "github" else settings.llm_base_url
        )

        timeout = settings.timeout_seconds
        max_tokens = settings.max_tokens

        client_kwargs: dict[str, str | int] = {
            "model": settings.llm_model,
            "timeout_seconds": timeout,
            "max_tokens": max_tokens,
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
        self.store = ChromaStore(
            settings.rag_chroma_persist_directory,
            settings.rag_chroma_collection_name,
            settings.rag_embedding_model,
        )
        self._initialized = True

    def _build_retrieval_context(self, payload: dict) -> tuple[str, float]:
        start_time = time.time()

        cleaned_summary = payload.get("cleaned_summary", "") or ""
        raw_ocr_text = payload.get("raw_ocr_text", "") or ""

        if len(cleaned_summary) > 2000:
            cleaned_summary = cleaned_summary[:2000] + "..."
        if len(raw_ocr_text) > 3000:
            raw_ocr_text = raw_ocr_text[:3000] + "..."

        parts = [cleaned_summary, raw_ocr_text]
        query = "\n".join(part for part in parts if part)
        if not query.strip():
            logger.debug("No query text provided for retrieval")
            return "", 0.0

        logger.info(f"Retrieving context for query (length: {len(query)})")
        try:
            results = self.store.query(query, top_k=4)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return "", 0.0

        if not results:
            logger.warning("No results returned from retrieval")
            return "", 0.0

        if isinstance(results, tuple):
            documents = results[0] or []
        else:
            documents = results[0] if isinstance(results, list) and results else []

        snippets = []
        for item in documents[:4]:
            doc = item[0] if isinstance(item, tuple) else item
            text = getattr(doc, "page_content", str(doc)).strip()
            metadata = getattr(doc, "metadata", {}) or {}
            if text:
                snippets.append(
                    f"[source: {metadata.get('artifact_id', 'unknown')}] {text}"
                )

        logger.info(f"Retrieved {len(snippets)} context snippets")

        context = "\n".join(snippets)
        max_tokens = 6000
        if len(context) > max_tokens * 4:
            context = context[: max_tokens * 4]
            logger.warning(f"Truncated context to {max_tokens} tokens")

        return context, time.time() - start_time

    def generate_report(self, payload: dict) -> dict[str, str]:
        try:
            return self._generate_report_internal(payload)
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            set_operation("error", str(e)[:200])
            raise

    def _generate_report_internal(self, payload: dict) -> dict[str, str]:
        model_name = str(payload.get("model") or settings.llm_model)

        logger.info("Building retrieval context...")
        set_operation("retrieving", "Retrieving context from RAG...")
        retrieved_context, retrieval_time = self._build_retrieval_context(payload)

        logger.info(f"Generating report with {model_name}...")
        set_operation("generating", "Generating report with LLM...")
        start_time = time.time()
        user_prompt = REPORT_USER_PROMPT.format(
            patient=dump_compact(payload.get("patient", {})),
            prediction=dump_compact(payload.get("prediction", {})),
            cleaned_summary=payload.get("cleaned_summary", ""),
            raw_ocr_text=payload.get("raw_ocr_text", ""),
            report_type=payload.get("report_type", "unknown"),
            language=payload.get("language", "en"),
            tone=payload.get("tone", "clinical"),
            retrieved_context=retrieved_context,
        )

        content = self.client.generate(user_prompt, REPORT_SYSTEM_PROMPT)
        generation_time = time.time() - start_time
        logger.info(f"Generation complete ({generation_time:.2f}s)")

        parsed = None
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = None

        # Store full structured JSON content for new format
        if isinstance(parsed, dict) and "patient_info" in parsed:
            # New structured format - store the full JSON
            report_content = json.dumps(parsed)
            summary = str(parsed.get("summary", ""))
        elif isinstance(parsed, dict):
            # Old format with content key
            report_content = str(parsed.get("content", content))
            summary = str(parsed.get("summary", report_content[:400]))
        else:
            # Plain text - not structured
            report_content = content
            summary = content[:400]

        logger.info(f"Report generated (content length: {len(report_content)})")
        set_operation("idle", "Ready")
        self._log_to_mlflow(
            model_name=model_name,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            has_retrieved_context=bool(retrieved_context),
            run_index=int(time.time()) % 1000,
        )

        return {"content": report_content, "summary": summary, "model_used": model_name}

    def _log_to_mlflow(
        self,
        model_name: str,
        retrieval_time: float,
        generation_time: float,
        has_retrieved_context: bool,
        run_index: int = 0,
    ) -> None:
        try:
            import mlflow

            if not settings.mlflow_tracking_uri:
                return

            with mlflow.start_run(run_name=f"llmops_inference_{run_index:03d}"):
                mlflow.log_params(
                    {
                        "model": model_name,
                        "report_type": "clinical",
                    }
                )
                mlflow.log_metrics(
                    {
                        "retrieval_latency_seconds": retrieval_time,
                        "generation_latency_seconds": generation_time,
                        "total_latency_seconds": retrieval_time + generation_time,
                        "retrieved_doc_count": int(has_retrieved_context),
                    }
                )
        except Exception:
            pass
