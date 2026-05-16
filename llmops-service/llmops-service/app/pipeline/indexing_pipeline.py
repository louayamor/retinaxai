from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import TypedDict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from app.core.config import settings
from app.rag.manifest_client import fetch_manifest
from app.services.operation_state import set_operation
from app.vectorstore.chroma_store import ChromaStore
from app.vectorstore.document_loader import normalize_artifact


class PipelineResult(TypedDict):
    schema_version: str
    run_id: str
    artifact_count: int
    pipeline: str
    document_count: int
    chunk_count: int


class IndexingPipeline:
    def run(self) -> PipelineResult:
        start_time = time.time()
        try:
            return self._run_internal(start_time)
        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            set_operation("error", str(e)[:200])
            raise

    def _run_internal(self, start_time: float) -> PipelineResult:
        set_operation("indexing", "Fetching manifest from MLOps...")

        logger.info("Fetching manifest from MLOps...")
        manifest = fetch_manifest(
            settings.rag_manifest_url, timeout=settings.timeout_seconds
        )
        logger.info(
            f"Manifest: run_id={manifest.run_id}, artifacts={manifest.artifact_count}"
        )

        set_operation("indexing", "Initializing ChromaDB store...")
        logger.info("Initializing ChromaDB store...")
        store = ChromaStore(
            settings.rag_chroma_persist_directory,
            settings.rag_chroma_collection_name,
            settings.rag_embedding_model,
        )
        store.ensure_ready()
        store.upsert_documents([])

        set_operation("indexing", "Loading and normalizing artifacts...")
        logger.info("Loading and normalizing artifacts...")
        documents = []
        for artifact in manifest.artifacts:
            if not artifact.indexable:
                logger.debug(f"Skipping non-indexable artifact: {artifact.artifact_id}")
                continue
            docs = normalize_artifact(artifact, run_id=manifest.run_id)
            logger.debug(f"Loaded {len(docs)} documents from {artifact.artifact_id}")
            documents.extend(docs)
        logger.info(f"Total documents loaded: {len(documents)}")

        set_operation("indexing", "Splitting documents into chunks...")
        logger.info("Splitting documents into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        logger.info(f"Total chunks created: {len(chunks)}")

        set_operation("indexing", "Rebuilding ChromaDB collection...")
        logger.info("Rebuilding ChromaDB collection...")
        state = {
            "schema_version": manifest.schema_version,
            "run_id": manifest.run_id,
            "artifact_count": manifest.artifact_count,
            "pipeline": manifest.pipeline,
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if hasattr(store, "rebuild_collection_atomically"):
            try:
                store.rebuild_collection_atomically(chunks, state)
            except Exception:
                logger.warning(
                    "Rebuild failed, nuking corrupted ChromaDB data and retrying..."
                )
                store._nuke_corrupted_data()
                store.rebuild_collection_atomically(chunks, state)
        else:
            store.upsert_documents(chunks)
            if hasattr(store, "write_state"):
                store.write_state(state)

        logger.info("Indexing complete!")
        set_operation("idle", "Ready")
        elapsed_time = time.time() - start_time

        self._log_to_mlflow(
            manifest=manifest,
            document_count=len(documents),
            chunk_count=len(chunks),
            elapsed_time=elapsed_time,
            run_index=len(chunks) % 1000,
        )

        return {
            "schema_version": manifest.schema_version,
            "run_id": manifest.run_id,
            "artifact_count": manifest.artifact_count,
            "pipeline": manifest.pipeline,
            "document_count": len(documents),
            "chunk_count": len(chunks),
        }

    def _log_to_mlflow(
        self,
        manifest,
        document_count: int,
        chunk_count: int,
        elapsed_time: float,
        run_index: int = 0,
    ) -> None:
        try:
            import mlflow

            if not settings.mlflow_tracking_uri:
                return

            with mlflow.start_run(
                run_name=f"llmops_indexing_{manifest.run_id}_{run_index:03d}"
            ):
                mlflow.log_params(
                    {
                        "run_id": manifest.run_id,
                        "pipeline": manifest.pipeline,
                        "schema_version": manifest.schema_version,
                    }
                )
                mlflow.log_metrics(
                    {
                        "artifact_count": manifest.artifact_count,
                        "document_count": document_count,
                        "chunk_count": chunk_count,
                        "indexing_duration_seconds": elapsed_time,
                    }
                )  # type: ignore[arg-type]
        except Exception:
            pass
