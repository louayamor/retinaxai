"""
API Routes for LLMOps Service.

Includes synchronous and asynchronous report generation,
job status tracking, RAG management, and training workflows.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from loguru import logger

from app.api.analytics_schemas import AnalyticsQueryRequest, AnalyticsQueryResponse
from app.core.config import settings
from app.pipeline.analytics_pipeline import AnalyticsPipeline
from app.pipeline.chat_pipeline import ChatPipeline, ChatRequest
from app.pipeline.indexing_pipeline import IndexingPipeline
from app.pipeline.inference_pipeline import InferencePipeline, get_inference_pipeline
from app.pipeline.xai_pipeline import XAIPipeline, get_xai_pipeline
from app.services.job_manager import JobManager, JobStatus, get_job_manager
from app.services.operation_state import (
    OperationStateManager,
    get_operation_state_manager,
)
from app.services.shap_service import ShapService, get_shap_service
from app.services.websocket_client import WebSocketClient, get_websocket_client
from app.vectorstore.chroma_store import ChromaStore

router = APIRouter(prefix="/api", tags=["llmops"])


class GenerateRequest(BaseModel):
    model: str | None = None
    prompt: str | None = None
    stream: bool = False
    format: str | None = None
    patient: dict | None = None
    prediction: dict | None = None
    cleaned_summary: str = ""
    raw_ocr_text: str = ""
    report_type: str = Field(default="report")
    language: str = Field(default="en")
    tone: str = Field(default="clinical")


class RagStatusResponse(BaseModel):
    status: str
    schema_version: str | None = None
    run_id: str | None = None
    artifact_count: int = 0
    total_documents: int = 0
    collection_name: str | None = None
    persist_directory: str | None = None
    last_updated: str | None = None


class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    model: str


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        llm_provider=settings.llm_provider.value,
        model=settings.llm_model,
    )


@router.post("/generate")
async def generate(
    payload: GenerateRequest,
    pipeline: InferencePipeline = Depends(get_inference_pipeline),
) -> dict[str, str]:
    """
    Synchronous report generation.

    Returns the report immediately. Use for quick generation.
    For long-running reports, use /generate/async instead.
    """
    try:
        result = await pipeline.generate_report(payload.model_dump())
        return {"response": json.dumps(result)}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/generate/async")
async def generate_async(
    payload: GenerateRequest,
    job_manager: JobManager = Depends(get_job_manager),
) -> dict[str, str]:
    """
    Asynchronous report generation.

    Submits a job and returns immediately with a job ID.
    Poll /jobs/{job_id} to check status.
    """
    job_id = await job_manager.submit(
        job_type="report_generation",
        payload=payload.model_dump(),
        max_retries=3,
    )
    return {"job_id": job_id, "status": "pending"}


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    job_type: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    result: dict | None = None
    error: str | None = None
    retry_count: int


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(
    job_id: str,
    job_manager: JobManager = Depends(get_job_manager),
) -> JobStatusResponse:
    """
    Get the status of an async report generation job.
    """
    job = job_manager.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return JobStatusResponse(
        job_id=job.id,
        status=job.status.value,
        job_type=job.job_type,
        created_at=job.created_at.isoformat(),
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        result=job.result,
        error=job.error,
        retry_count=job.retry_count,
    )


@router.get("/jobs")
def list_jobs(
    status: str | None = Query(
        None, description="Filter by status: pending, running, completed, failed"
    ),
    limit: int = Query(100, ge=1, le=1000),
    job_manager: JobManager = Depends(get_job_manager),
) -> dict:
    """
    List recent report generation jobs.
    """
    status_filter = None
    if status:
        try:
            status_filter = JobStatus(status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

    jobs = job_manager.get_jobs(status=status_filter, limit=limit)

    return {
        "total": len(jobs),
        "jobs": [
            {
                "job_id": j.id,
                "status": j.status.value,
                "job_type": j.job_type,
                "created_at": j.created_at.isoformat(),
                "completed_at": j.completed_at.isoformat() if j.completed_at else None,
            }
            for j in jobs
        ],
    }


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    job_manager: JobManager = Depends(get_job_manager),
) -> dict:
    """
    Cancel a pending or running job.
    """
    result = await job_manager.cancel_job(job_id)

    if not result:
        raise HTTPException(status_code=400, detail=f"Job {job_id} cannot be cancelled")

    return {"job_id": job_id, "status": "cancelled"}


@router.post("/rag/reindex")
def rag_reindex() -> dict[str, object]:
    result = IndexingPipeline().run()
    return {"status": "ok", "result": result}


@router.get("/rag/status", response_model=RagStatusResponse)
def rag_status() -> RagStatusResponse:
    store = ChromaStore(
        settings.rag_chroma_persist_directory,
        settings.rag_chroma_collection_name,
        settings.rag_embedding_model,
    )
    state = store.read_state() or {}
    return RagStatusResponse(
        status="ready" if state.get("artifact_count", 0) > 0 else "idle",
        schema_version=str(state.get("schema_version"))
        if state.get("schema_version")
        else None,
        run_id=str(state.get("run_id")) if state.get("run_id") else None,
        artifact_count=int(state.get("artifact_count") or 0),
        total_documents=int(state.get("artifact_count") or 0),
        collection_name=store.collection_name,
        persist_directory=str(store.persist_directory),
        last_updated=state.get("updated_at"),
    )


@router.get("/operation/status")
def operation_status(
    op_manager: OperationStateManager = Depends(get_operation_state_manager),
) -> dict:
    op = op_manager.get_operation()
    return {
        "operation": op.state,
        "status": op.state,
        "message": op.message,
        "progress": op.progress or 0.0,
        "started_at": op.started_at,
    }


@router.get("/operation")
def operation(
    op_manager: OperationStateManager = Depends(get_operation_state_manager),
) -> dict:
    """Alias for /operation/status for frontend compatibility."""
    op = op_manager.get_operation()
    return {
        "operation": op.state,
        "status": op.state,
        "message": op.message,
        "progress": op.progress or 0.0,
        "started_at": op.started_at,
    }


class XAIPredictionRequest(BaseModel):
    prediction_id: str
    dr_grade: str | int
    confidence: float
    clinical_features: dict | None = None
    gradcam_regions: dict | None = None


def _normalize_regions(
    regions: list[str] | list[dict] | None,
) -> list[dict]:
    """Normalize mixed str/dict region input to uniform list[dict]."""
    if not regions:
        return []
    result = []
    for r in regions:
        if isinstance(r, dict):
            result.append(r)
        elif isinstance(r, str):
            result.append(
                {"name": r, "intensity": 0.0, "area": 0, "saliency_score": 0.0}
            )
    return result


class XAIGradCAMRequest(BaseModel):
    prediction_id: str
    left_eye_regions: list[str] | list[dict] | None = None
    right_eye_regions: list[str] | list[dict] | None = None
    dr_grade: str | int | None = None
    confidence: float | None = None


class XAISeverityRequest(BaseModel):
    prediction_id: str
    patient_data: dict
    dr_grade: str | int
    risk_factors: list[str] = []


@router.post("/xai/explain")
async def explain_prediction(
    payload: XAIPredictionRequest,
    pipeline: XAIPipeline = Depends(get_xai_pipeline),
) -> dict:
    """
    Generate natural language explanation of DR prediction.
    Uses GradCAM regions for imaging-based predictions or SHAP for clinical features.
    """
    return await pipeline.explain_prediction(
        prediction_id=payload.prediction_id,
        dr_grade=payload.dr_grade,
        confidence=payload.confidence,
        clinical_features=payload.clinical_features,
        gradcam_regions=payload.gradcam_regions,
    )


@router.post("/xai/gradcam")
async def explain_gradcam(
    payload: XAIGradCAMRequest,
    pipeline: XAIPipeline = Depends(get_xai_pipeline),
) -> dict:
    """
    Interpret highlighted regions in GradCAM heatmaps with clinical specificity.
    Includes per-eye analysis with DR grade and confidence context.
    """
    return await pipeline.explain_gradcam(
        prediction_id=payload.prediction_id,
        left_eye_regions=_normalize_regions(payload.left_eye_regions),
        right_eye_regions=_normalize_regions(payload.right_eye_regions),
        dr_grade=payload.dr_grade,
        confidence=payload.confidence,
    )


@router.post("/xai/severity")
async def generate_severity(
    payload: XAISeverityRequest,
    pipeline: XAIPipeline = Depends(get_xai_pipeline),
) -> dict:
    """
    Generate clinical severity report with risk level and recommendations.
    """
    return await pipeline.generate_severity_report(
        prediction_id=payload.prediction_id,
        patient_data=payload.patient_data,
        dr_grade=payload.dr_grade,
        risk_factors=payload.risk_factors,
    )


class TrainingCompleteRequest(BaseModel):
    job_id: str
    pipeline: str
    imaging_version: str | None = None
    clinical_version: str | None = None


@router.post("/workflows/training-complete")
async def workflow_training_complete(
    payload: TrainingCompleteRequest,
    ws_client: WebSocketClient = Depends(get_websocket_client),
) -> dict:
    """
    Handle training completion event from MLOps.
    Triggers RAG reindexing and batch GradCAM analysis.
    """
    logger.info(
        f"Training workflow triggered: job_id={payload.job_id}, "
        f"pipeline={payload.pipeline}, imaging={payload.imaging_version}, "
        f"clinical={payload.clinical_version}"
    )

    asyncio.create_task(
        ws_client.send_llmops_event(
            event_type="rag_indexing",
            status="started",
            progress=0,
            message="Starting RAG reindexing after training...",
            details={
                "job_id": payload.job_id,
                "pipeline": payload.pipeline,
                "imaging_version": payload.imaging_version,
                "clinical_version": payload.clinical_version,
            },
        )
    )

    try:
        from app.pipeline.indexing_pipeline import IndexingPipeline

        pipeline = IndexingPipeline()
        result = pipeline.run()

        asyncio.create_task(
            ws_client.send_llmops_event(
                event_type="rag_indexing",
                status="completed",
                progress=100,
                message=f"RAG reindexing complete: {result.get('indexed', 0)} artifacts",
                details={"result": result},
            )
        )
        logger.info(f"RAG reindexing complete: {result}")

    except Exception as e:
        asyncio.create_task(
            ws_client.send_llmops_event(
                event_type="rag_indexing",
                status="failed",
                progress=0,
                message=f"RAG reindexing failed: {e}",
                details={"error": str(e)},
            )
        )
        logger.warning(f"RAG reindexing failed: {e}")

    return {
        "status": "ok",
        "workflow_id": f"workflow_{payload.job_id}",
        "message": "Training workflow triggered successfully",
    }


class ShapExplainRequest(BaseModel):
    features: dict[str, Any]
    pipeline: str = "clinical"


class ShapExplainResponse(BaseModel):
    model_type: str
    expected_value: float
    pipeline: str
    explanation: dict


class GlobalImportanceResponse(BaseModel):
    pipeline: str
    importance: dict[str, float]


class BiasCheckResponse(BaseModel):
    pipeline: str
    demographic_column: str
    results: dict[str, Any]


@router.post("/xai/shap/explain", response_model=ShapExplainResponse)
async def shap_explain_prediction(
    payload: ShapExplainRequest,
    service: ShapService = Depends(get_shap_service),
) -> ShapExplainResponse:
    """
    Generate SHAP explanation for clinical model prediction.
    DEPRECATED: Use /xai/explain for unified XAI explanation.
    """
    logger.info(f"SHAP explanation requested for pipeline: {payload.pipeline}")

    try:
        explanation = await service.explain_prediction(
            features=payload.features,
            pipeline=payload.pipeline,
        )

        return ShapExplainResponse(
            model_type=explanation.model_type,
            expected_value=explanation.expected_value,
            pipeline=explanation.pipeline,
            explanation=explanation.to_dict(),
        )

    except Exception as e:
        logger.error(f"SHAP explanation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/xai/shap/importance/{pipeline}", response_model=GlobalImportanceResponse)
async def shap_get_global_importance(
    pipeline: str,
    service: ShapService = Depends(get_shap_service),
) -> GlobalImportanceResponse:
    """
    Get cached global SHAP feature importance.
    """
    importance = service.get_global_importance(pipeline)

    return GlobalImportanceResponse(
        pipeline=pipeline,
        importance=importance,
    )


@router.post(
    "/xai/shap/importance/{pipeline}/compute", response_model=GlobalImportanceResponse
)
async def shap_compute_global_importance(
    pipeline: str,
    test_path: str | None = None,
    sample_size: int = 100,
    service: ShapService = Depends(get_shap_service),
) -> GlobalImportanceResponse:
    """
    Compute global SHAP feature importance on test dataset.
    """
    if test_path:
        test_csv = Path(test_path)
        if not test_csv.is_absolute():
            test_csv = settings.artifacts_root / test_path
    else:
        test_csv = (
            settings.artifacts_root / "data" / "processed" / pipeline / "test.csv"
        )

    if not test_csv.exists():
        raise HTTPException(status_code=404, detail=f"Test data not found: {test_csv}")

    importance = await service.compute_global_importance(
        test_csv=test_csv,
        pipeline=pipeline,
        sample_size=sample_size,
    )

    return GlobalImportanceResponse(
        pipeline=pipeline,
        importance=importance,
    )


@router.post("/xai/shap/bias/{pipeline}", response_model=BiasCheckResponse)
async def shap_check_bias(
    pipeline: str,
    demographic_column: str = "patient_gender",
    test_path: str | None = None,
    service: ShapService = Depends(get_shap_service),
) -> BiasCheckResponse:
    """
    Check for potential bias in model predictions across demographic groups.
    """
    if test_path:
        test_csv = Path(test_path)
        if not test_csv.is_absolute():
            test_csv = settings.artifacts_root / test_path
    else:
        test_csv = (
            settings.artifacts_root / "data" / "processed" / pipeline / "test.csv"
        )

    if not test_csv.exists():
        raise HTTPException(status_code=404, detail=f"Test data not found: {test_csv}")

    results = await service.check_bias(
        test_csv=test_csv,
        demographic_col=demographic_column,
        pipeline=pipeline,
    )

    return BiasCheckResponse(
        pipeline=pipeline,
        demographic_column=demographic_column,
        results=results,
    )


_analytics_pipeline: AnalyticsPipeline | None = None


def _get_analytics_pipeline() -> AnalyticsPipeline:
    global _analytics_pipeline
    if _analytics_pipeline is None:
        _analytics_pipeline = AnalyticsPipeline()
    return _analytics_pipeline


@router.post(
    "/analytics/query",
    response_model=AnalyticsQueryResponse,
    summary="Query analytics",
    description="Analyze RAG-indexed data with natural language. Returns summary + optional chart spec.",
)
async def analytics_query(
    payload: AnalyticsQueryRequest,
) -> AnalyticsQueryResponse:
    try:
        pipeline = _get_analytics_pipeline()
        return await pipeline.run(payload)
    except Exception as exc:
        logger.error(f"analytics_query_failed: {exc}")
        raise HTTPException(status_code=503, detail=str(exc)) from exc


_chat_pipeline: ChatPipeline | None = None


def _get_chat_pipeline() -> ChatPipeline:
    global _chat_pipeline
    if _chat_pipeline is None:
        _chat_pipeline = ChatPipeline()
    return _chat_pipeline


@router.post(
    "/chat",
    response_model=AnalyticsQueryResponse,
    summary="Chat with AI",
)
async def chat(
    payload: ChatRequest,
) -> AnalyticsQueryResponse:
    try:
        pipeline = _get_chat_pipeline()
        return await pipeline.run(payload.messages, payload.question, payload.top_k)
    except Exception as exc:
        logger.error(f"chat_failed: {exc}")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
