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

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from loguru import logger

from app.core.config import settings
from app.pipeline.indexing_pipeline import IndexingPipeline
from app.pipeline.inference_pipeline import InferencePipeline
from app.services.operation_state import get_operation
from app.services.job_manager import JobStatus, get_job_manager
from app.services.shap_service import get_shap_service
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
async def generate(payload: GenerateRequest) -> dict[str, str]:
    """
    Synchronous report generation.

    Returns the report immediately. Use for quick generation.
    For long-running reports, use /generate/async instead.
    """
    try:
        pipeline = InferencePipeline()
        result = await pipeline.generate_report(payload.model_dump())
        return {"response": json.dumps(result)}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/generate/async")
async def generate_async(payload: GenerateRequest) -> dict[str, str]:
    """
    Asynchronous report generation.

    Submits a job and returns immediately with a job ID.
    Poll /jobs/{job_id} to check status.
    """
    job_manager = get_job_manager()
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
def get_job_status(job_id: str) -> JobStatusResponse:
    """
    Get the status of an async report generation job.
    """
    job_manager = get_job_manager()
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
) -> dict:
    """
    List recent report generation jobs.
    """
    job_manager = get_job_manager()

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
async def cancel_job(job_id: str) -> dict:
    """
    Cancel a pending or running job.
    """
    job_manager = get_job_manager()
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
def operation_status() -> dict:
    op = get_operation()
    return {
        "operation": op.state,
        "status": op.state,
        "message": op.message,
        "progress": op.progress or 0.0,
        "started_at": op.started_at,
    }


@router.get("/operation")
def operation() -> dict:
    """Alias for /operation/status for frontend compatibility."""
    op = get_operation()
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
    vascular_biomarkers: dict | None = None


class XAIGradCAMRequest(BaseModel):
    prediction_id: str
    left_eye_regions: list[str] | None = None
    right_eye_regions: list[str] | None = None


class XAISeverityRequest(BaseModel):
    prediction_id: str
    patient_data: dict
    dr_grade: str | int
    risk_factors: list[str] = []


@router.post("/xai/explain")
async def explain_prediction(payload: XAIPredictionRequest) -> dict:
    """
    Generate natural language explanation of DR prediction.
    Uses GradCAM regions for imaging-based predictions or SHAP for clinical features.
    """
    from app.pipeline.xai_pipeline import get_xai_pipeline

    pipeline = get_xai_pipeline()
    return await pipeline.explain_prediction(
        prediction_id=payload.prediction_id,
        dr_grade=payload.dr_grade,
        confidence=payload.confidence,
        clinical_features=payload.clinical_features,
        gradcam_regions=payload.gradcam_regions,
        vascular_biomarkers=payload.vascular_biomarkers,
    )


@router.post("/xai/gradcam")
async def explain_gradcam(payload: XAIGradCAMRequest) -> dict:
    """
    Interpret highlighted regions in GradCAM heatmaps.
    """
    from app.pipeline.xai_pipeline import get_xai_pipeline

    pipeline = get_xai_pipeline()
    return await pipeline.explain_gradcam(
        prediction_id=payload.prediction_id,
        left_eye_regions=payload.left_eye_regions or [],
        right_eye_regions=payload.right_eye_regions or [],
    )


@router.post("/xai/severity")
async def generate_severity(payload: XAISeverityRequest) -> dict:
    """
    Generate clinical severity report with risk level and recommendations.
    """
    from app.pipeline.xai_pipeline import get_xai_pipeline

    pipeline = get_xai_pipeline()
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
async def workflow_training_complete(payload: TrainingCompleteRequest) -> dict:
    """
    Handle training completion event from MLOps.
    Triggers RAG reindexing and batch GradCAM analysis.
    """
    from app.services.websocket_client import get_websocket_client

    ws_client = get_websocket_client()

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
async def shap_explain_prediction(payload: ShapExplainRequest) -> ShapExplainResponse:
    """
    Generate SHAP explanation for clinical model prediction.
    DEPRECATED: Use /xai/explain for unified XAI explanation.
    """
    from app.services.shap_service import get_shap_service

    logger.info(f"SHAP explanation requested for pipeline: {payload.pipeline}")

    try:
        service = get_shap_service()
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
async def shap_get_global_importance(pipeline: str) -> GlobalImportanceResponse:
    """
    Get cached global SHAP feature importance.
    """
    service = get_shap_service()
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
) -> GlobalImportanceResponse:
    """
    Compute global SHAP feature importance on test dataset.
    """
    from app.core.config import settings

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

    service = get_shap_service()
    importance = service.compute_global_importance(
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
) -> BiasCheckResponse:
    """
    Check for potential bias in model predictions across demographic groups.
    """
    from app.core.config import settings

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

    service = get_shap_service()
    results = service.check_bias(
        test_csv=test_csv,
        demographic_col=demographic_column,
        pipeline=pipeline,
    )

    return BiasCheckResponse(
        pipeline=pipeline,
        demographic_column=demographic_column,
        results=results,
    )
