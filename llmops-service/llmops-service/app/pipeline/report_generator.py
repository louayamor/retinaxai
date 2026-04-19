"""
Report Generation Handler for Async Job Processing.

This module provides the handler for async report generation jobs,
including token counting and retry logic.
"""

from __future__ import annotations

from loguru import logger

from app.core.config import settings
from app.pipeline.inference_pipeline import InferencePipeline
from app.services.job_manager import Job
from app.utils.helpers import dump_compact


def count_tokens(text: str) -> int:
    """
    Estimate token count for text.

    Simple approximation: ~4 chars per token for English text.
    For production, use tiktoken or model-specific tokenizer.
    """
    return len(text) // 4


def truncate_context(context: str, max_tokens: int = 2000) -> str:
    """
    Truncate context to fit within token limits.

    Keeps the beginning and end of the context, truncates middle if needed.
    """
    estimated_tokens = count_tokens(context)

    if estimated_tokens <= max_tokens:
        return context

    # Simple truncation: keep first 60% and last 40%
    max_chars = max_tokens * 4
    first_part = int(max_chars * 0.6)
    last_part = int(max_chars * 0.4)

    return context[:first_part] + "\n...[truncated]...\n" + context[-last_part:]


async def generate_report_handler(job: Job) -> dict:
    """
    Handle report generation job.

    This is the async handler registered with the JobManager.

    Args:
        job: The job containing report generation parameters

    Returns:
        dict: The generated report with content, summary, and model info
    """
    payload = job.payload

    logger.info(f"Starting report generation for job {job.id}")

    # Validate required fields
    required_fields = ["patient", "prediction"]
    for field in required_fields:
        if field not in payload:
            raise ValueError(f"Missing required field: {field}")

    # Token counting for context management
    patient_str = dump_compact(payload.get("patient", {}))
    prediction_str = dump_compact(payload.get("prediction", {}))
    cleaned_summary = payload.get("cleaned_summary", "")
    raw_ocr_text = payload.get("raw_ocr_text", "")

    total_input = patient_str + prediction_str + cleaned_summary + raw_ocr_text
    input_tokens = count_tokens(total_input)
    logger.info(f"Job {job.id}: Input tokens estimated: {input_tokens}")

    # Truncate if needed
    max_context_tokens = 3000  # Leave room for system prompt and retrieved context
    if input_tokens > max_context_tokens:
        logger.warning(f"Job {job.id}: Truncating context from {input_tokens} tokens")
        # Proportionally truncate each component
        ratio = max_context_tokens / input_tokens
        cleaned_summary = truncate_context(cleaned_summary, int(1500 * ratio))
        raw_ocr_text = truncate_context(raw_ocr_text, int(1000 * ratio))

    # Build truncated payload
    processed_payload = {
        **payload,
        "cleaned_summary": cleaned_summary,
        "raw_ocr_text": raw_ocr_text,
    }

    # Use InferencePipeline for report generation
    pipeline = InferencePipeline()
    result = await pipeline.generate_report(processed_payload)

    output_tokens = count_tokens(result.get("content", ""))
    logger.info(f"Job {job.id}: Output tokens estimated: {output_tokens}")

    return {
        "content": result.get("content", ""),
        "summary": result.get("summary", ""),
        "model_used": result.get("model_used", settings.llm_model),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "job_id": job.id,
    }


async def generate_report_sync(payload: dict) -> dict:
    """
    Synchronous report generation (for backward compatibility).

    Use this for quick report generation without job queue.
    For production, prefer the async job queue.

    Args:
        payload: Report generation parameters

    Returns:
        dict: The generated report
    """
    from app.services.job_manager import Job, JobStatus

    job = Job(
        id="sync",
        job_type="report_generation",
        payload=payload,
        status=JobStatus.RUNNING,
    )

    return await generate_report_handler(job)
