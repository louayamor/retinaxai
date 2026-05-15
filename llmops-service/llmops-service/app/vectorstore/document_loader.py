from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.manifest_client import RagArtifact
from app.utils.helpers import normalize_whitespace

_BASE64_PATTERN = re.compile(r"^[A-Za-z0-9+/=]{100,}$")


def _is_base64_blob(value: str) -> bool:
    return bool(_BASE64_PATTERN.match(value.strip()))


def _strip_binary(d: dict, depth: int = 0) -> dict:
    if depth > 5:
        return {}
    cleaned: dict = {}
    for key, value in d.items():
        if isinstance(value, str) and len(value) > 80 and _is_base64_blob(value):
            continue
        if key in ("base64_png", "base64_jpg", "base64", "png_path", "jpg_path"):
            continue
        if isinstance(value, dict):
            nested = _strip_binary(value, depth + 1)
            if nested:
                cleaned[key] = nested
        elif isinstance(value, list):
            cleaned_list = []
            for item in value:
                if isinstance(item, dict):
                    nested = _strip_binary(item, depth + 1)
                    if nested:
                        cleaned_list.append(nested)
                elif isinstance(item, str) and len(item) > 80 and _is_base64_blob(item):
                    continue
                else:
                    cleaned_list.append(item)
            if cleaned_list:
                cleaned[key] = cleaned_list
        else:
            cleaned[key] = value
    return cleaned


def _format_ocr_reports(reports: list[dict]) -> str:
    lines: list[str] = []
    for i, report in enumerate(reports):
        lines.append(f"--- OCT Report {i + 1} ---")
        meta = report.get("metadata", {})
        lines.append(f"Device: {meta.get('device', 'N/A')}")
        lines.append(f"Report Type: {meta.get('report_type', 'N/A')}")
        lines.append(f"Eye: {meta.get('eye', 'N/A')}")
        lines.append(f"Capture Date: {meta.get('capture_date', 'N/A')}")
        lines.append(f"Image Quality: {meta.get('image_quality', 'N/A')}")
        lines.append(f"Fixation: {meta.get('fixation', 'N/A')}")

        patient = report.get("patient", {})
        lines.append(f"Patient ID: {patient.get('patient_id', 'N/A')}")
        lines.append(f"Age: {patient.get('age', 'N/A')}")
        lines.append(f"Gender: {patient.get('gender', 'N/A')}")

        thickness = report.get("thickness", {})
        lines.append("Thickness:")
        for k, v in thickness.items():
            if v is not None:
                lines.append(f"  {k}: {v}")

        clinical = report.get("clinical", {})
        lines.append("Clinical Findings:")
        for k, v in clinical.items():
            if v is not None:
                lines.append(f"  {k}: {v}")

        lines.append("")
    return "\n".join(lines)


def _format_predictions(predictions: list[dict]) -> str:
    lines: list[str] = []
    for i, pred in enumerate(predictions):
        lines.append(f"--- Prediction {i + 1} ---")
        lines.append(f"Model: {pred.get('model_name', 'N/A')}")
        lines.append(f"Version: {pred.get('model_version', 'N/A')}")
        lines.append(f"Status: {pred.get('status', 'N/A')}")
        conf = pred.get("confidence_score")
        if conf is not None:
            lines.append(f"Confidence Score: {conf}")
        output = pred.get("output_payload") or {}
        grade = output.get("combined_grade") or output.get("dr_grade")
        if grade is not None:
            lines.append(f"DR Grade: {grade}")
        risk = output.get("risk_level") or output.get("severity")
        if risk is not None:
            lines.append(f"Risk Level: {risk}")
        error = pred.get("error_message")
        if error:
            lines.append(f"Error: {error}")
        lines.append(f"Timestamp: {pred.get('created_at', 'N/A')}")
        patient_info = pred.get("patient", {})
        if patient_info:
            lines.append(
                f"Patient: age={patient_info.get('age', 'N/A')}, gender={patient_info.get('gender', 'N/A')}"
            )
        lines.append("")
    return "\n".join(lines)


def _format_explanations(explanations: list[dict]) -> str:
    lines: list[str] = []
    for i, exp in enumerate(explanations):
        lines.append(f"--- XAI Explanation {i + 1} ---")
        lines.append(f"Model Used: {exp.get('model_used', 'N/A')}")
        lines.append(f"Status: {exp.get('status', 'N/A')}")
        summary = exp.get("summary")
        if summary:
            lines.append(f"Summary: {summary}")
        content = exp.get("content", "")
        if content:
            lines.append(f"Explanation: {content[:500]}")
        shap = exp.get("shap_values")
        if shap and isinstance(shap, dict):
            lines.append("SHAP Values (top features):")
            sorted_features = sorted(
                shap.items(),
                key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0,
                reverse=True,
            )
            for feat, val in sorted_features[:10]:
                lines.append(f"  {feat}: {val}")
        prediction_id = exp.get("prediction_id")
        if prediction_id:
            lines.append(f"Prediction ID: {prediction_id}")
        error = exp.get("error_message")
        if error:
            lines.append(f"Error: {error}")
        lines.append("")
    return "\n".join(lines)


def _format_patients(patients: list[dict]) -> str:
    """Format patient demographics, excluding sensitive PII."""
    lines: list[str] = []
    for i, pat in enumerate(patients):
        lines.append(f"--- Patient {i + 1} ---")
        lines.append(f"Age: {pat.get('age', 'N/A')}")
        lines.append(f"Gender: {pat.get('gender', 'N/A')}")
        mrn = pat.get("medical_record_number")
        if mrn:
            lines.append(f"Medical Record: {mrn}")
        ocr_id = pat.get("ocr_patient_id")
        if ocr_id:
            lines.append(f"OCR Patient ID: {ocr_id}")
        lines.append("")
    return "\n".join(lines)


def _serialize_content(artifact_id: str, content: Any) -> str:
    artifact_id = artifact_id.lower()

    if artifact_id == "ocr_reports" and isinstance(content, list):
        return _format_ocr_reports(content)

    if artifact_id == "db_predictions" and isinstance(content, list):
        return _format_predictions(content)

    if artifact_id == "db_explanations" and isinstance(content, list):
        return _format_explanations(content)

    if artifact_id == "db_patients" and isinstance(content, list):
        return _format_patients(content)

    if isinstance(content, list):
        return "\n".join(str(item) for item in content)

    if isinstance(content, dict):
        cleaned = _strip_binary(content)
        parts: list[str] = []
        _flatten_dict(cleaned, parts, prefix="")
        return "\n".join(parts)

    return str(content)


def _flatten_dict(d: dict, parts: list[str], prefix: str = "", depth: int = 0) -> None:
    if depth > 3:
        parts.append(f"{prefix}: (nested)")
        return
    for key, value in d.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            _flatten_dict(value, parts, full_key, depth + 1)
        elif isinstance(value, list):
            if value and isinstance(value[0], dict):
                for i, item in enumerate(value):
                    _flatten_dict(item, parts, f"{full_key}[{i}]", depth + 1)
            else:
                parts.append(f"{full_key}: {', '.join(str(v) for v in value)}")
        else:
            parts.append(f"{full_key}: {value}")


def normalize_artifact(
    artifact: RagArtifact, run_id: str | None = None
) -> list[Document]:
    artifact_id = artifact.artifact_id.value
    content = _serialize_content(artifact_id, artifact.content)
    text = normalize_whitespace(content)

    if artifact_id == "ocr_reports" and isinstance(artifact.content, list):
        cleaned_reports = [_strip_binary(r) for r in artifact.content]
        re_serialized = _serialize_content(artifact_id, cleaned_reports)
        text = normalize_whitespace(re_serialized)

    metadata = {
        "schema_version": artifact.schema_version,
        "artifact_id": artifact_id,
        "artifact_type": artifact.artifact_type.value
        if hasattr(artifact.artifact_type, "value")
        else artifact.artifact_type,
        "run_id": run_id or "",
        "source_path": artifact.source_path,
        "content_hash": artifact.content_hash,
        "content_length": artifact.content_length,
        "indexable": artifact.indexable,
    }
    return [Document(page_content=text, metadata=metadata)]


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 80,
    run_id: str | None = None,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    result: list[Document] = []
    for chunk in chunks:
        artifact_id = str(chunk.metadata.get("artifact_id", "unknown"))
        content_hash = str(chunk.metadata.get("content_hash", ""))
        chunk_index = len(result)
        chunk.id = f"{artifact_id}:{content_hash}:{chunk_index}"
        chunk.metadata["chunk_index"] = chunk_index
        if run_id is not None:
            chunk.metadata["run_id"] = run_id
        result.append(chunk)
    return result
