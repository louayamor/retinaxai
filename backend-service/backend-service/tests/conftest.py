from __future__ import annotations

from typing import AsyncGenerator

import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio

from app.main import app
from app.auth.jwt_handler import create_access_token, create_refresh_token


@pytest.fixture
def user_id() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture
def access_token(user_id: uuid.UUID) -> str:
    token, _jti = create_access_token(user_id)
    return token


@pytest.fixture
def refresh_token(user_id: uuid.UUID) -> str:
    return create_refresh_token(user_id)


@pytest.fixture
def auth_db_session() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def auth_session(user_id: uuid.UUID) -> SimpleNamespace:
    return SimpleNamespace(
        revoked=False,
        user_id=user_id,
        expires_at=datetime.now(timezone.utc) + timedelta(days=1),
        access_token_jti=None,
        refresh_token_jti=None,
        token_family=uuid.uuid4(),
    )


@pytest_asyncio.fixture
async def api_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        yield client


@pytest.fixture
def report_generate_payload() -> dict[str, str]:
    return {"prediction_id": str(uuid.uuid4())}


@pytest.fixture
def llm_report_payload() -> dict[str, object]:
    return {
        "patient": {"patient_id": str(uuid.uuid4()), "age": 70, "gender": "M"},
        "prediction": {"label": "mild", "confidence": 0.93},
        "cleaned_summary": "Mild non-proliferative diabetic retinopathy.",
        "raw_ocr_text": "raw ocr text",
        "report_type": "report",
        "language": "en",
        "tone": "clinical",
        "model_name": "gpt-4o",
        "model_version": "2026-04-01",
        "prediction_output": {"label": "mild"},
        "confidence_score": 0.93,
    }


@pytest.fixture
def user_create_payload() -> dict[str, str]:
    return {
        "email": "patient@example.com",
        "username": "patient_one",
        "password": "secret123",
    }


@pytest.fixture
def patient_create_payload() -> dict[str, object]:
    return {
        "first_name": "Louay",
        "last_name": "Amor",
        "age": 70,
        "gender": "M",
        "phone": "+15550000000",
        "address": "123 Main St",
        "medical_record_number": "MRN-001",
        "ocr_patient_id": "ocr-001",
    }


@pytest.fixture
def prediction_request_payload(user_id: uuid.UUID) -> dict[str, object]:
    return {
        "patient_id": str(uuid.uuid4()),
        "mri_scan_id": str(uuid.uuid4()),
        "model_name": "retina-cnn",
        "model_version": "1.0.0",
        "input_payload": {"image_path": "/tmp/scan.png", "user_id": str(user_id)},
    }
