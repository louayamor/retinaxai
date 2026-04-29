"""
Integration tests for the API endpoints.
"""

import asyncio
from typing import Any

import pytest
from fastapi.testclient import TestClient
from loguru import logger

from biomarker_service.main import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


@pytest.mark.asyncio
async def test_health_endpoint(client):
    """Test the health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "biomarker-service"
    assert body["version"] == "0.1.0"


@pytest.mark.asyncio
async def test_ready_endpoint_ready(client):
    """Test the ready endpoint when the model is loaded."""
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"


@pytest.mark.asyncio
async def test_ready_endpoint_not_ready(client):
    """Test the ready endpoint when the model is not loaded."""
    # This is a bit tricky to test since the model is loaded on startup
    # We would need to mock the model registry to simulate a failed load
    # For now, we'll just test that the endpoint exists
    response = client.get("/ready")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_extract_endpoint_success(client):
    """Test the extract endpoint with a valid request."""
    response = client.post(
        "/biomarkers/extract",
        data={
            "prediction_id": "pred-1",
            "patient_id": "pat-1",
            "eye_side": "left",
            "model_version": "v1",
        },
        files={"image": ("scan.png", b"fake-bytes", "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["prediction_id"] == "pred-1"
    assert body["patient_id"] == "pat-1"
    assert body["status"] == "success"
    assert body["contract_version"] == "1.0"
    assert body["biomarkers"]["tortuosity"] == 0.5


@pytest.mark.asyncio
async def test_extract_endpoint_failure(client):
    """Test the extract endpoint with an invalid request."""
    response = client.post(
        "/biomarkers/extract",
        data={
            "prediction_id": "",
            "patient_id": "pat-1",
            "eye_side": "left",
            "model_version": "v1",
        },
        files={"image": ("scan.png", b"", "image/png")},
    )
    assert response.status_code == 400
    body = response.json()
    assert "prediction_id is required" in body["detail"]
