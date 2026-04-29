"""
Unit tests for the VascXAdapter.
"""

import asyncio
from typing import Any

import pytest
from loguru import logger

from biomarker_service.infrastructure.vascx_adapter import VascXAdapter


class MockVascX:
    def run(self, image: Any) -> dict[str, Any]:
        return {
            "tortuosity": 0.5,
            "avr": 0.7,
            "fractal_dimension": 1.2,
            "vessel_density": 0.3,
            "bifurcation_count": 10,
            "bifurcation_angles": [15.0, 30.0],
            "cre": {"artery_cre": 1.1, "vein_cre": 1.4},
            "raw_feature_vector": [0.5, 0.7, 1.2, 0.3, 10],
        }


@pytest.fixture
def vascx_adapter():
    adapter = VascXAdapter()
    adapter.model = MockVascX()
    return adapter


@pytest.mark.asyncio
async def test_predict_success(vascx_adapter):
    """Test successful VascX inference."""
    result = await vascx_adapter.predict("fake-image")

    assert result["tortuosity"] == 0.5
    assert result["avr"] == 0.7
    assert result["fractal_dimension"] == 1.2
    assert result["vessel_density"] == 0.3
    assert result["bifurcation_count"] == 10
    assert result["bifurcation_angles"] == [15.0, 30.0]
    assert result["cre"] == {"artery_cre": 1.1, "vein_cre": 1.4}
    assert result["raw_feature_vector"] == [0.5, 0.7, 1.2, 0.3, 10]


@pytest.mark.asyncio
async def test_predict_failure(vascx_adapter):
    """Test failed VascX inference."""

    # Override the model to raise an exception
    class FailingMockVascX:
        def run(self, image: Any) -> dict[str, Any]:
            raise Exception("VascX inference failed")

    vascx_adapter.model = FailingMockVascX()

    with pytest.raises(RuntimeError):
        await vascx_adapter.predict("fake-image")
