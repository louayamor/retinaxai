"""
Model registry for the Biomarker Service.

This module manages the lifecycle of the VascX model, ensuring it is loaded only once at startup.
"""

import asyncio
from typing import Any, Optional

from loguru import logger

from biomarker_service.infrastructure.vascx_adapter import VascXAdapter


class VascXRegistry:
    """
    Singleton registry for the VascX model.
    """

    _instance: Optional["VascXRegistry"] = None

    def __new__(cls) -> "VascXRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._adapter: Optional[VascXAdapter] = None
        self._initialized = True

    async def load(self) -> VascXAdapter:
        """
        Load the VascX adapter.
        """
        if self._adapter is None:
            self._adapter = VascXAdapter()
            await self._adapter._load_model()
        return self._adapter

    async def get(self) -> VascXAdapter:
        """
        Get the VascX adapter.
        """
        return await self.load()

    @property
    def is_loaded(self) -> bool:
        """
        Check if the VascX model is loaded.
        """
        return self._adapter is not None


async def initialize_model() -> None:
    """
    Initialize the VascX model on startup.
    """
    registry = VascXRegistry()
    await registry.load()
    logger.info("VascX model initialized successfully")
