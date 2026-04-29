"""
Thin wrapper for the VascX model inference.

This module provides a clean interface to the retinalysis-vascx library.
"""

import asyncio
from typing import Any, Optional

from loguru import logger


class VascXAdapter:
    """
    Thin adapter for the VascX model.
    """

    def __init__(self):
        self.model: Optional[Any] = None

    async def predict(self, image: Any) -> dict[str, Any]:
        """
        Run VascX inference on the provided image.
        """
        if self.model is None:
            await self._load_model()

        try:
            result = await asyncio.to_thread(self.model.run, image)
            logger.debug("VascX inference completed successfully")
            return result
        except Exception as exc:
            logger.exception("VascX inference failed")
            raise RuntimeError(f"vascx inference failed: {exc}") from exc

    async def _load_model(self) -> None:
        """
        Load the VascX model.
        """
        try:
            from retinalysis_vascx import VascX

            logger.info("Loading VascX model...")
            self.model = VascX()
            logger.info("VascX model loaded successfully")
        except ImportError:
            logger.error("retinalysis-vascx package not installed")
            raise RuntimeError("retinalysis-vascx package not installed") from None
        except Exception as exc:
            logger.exception("Failed to load VascX model")
            raise RuntimeError(f"failed to load VascX model: {exc}") from exc
