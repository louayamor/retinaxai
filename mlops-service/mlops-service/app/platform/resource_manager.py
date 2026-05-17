from dataclasses import dataclass
from typing import Optional

from loguru import logger

from app.services.orchestration.training_service import get_active_jobs_count
from app.services.monitoring.prometheus_metrics import (
    TRAINING_REJECTIONS_TOTAL,
    TRAINING_SLOTS_USED,
)


@dataclass
class CapacityCheckResult:
    allowed: bool
    reason: Optional[str] = None


class ResourceManager:
    def __init__(self, max_jobs: int, max_jobs_per_pipeline: int):
        self.max_jobs = max_jobs
        self.max_jobs_per_pipeline = max_jobs_per_pipeline

    def can_start(self, pipeline: str) -> CapacityCheckResult:
        total_active, pipeline_active = get_active_jobs_count(pipeline)

        TRAINING_SLOTS_USED.labels(pipeline="all").set(total_active)
        TRAINING_SLOTS_USED.labels(pipeline=pipeline).set(pipeline_active)

        if total_active >= self.max_jobs:
            reason = "capacity_total"
            TRAINING_REJECTIONS_TOTAL.labels(pipeline=pipeline, reason=reason).inc()
            logger.warning("Training rejected: total capacity reached")
            return CapacityCheckResult(allowed=False, reason=reason)

        if pipeline_active >= self.max_jobs_per_pipeline:
            reason = "capacity_pipeline"
            TRAINING_REJECTIONS_TOTAL.labels(pipeline=pipeline, reason=reason).inc()
            logger.warning("Training rejected: pipeline capacity reached")
            return CapacityCheckResult(allowed=False, reason=reason)

        return CapacityCheckResult(allowed=True)
