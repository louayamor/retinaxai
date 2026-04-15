from fastapi import APIRouter

from app.api.v1.routes.auth_routes import router as auth_router
from app.api.v1.routes.dashboard_routes import router as dashboard_router
from app.api.v1.routes.explanation_routes import router as explanation_router
from app.api.v1.routes.metrics_routes import router as metrics_router
from app.api.v1.routes.mri_scan_routes import router as mri_scan_router
from app.api.v1.routes.notification_routes import router as notification_router
from app.api.v1.routes.oct_report_routes import router as oct_report_router
from app.api.v1.routes.oct_stats_routes import router as oct_stats_router
from app.api.v1.routes.orchestration_routes import router as orchestration_router
from app.api.v1.routes.patient_routes import router as patient_router
from app.api.v1.routes.predict_routes import router as predict_router
from app.api.v1.routes.report_routes import router as report_router
from app.api.v1.routes.user_routes import router as user_router

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(auth_router)
api_router.include_router(user_router)
api_router.include_router(patient_router)
api_router.include_router(mri_scan_router)
api_router.include_router(predict_router)
api_router.include_router(report_router)
api_router.include_router(oct_stats_router)
api_router.include_router(oct_report_router)
api_router.include_router(metrics_router)
api_router.include_router(dashboard_router)
api_router.include_router(notification_router)
api_router.include_router(orchestration_router)
api_router.include_router(explanation_router)
