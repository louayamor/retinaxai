import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

BASE_DIR = Path("backend-service")

dirs = [
    "app",
    "app/api",
    "app/api/v1",
    "app/api/v1/routes",
    "app/core",
    "app/core/middleware",
    "app/db",
    "app/db/migrations",
    "app/db/migrations/versions",
    "app/models",
    "app/schemas",
    "app/auth",
    "app/services",
    "app/services/ml_client",
    "app/services/llm_client",
    "app/patients",
    "app/users",
    "app/reports",
    "app/predictions",
    "app/mri_scans",
    "app/oct_reports",
    "app/observability",
    "app/scripts",
    "tests",
    "config",
]

files = [
    "app/__init__.py",
    "app/main.py",
    "app/core/__init__.py",
    "app/core/config.py",
    "app/core/security.py",
    "app/core/logging.py",
    "app/core/exceptions.py",
    "app/core/middleware/__init__.py",
    "app/core/middleware/cors.py",
    "app/core/middleware/request_id.py",
    "app/db/__init__.py",
    "app/db/base.py",
    "app/db/session.py",
    "app/db/migrations/env.py",
    "app/models/__init__.py",
    "app/models/user.py",
    "app/models/patient.py",
    "app/models/prediction.py",
    "app/models/report.py",
    "app/schemas/__init__.py",
    "app/schemas/common.py",
    "app/schemas/user_schema.py",
    "app/schemas/patient_schema.py",
    "app/schemas/prediction_schema.py",
    "app/schemas/report_schema.py",
    "app/schemas/token_schema.py",

    "app/auth/__init__.py",
    "app/auth/jwt_handler.py",
    "app/auth/dependencies.py",
    "app/api/__init__.py",
    "app/api/v1/__init__.py",
    "app/api/v1/router.py",
    "app/api/v1/routes/__init__.py",
    "app/api/v1/routes/auth_routes.py",
    "app/api/v1/routes/patient_routes.py",
    "app/api/v1/routes/predict_routes.py",
    "app/api/v1/routes/report_routes.py",

    "app/services/__init__.py",
    "app/services/ml_client/__init__.py",
    "app/services/ml_client/ml_service.py",
    "app/services/ml_client/schemas.py",
    "app/services/llm_client/__init__.py",
    "app/services/llm_client/llm_service.py",
    "app/services/llm_client/schemas.py",
    "app/patients/__init__.py",
    "app/patients/service.py",
    "app/patients/repository.py",

    "app/users/__init__.py",
    "app/users/service.py",
    "app/users/repository.py",

    "app/predictions/__init__.py",
    "app/predictions/service.py",
    "app/predictions/repository.py",

    "app/reports/__init__.py",
    "app/reports/service.py",
    "app/reports/repository.py",
    "app/observability/__init__.py",
    "app/observability/metrics.py",
    "app/observability/tracing.py",
    "app/scripts/import_oct_reports.py",
    "config/app_config.yaml",
    "config/logging_config.yaml",
    "tests/__init__.py",
    ".env.example",
    "alembic.ini",
]

for d in dirs:
    path = BASE_DIR / d
    path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Ensured directory: {path}")

for f in files:
    path = BASE_DIR / f
    if not path.exists():
        path.touch()
        logging.info(f"Created file:    {path}")
    else:
        logging.info(f"Already exists:  {path}")

logging.info("Scaffold complete.")
logging.info(f"Total directories : {len(dirs)}")
logging.info(f"Total files       : {len(files)}")
