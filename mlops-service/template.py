import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

BASE_DIR = Path("mlops-service")

dirs = [
    "data",
    "data/models",
    "data/uploads",
    "data/outputs",
    "data/ocr_reports",
    "artifacts",
    "artifacts/model",
    "artifacts/data",
    "artifacts/ocr",
    "artifacts/ocr/output",
    "config",
    "research",
    "app",
    "app/components",
    "app/utils",
    "app/config",
    "app/pipeline",
    "app/entity",
    "app/constants",
    "monitoring",
    "tests",
]

for d in dirs:
    path = BASE_DIR / d
    path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Ensured directory: {path}")

files = [
    "app/__init__.py",
    "app/components/__init__.py",
    "app/components/data_ingestion.py",
    "app/components/preprocessing.py",
    "app/components/model_trainer.py",
    "app/components/model_evaluation.py",
    "app/components/inference.py",
    "app/components/ocr/__init__.py",
    "app/components/ocr/parser.py",
    "app/components/ocr/region_detector.py",
    "app/components/ocr/ocr_pipeline.py",
    "app/utils/__init__.py",
    "app/utils/common.py",
    "app/config/__init__.py",
    "app/config/configuration.py",
    "app/pipeline/__init__.py",
    "app/pipeline/training_pipeline.py",
    "app/pipeline/inference_pipeline.py",
    "app/pipeline/imaging/__init__.py",
    "app/pipeline/clinical/__init__.py",
    "app/pipeline/ocr/__init__.py",
    "app/entity/__init__.py",
    "app/entity/config_entity.py",
    "app/constants/__init__.py",
    "app/constants/constants.py",
    "config/config.yaml",
    "config/params.yaml",
    "config/schema.yaml",
    "monitoring/__init__.py",
    "monitoring/evidently_report.py",
    "monitoring/prometheus_metrics.py",
    "tests/__init__.py",
    "research/trials.ipynb",
    "main.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    ".env.example",
    "README.md",
]

for f in files:
    path = BASE_DIR / f
    if not path.exists():
        path.touch()
        logging.info(f"Created file: {path}")
    else:
        logging.info(f"File exists: {path}")
