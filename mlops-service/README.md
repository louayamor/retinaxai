![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=fff) ![HuggingFace](https://img.shields.io/badge/HuggingFace-F18C00?logo=huggingface&logoColor=fff) ![timm](https://img.shields.io/badge/timm-FF6F61?logoColor=fff) ![MLflow](https://img.shields.io/badge/MLflow-00BFFF?logo=mlflow&logoColor=fff) ![DagsHub](https://img.shields.io/badge/DagsHub-FF7F50?logo=dagshub&logoColor=fff) ![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=fff) ![Evidently](https://img.shields.io/badge/Evidently-6A0DAD?logoColor=fff) ![Prometheus](https://img.shields.io/badge/Prometheus-E6522C?logo=prometheus&logoColor=fff)

**RetinaXAI MLOps Service** is a production-grade MLOps pipeline for **diabetic retinopathy (DR) classification** from retinal fundus images. It handles end-to-end model lifecycle management including data ingestion, preprocessing, training, evaluation, experiment tracking, data drift monitoring, and inter-service communication for the RetinaXAI platform.

---

## Features

- **Data Ingestion**: Load the [Messidor-2](https://huggingface.co/datasets/OctoMed/Messidor2) retinal image dataset directly from HuggingFace Hub.
- **Preprocessing**: Resize, normalize, and export retinal images to structured CSVs for reproducible training.
- **Model Training**: Fine-tune EfficientNet-B3 via [timm](https://github.com/huggingface/pytorch-image-models) with configurable augmentation, early stopping, and cosine annealing.
- **Experiment Tracking**: Log all parameters, metrics, and model artifacts to MLflow tracked on DagsHub.
- **Model Evaluation**: Evaluate on held-out test set using accuracy and quadratic weighted kappa (QWK).
- **Drift Monitoring**: Detect data drift and classification degradation using Evidently reports.
- **Metrics Exposure**: Expose training and evaluation metrics via Prometheus for Grafana dashboards.
- **Inter-Service API**: FastAPI endpoints for health checks, training triggers, and metrics retrieval across RetinaXAI services.

---

## Project Structure

```
mlops-service/
│
├── app/
│   ├── components/
│   │   ├── data_ingestion.py       # HuggingFace Hub ingestion
│   │   ├── preprocessing.py        # Image resizing and CSV export
│   │   ├── model_trainer.py        # EfficientNet-B3 training loop
│   │   ├── model_evaluation.py     # Test set evaluation (accuracy, QWK)
│   │   └── inference.py            # Outbound HTTP client to LLMOps service
│   ├── config/
│   │   └── configuration.py        # ConfigurationManager
│   ├── constants/
│   │   └── __init__.py             # Config file paths
│   ├── entity/
│   │   └── config_entity.py        # Frozen dataclasses for stage configs
│   ├── pipeline/
│   │   ├── training_pipeline.py    # 4-stage training orchestrator
│   │   └── inference_pipeline.py   # Outbound call orchestration
│   └── utils/
│       └── common.py               # YAML/JSON I/O, seed utilities
│
├── config/
│   ├── config.yaml                 # Artifact paths and stage configuration
│   ├── params.yaml                 # Hyperparameters and augmentation settings
│   └── schema.yaml                 # DR label schema and column definitions
│
├── monitoring/
│   ├── evidently_report.py         # Data drift and classification reports
│   └── prometheus_metrics.py       # Prometheus counters, gauges, histograms
│
├── tests/                          # pytest unit and integration tests
├── research/                       # Experimental notebooks
├── artifacts/                      # Generated model and data artifacts
├── logs/                           # Structured JSON logs
│
├── main.py                         # Service entrypoint
├── Dockerfile                      # Multi-stage Docker build
├── requirements.txt
├── setup.py
└── .env.example
```

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/louayamor/retinaxai.git
cd retinaxai/mlops-service
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

4. Copy and configure environment variables:

```bash
cp .env.example .env
```

---

## Configuration

All pipeline behavior is controlled via YAML files under `config/`:

- **config.yaml**: artifact root paths, dataset name, split names, preprocessed CSV paths, model output directory.
- **params.yaml**: epochs, batch size, learning rate, weight decay, scheduler, dropout, augmentation settings, random seed.
- **schema.yaml**: DR grading label definitions (0: No DR → 4: Proliferative DR), column types and constraints.

---

## Usage

### Run Full Training Pipeline

```bash
python main.py
```

Pipeline stages:

1. **Data Ingestion** — streams Messidor-2 from HuggingFace Hub and saves to disk.
2. **Preprocessing** — resizes images to 224×224, exports train/test CSVs.
3. **Model Training** — fine-tunes EfficientNet-B3 with MLflow tracking and early stopping.
4. **Model Evaluation** — evaluates on test set, logs accuracy and QWK to MLflow.

### Run via Orchestration

```bash
python main.py pipeline
```

### Run Individual Stages

```bash
python -c "from app.pipeline.training_pipeline import TrainingPipeline; TrainingPipeline().run_stage_1()"
python -c "from app.pipeline.training_pipeline import TrainingPipeline; TrainingPipeline().run_stage_2()"
python -c "from app.pipeline.training_pipeline import TrainingPipeline; TrainingPipeline().run_stage_3()"
python -c "from app.pipeline.training_pipeline import TrainingPipeline; TrainingPipeline().run_stage_4()"
```

### Run FastAPI Service

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## Experiment Tracking

All runs are tracked on DagsHub via MLflow:

- **Parameters**: model architecture, epochs, batch size, learning rate, scheduler, dropout, seed.
- **Metrics**: train loss, train accuracy, validation accuracy per epoch, best validation accuracy, test accuracy, QWK.
- **Artifacts**: best model checkpoint (`model.pth`), evaluation metrics JSON.

View experiments at: `https://dagshub.com/louayamor/retinaxai`

---

## Monitoring

**Evidently** generates HTML reports for:
- Data drift between reference and current distributions.
- Classification performance degradation over time.

**Prometheus** exposes:
- `training_runs_total` — total training runs triggered.
- `best_val_accuracy` — best validation accuracy from last run.
- `quadratic_weighted_kappa` — QWK from last evaluation.
- `epoch_train_loss` — per-epoch training loss histogram.

---

## DR Grading Labels

| Grade | Label | Description |
|-------|-------|-------------|
| 0 | No DR | No diabetic retinopathy |
| 1 | Mild | Mild non-proliferative DR |
| 2 | Moderate | Moderate non-proliferative DR |
| 3 | Severe | Severe non-proliferative DR |
| 4 | Proliferative DR | Proliferative diabetic retinopathy |

---

## Author

**Louay Amor** – [GitHub](https://github.com/louayamor) | [LinkedIn](https://linkedin.com/in/louayamor)