
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-D71F00?style=for-the-badge&logo=sqlalchemy&logoColor=white)
![Alembic](https://img.shields.io/badge/Alembic-6BA539?style=for-the-badge&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Python](https://img.shields.io/badge/Python_3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![JWT](https://img.shields.io/badge/JWT-000000?style=for-the-badge&logo=jsonwebtokens&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-E6522C?style=for-the-badge&logo=prometheus&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)


# RetinaXAI Backend Service

A production-grade REST API for managing patients, MRI scans, ML-based predictions, and LLM-generated reports.

---


## Overview

This service is the core backend for a healthcare application used by doctors to manage patient records, upload MRI scans, run ML-based predictions, and generate LLM-powered diagnostic reports.

```
Doctor → REST API → PostgreSQL
                 → ML Service  (predictions)
                 → LLM Service (reports)
                 → Local Filesystem (MRI scans)
```

---

## Project Structure

```
backend-service/
├── app/
│   ├── api/v1/routes/        # Route handlers
│   ├── auth/                 # JWT handler and dependencies
│   ├── core/                 # Config, logging, exceptions, security
│   ├── db/                   # Engine, session, migrations
│   ├── models/               # SQLAlchemy ORM models
│   ├── schemas/              # Pydantic v2 schemas
│   ├── services/             # ML and LLM HTTP clients
│   ├── patients/             # Patient service and repository
│   ├── predictions/          # Prediction service and repository
│   ├── reports/              # Report service and repository
│   ├── users/                # User service and repository
│   ├── observability/        # Prometheus metrics and tracing
│   └── main.py               # App factory
├── config/                   # YAML config files
├── logs/                     # Structured JSON logs
├── tests/                    # Unit and integration tests
├── .env.example
├── alembic.ini
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Data Model

```
User (doctor)
    │
    ├── manages → Patient
    │               │
    │               ├── MRIScan (left + right scan paths)
    │               ├── Prediction (ML output, links to MRIScan)
    │               └── Report (LLM output, links to Prediction)
```

---

## Getting Started

### Prerequisites

- Python 3.12
- PostgreSQL 16
- pip

### Installation

```bash
git clone https://github.com/louayamor/retinaxai/tree/main/backend-service
cd backend-service

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
```

Edit `.env` and fill in:
- `DATABASE_URL` — your PostgreSQL connection string
- `SECRET_KEY` — generate with `openssl rand -hex 32`
- `ML_SERVICE_URL` / `LLM_SERVICE_URL` — upstream service URLs

### Database Setup

```bash
# Create the database (if not already done)
sudo -u postgres psql -c "CREATE DATABASE healthcare_db;"

# Run migrations
alembic upgrade head
```

### Run

```bash
uvicorn app.main:app --reload
```

API docs available at `http://localhost:8000/docs` when `DEBUG=true`.

---

## Running with Docker

```bash
docker compose up -d
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/auth/register` | Register a new user |
| POST | `/api/v1/auth/login` | Obtain JWT tokens |
| GET | `/api/v1/patients/` | List all patients |
| POST | `/api/v1/patients/` | Create a patient |
| GET | `/api/v1/patients/{id}` | Get a patient |
| POST | `/api/v1/patients/{id}/scans` | Upload MRI scans |
| POST | `/api/v1/predictions/` | Run a prediction |
| GET | `/api/v1/predictions/{id}` | Get a prediction |
| POST | `/api/v1/reports/` | Generate a report |
| GET | `/api/v1/reports/{id}` | Get a report |
| GET | `/api/v1/explanations/{prediction_id}` | Get XAI explanation bundle |
| GET | `/metrics` | Prometheus metrics |
| GET | `/health` | Health check |

---

## XAI Contract Governance

`/api/v1/explanations/{prediction_id}` is the canonical backend contract for persisted XAI results.

- `risk_level` values are canonicalized to: `low`, `moderate`, `high`, `severe`.
- Legacy incoming value `very_high` is normalized to `severe` at persistence boundary.
- `gradcam_explanation` is returned as a nullable object with:
  - `id`
  - `left_eye_explanation`
  - `right_eye_explanation`
  - `highlighted_regions`
  - `model_used`

When evolving this contract, use a backward-compatible two-step rollout:
1. Add/normalize backend fields while preserving old clients.
2. Switch frontend consumers after validation, then remove legacy fallbacks.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `APP_ENV` | Environment name | `development` |
| `DEBUG` | Enable debug mode and docs | `false` |
| `DATABASE_URL` | PostgreSQL async connection string | required |
| `SECRET_KEY` | JWT signing key | required |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Access token TTL | `30` |
| `REFRESH_TOKEN_EXPIRE_DAYS` | Refresh token TTL | `7` |
| `ML_SERVICE_URL` | ML inference service base URL | required |
| `LLM_SERVICE_URL` | LLM service base URL | required |
| `LOG_FORMAT` | `json` or `console` | `json` |
| `CORS_ORIGINS` | Allowed CORS origins (JSON array) | `[]` |

---

## Logging

All logs are structured JSON written to:

- `logs/system/app.log` — general application logs
- `logs/auth/auth.log` — authentication events
- `logs/requests/requests.log` — per-request logs

Set `LOG_FORMAT=console` in `.env` for human-readable output during local development.
