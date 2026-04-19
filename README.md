# RetinaXAI

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=fff" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Next.js-000000?logo=next.js&logoColor=fff" alt="Next.js" />
  <img src="https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=fff" alt="Docker" />
  <img src="https://img.shields.io/badge/PostgreSQL-4169E1?logo=postgresql&logoColor=fff" alt="PostgreSQL" />
  <img src="https://img.shields.io/badge/TailwindCSS-38B2AC?logo=tailwind-css&logoColor=fff" alt="TailwindCSS" />
</p>

---

## Overview

**RetinaXAI** is an AI-powered medical platform for **automated diabetic retinopathy detection and analysis**. It combines computer vision for fundus image analysis with LLM-based medical report generation.

### Key Features

- **Fundus Image Analysis**: Deep learning models (EfficientNet-B3) classify diabetic retinopathy severity
- **Clinical Risk Assessment**: XGBoost models predict DR from structured patient data
- **OCR Pipeline**: Extract structured data from OCT reports
- **LLM Report Generation**: RAG-powered medical reports with contextual insights
- **Real-time Dashboard**: Next.js frontend for patient management and visualization

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│    Frontend     │────▶│     Backend      │────▶│    PostgreSQL   │
│    (Next.js)    │     │    (FastAPI)     │     │    (Database)   │
│    Port 3001    │     │    Port 8000     │     │    Port 5432    │
└─────────────────┘     └────────┬─────────┘     └─────────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  MLOps Service  │     │  LLMOps Service  │     │   Local Data    │
│   (FastAPI)     │     │    (FastAPI)     │     │   Directories   │
│   Port 8004     │     │    Port 8002     │     │                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘

- Imaging: EfficientNet-B3        - RAG with ChromaDB           - backend-service/data/
- Clinical: XGBoost               - GitHub AI Inference         - mlops-service/data/
- OCR: Tesseract + OpenCV         - LangChain                   - llmops-service/data/
```

### Data Flow

1. **Upload**: Frontend uploads fundus scans → Backend stores in local data directory
2. **Predict**: Backend sends base64 images to MLOps → Runs EfficientNet + XGBoost
3. **Store**: Prediction results saved to PostgreSQL
4. **Report**: Backend requests LLM report → LLMOps retrieves RAG context → Generates report
5. **Visualize**: Grad-CAM outputs displayed in frontend dashboard

---

## Quick Start

### Prerequisites

- Python 3.12+
- Bun (for frontend)
- Docker & Docker Compose (optional)
- PostgreSQL 16 (if not using Docker)

### Option 1: Docker Compose (Recommended)

```bash
cd infra/infra
docker-compose up -d
```

Services available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- MLOps API: http://localhost:8004
- LLMOps API: http://localhost:8002

### Option 2: Manual Setup

```bash
# 1. Backend Service
cd backend-service
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn uvicorn app.main:app --reload

# 2. MLOps Service
cd mlops-service
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m main serve

# 3. LLMOps Service
cd llmops-service
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn llmops-service.app.main:app --reload --port 8002

# 4. Frontend
cd frontend-service
bun install
bun dev  # runs on port 3001
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | FastAPI, SQLAlchemy 2.0, asyncpg, Pydantic v2, PyJWT |
| **Frontend** | Next.js 16, React 19, Tailwind CSS 4, shadcn/ui, Zustand |
| **ML/AI** | PyTorch, timm, XGBoost, MLflow, DVC, evidently |
| **LLM/RAG** | LangChain, ChromaDB, sentence-transformers, GitHub AI Inference |
| **Database** | PostgreSQL 16 |
| **Infra** | Docker, Docker Compose, Kubernetes |

---

## Project Structure

```
RetinaXAI/
├── backend-service/          # FastAPI API (port 8000)
│   ├── backend-service/app/
│   │   ├── main.py           # FastAPI entry point
│   │   ├── core/config.py    # Settings
│   │   ├── models/           # SQLAlchemy models
│   │   ├── api/v1/           # API routes
│   │   └── services/         # ML/LLM clients
│   └── Dockerfile
├── frontend-service/         # Next.js app (port 3001)
│   ├── src/
│   │   ├── app/              # Next.js App Router
│   │   ├── components/       # UI components
│   │   └── lib/              # API client, auth
│   └── Dockerfile
├── mlops-service/            # ML training/inference (port 8004)
│   ├── mlops-service/app/
│   │   ├── api/              # Routes (predict, train, health)
│   │   ├── services/         # Inference, training
│   │   └── pipeline/         # Training, OCR pipelines
│   └── Dockerfile
├── llmops-service/          # LLM/RAG (port 8002)
│   ├── llmops-service/app/
│   │   ├── api/routes.py     # Generate, RAG endpoints
│   │   ├── llm/client.py     # GitHub/Ollama/Mock clients
│   │   ├── pipeline/         # Indexing, inference
│   │   └── vectorstore/      # ChromaDB store
│   └── Dockerfile
├── infra/
│   └── infra/
│       ├── docker-compose.yml
│       └── k8s/
└── docs/
    ├── plan.md
    └── llmops-plan.md
```

---

## Configuration

### Required Environment Variables

Create `.env` files in each service directory:

**backend-service/.env:**
```env
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/retinaxai
SECRET_KEY=your-secret-key
ML_SERVICE_URL=http://localhost:8004
LLM_SERVICE_URL=http://localhost:8002
```

**llmops-service/.env:**
```env
LLM_PROVIDER=github
GITHUB_ACCESS_TOKEN=ghp_xxx
RAG_CHROMA_PERSIST_DIRECTORY=./data/chroma
```

**frontend-service/.env:**
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## API Endpoints

### Backend Service (Port 8000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/auth/login` | POST | User login |
| `/api/v1/auth/register` | POST | User registration |
| `/api/v1/patients/` | CRUD | Patient management |
| `/api/v1/predictions/` | POST | Run ML prediction |
| `/api/v1/reports/` | POST | Generate LLM report |

### MLOps Service (Port 8004)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Run inference |
| `/train` | POST | Trigger training |
| `/api/rag/manifest` | GET | Artifact manifest for RAG |

### LLMOps Service (Port 8002)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/generate` | POST | Generate report |
| `/api/rag/reindex` | POST | Rebuild RAG index |
| `/api/rag/status` | GET | RAG index status |

---

## Development

### Running Tests

```bash
# Backend/MLOps/LLMOps
cd <service-directory>
pytest

# Frontend
cd frontend-service
bun run build
```

### Code Style

- **Python**: Black, Ruff
- **TypeScript**: Prettier, ESLint
- **Commits**: Conventional commits format

---

## ML Models

### Diabetic Retinopathy Classification

- **Imaging Model**: EfficientNet-B3 (timm)
  - Input: 224x224 RGB fundus images
  - Output: 5-class severity (No DR, Mild, Moderate, Severe, Proliferative)
  
- **Clinical Model**: XGBoost
  - Input: Structured clinical features
  - Output: Risk score with probability distribution

### RAG System

- **Vector Store**: ChromaDB
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **LLM Provider**: GitHub AI Inference (gpt-4o)
- **Document Types**: OCR reports, clinical metrics, feature importance, imaging metrics

---

## CI/CD

GitHub Actions workflows:

| Workflow | Triggers |
|----------|----------|
| `tests.yml` | PRs to main |
| `backend.yml` | Push/PR to backend-service |
| `mlops.yaml` | Push/PR to mlops-service |
| `llmops.yml` | Push/PR to llmops-service |
| `frontend.yml` | Push/PR to frontend-service |

---

## Documentation

- **AGENTS.md**: Development guide for AI agents
- **docs/plan.md**: Development roadmap
- **docs/llmops-plan.md**: RAG implementation details

---

## License

MIT License

---

## Contact

**Louay Amor** - [GitHub](https://github.com/louayamor) - [Email](mailto:amor.louay20@gmail.com)
