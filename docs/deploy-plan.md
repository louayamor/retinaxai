# RetinaXAI Deployment Plan

Goal: Deploy RetinaXAI (diabetic retinopathy grading system) to production.

## Architecture

```
Vercel (us-east)              GCP (us-central1)
┌──────────┐                   ┌────────────────────────────┐
│ Frontend │──api.retinaxai.me─▶│ Cloud Run (3 services)    │
│ (Next.js)│  (HTTPS, CORS)    │  backend  :$PORT           │
│retinaxai.me│                 │  mlops    :$PORT           │
└──────────┘                   │  llmops   :$PORT           │
                               │          │                 │
                               │   Serverless VPC Connector  │
                               │          │                 │
                               │  ┌───────┴────────┐        │
                               │  │ Cloud SQL       │        │
                               │  │ PostgreSQL 15   │        │
                               │  │ (private IP)    │        │
                               │  └────────────────┘        │
                               │  ┌────────────────┐        │
                               │  │ ChromaDB       │        │
                               │  │ GCE e2-micro   │        │
                               │  │ 10.x.x.x:8000  │        │
                               │  └────────────────┘        │
                               │  ┌────────────────────┐    │
                               │  │ GCS Buckets (3)    │    │
                               │  │  uploads/          │    │
                               │  │  models/           │    │
                               │  │  static/           │    │
                               │  └────────────────────┘    │
                               └────────────────────────────┘
```

## Budget

| Resource | Monthly | Notes |
|----------|---------|-------|
| VPC Connector | ~$16 | 2 min instances always-on |
| Cloud SQL db-f1-micro | ~$8 | 10GB SSD, 1 vCPU, 0.6GB RAM |
| ChromaDB e2-micro | ~$6 | 0.25GB memory-optimized |
| Cloud Run + GCS + networking | ~$4 | Per-request billing, cold starts |
| **Total** | **~$34/mo** | |
| **Credit runway** | **~9 months** | $300 GCP credits |

## Approach

**Single codebase, env-flagged.** No separate production branch. Every feature auto-detects
local vs. production via env vars (e.g. `CHROMA_HOST` → remote ChromaDB; unset → local).
Keeps dev and prod in sync with zero backport overhead.

## Running tally (lines changed for production)

| Phase | Files changed | Lines changed |
|-------|---------------|---------------|
| **1** Dockerfiles + port fixes | 6 | ~31 removed/edited |
| **2** ChromaDB HTTP client + systemd | 4 | ~40 added |
| **3** GCS StorageService + model loader | 9 | ~140 added |
| **4** CI/CD workflows | 3 | ~169 added |
| **Total** | **~22 files** | **~380 lines** |

## Phases

| Phase | Description | Est. time |
|-------|-------------|-----------|
| **1** | Dockerfile fixes: `$PORT`, remove `USER appuser`, remove `HEALTHCHECK` | ✅ done |
| **2** | ChromaDB HTTP client + systemd service on e2-micro | ✅ done |
| **3** | GCS StorageService for backend uploads + GCS model loader | ✅ done |
| **4** | CI/CD workflow files (mlops, llmops, backend) | ✅ done |
| **5** | Manual deploy mlops → llmops → backend | ~30 min |
| **6** | Set GitHub secrets from deployed URLs | ~15 min |
| **7** | Custom domain: `api.retinaxai.me` → Cloud Run | ~15 min |
| **Total** | | **~1 hr remaining** |

## What we skip

| Component | Reason |
|-----------|--------|
| **Redis** | Auth degrades to DB lookups; rate-limiting works per-instance. Saves $16/mo. |
| **Kubernetes** | Overkill for 3 containerized services. Cloud Run abstracts it. |
| **Heroku** | Cold starts, cross-platform auth, file upload timeout. Use for staging only. |
| **Self-hosted MLflow** | Already on DagsHub. Cloud SQL too expensive for MLflow metadata. |
| **Self-hosted Ollama** | LLMs served by GitHub Models / NVIDIA APIs — free tier sufficient for RAG. |

## Prerequisites (GCP infra already created via Gemini)

- [x] GCP project with billing enabled
- [x] VPC + Serverless VPC Connector
- [x] Cloud SQL PostgreSQL (private IP)
- [x] GCS buckets: `uploads`, `models`, `static`
- [x] ChromaDB e2-micro VM
- [x] Service account `retinaxai-runner` with GCS + Cloud SQL IAM
- [x] Vercel project + `retinaxai.me` domain

## Environment variables per service

### backend-service (in `app/core/config.py`)
| Var | Source | Notes |
|-----|--------|-------|
| `PORT` | Cloud Run | Automatic |
| `DATABASE_URL` | GitHub secret | Cloud SQL private IP |
| `SECRET_KEY` | GitHub secret | JWT signing key |
| `CORS_ORIGINS` | Hardcoded | `["https://retinaxai.me","https://api.retinaxai.me"]` |
| `GCS_BUCKET_UPLOADS` | Hardcoded | `retinaxai-uploads` |
| `ML_SERVICE_URL` | GitHub secret | mlops-service Cloud Run URL |
| `LLM_SERVICE_URL` | GitHub secret | llmops-service Cloud Run URL |
| `ML_SERVICE_API_KEY` | GitHub secret | |
| `LLM_SERVICE_API_KEY` | GitHub secret | |

### mlops-service (in `settings.py`)
| Var | Source | Notes |
|-----|--------|-------|
| `PORT` | Cloud Run | Automatic, reads via `Field(validation_alias="PORT")` |
| `gcs_model_bucket` | Hardcoded/env | `retinaxai-models` |
| `gcs_model_prefix` | Default | `models/imaging/` |
| `mlflow_tracking_uri` | GitHub secret | DagsHub |
| `dagshub_token` | GitHub secret | |

### llmops-service (in `app/core/config.py`)
| Var | Source | Notes |
|-----|--------|-------|
| `PORT` | Cloud Run | Automatic, reads via `Field(validation_alias="PORT")` |
| `CHROMA_HOST` | Hardcoded | `10.128.0.2` |
| `CHROMA_PORT` | Hardcoded | `8000` |
| `CORS_ORIGINS` | Hardcoded | `["https://retinaxai.me","https://api.retinaxai.me"]` |
| `RAG_EMBEDDINGS_OFFLINE` | Hardcoded | `true` (no HF Hub on Cloud Run) |
| `GITHUB_ACCESS_TOKEN` | GitHub secret | GitHub Models API |
| `NVIDIA_API_KEY` | GitHub secret | NVIDIA API |
| `MLOPS_SERVICE_URL` | GitHub secret | mlops-service Cloud Run URL |
| `BACKEND_SERVICE_URL` | GitHub secret | backend-service Cloud Run URL |
| `LLMOPS_API_KEY` | GitHub secret | |
| `BACKEND_API_KEY` | GitHub secret | |
| `MLFLOW_TRACKING_URI` | GitHub secret | |
| `MLFLOW_TRACKING_USERNAME` | GitHub secret | |
| `MLFLOW_TRACKING_PASSWORD` | GitHub secret | |

## Cloud Run constraints

| Constraint | Fix |
|------------|-----|
| `$PORT` injected dynamically | Uvicorn must read `$PORT` env var (not hardcoded 8000/8002/8004) |
| No `USER appuser` | Cloud Run runs as random UID; remove custom user |
| No Docker `HEALTHCHECK` | Cloud Run uses HTTP probe at `/health` |
| No `--reload` | Remove `reload=True` in production uvicorn calls |
| Ephemeral filesystem | Use GCS for persistent uploads; `/tmp` for model cache |
