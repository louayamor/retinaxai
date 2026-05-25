# Copilot instructions for RetinaXAI

Purpose: give future Copilot sessions concise, repository-specific guidance (build/test/lint, architecture, and conventions).

---

## Quick build / test / lint commands (per service)

General: repo uses per-service virtualenvs for Python services and Bun for frontend. Prefer docker-compose (infra/infra/docker-compose.yml) for full stack.

Docker-compose (recommended)
- cd infra/infra && docker-compose up -d

Backend (backend-service)
- Setup: cd backend-service; python3.12 -m venv .venv && source .venv/bin/activate; pip install -r requirements.txt
- Run dev: uvicorn app.main:app --reload --port 8000
- Tests: pytest
- Single test: pytest path/to/test_file.py::test_name  OR pytest -k test_name
- Lint/format: ruff check .  ; black .  ; pyright (pyrightconfig.json)

MLOps (mlops-service)
- Setup: cd mlops-service; python3.12 -m venv .venv && source .venv/bin/activate; pip install -r requirements.txt
- Run dev: python -m main serve
- Tests: pytest
- Lint/format: ruff / black

LLMOps (llmops-service)
- Setup/run: cd llmops-service; pip install -r requirements.txt; uvicorn llmops-service.app.main:app --reload --port 8002
- Tests: pytest

Frontend (frontend-service)
- Install: cd frontend-service; bun install
- Dev: bun dev (port 3001)
- Build: bun run build
- Lint: bun run lint ; fix: bun run lint:fix ; format: bun run format or prettier --write .
- Single-file lint: npx eslint src/path/to/file --fix

Notes: README lists same commands and required env variables in each service (.env). Use infra/docker-compose for local integration tests.

---

## High-level architecture (short)

- Multi-service monorepo: each top-level folder is a standalone service (backend-service, frontend-service, mlops-service, llmops-service, biomarker-service).
- Backend: FastAPI (app.main), async SQLAlchemy 2.0, asyncpg → PostgreSQL (DATABASE_URL). Ports: backend 8000, mlops 8004, llmops 8002, frontend 3001.
- MLOps: model training/inference endpoints (EfficientNet-B3 for imaging; XGBoost for clinical features). Outputs stored on disk and pushed to DB.
- LLMOps: RAG stack (ChromaDB vectorstore, sentence-transformers embeddings) + LLM provider (GitHub AI Inference) for report generation.
- Frontend: Next.js (App Router) built with Bun, Tailwind, shadcn/ui. Frontend calls backend API and visualizes Grad-CAM and reports.
- Data flow: frontend uploads images → backend stores locally → backend calls MLOps for inference → results persisted to PostgreSQL → backend requests RAG/LLMOps for reports → frontend renders results.

---

## Key repository conventions and patterns

- Service isolation: treat each top-level service folder as an independent app with its own venv and requirements. Changes should be scoped to the service unless cross-service behavior needs updating (e.g., API contract).  
- Async-first Python: code uses async FastAPI endpoints, SQLAlchemy 2.0 async patterns, and async clients. Prefer async functions in services and tests that exercise async flows.
- Pydantic v2 and SQLAlchemy 2.0: data validation and DB models follow these versions—migrations, typing, and model conversions assume Pydantic v2 shapes.
- Entrypoints: backend FastAPI entrypoint is app.main:app; llmops and mlops have similar module entrypoints noted in README. Use those for uvicorn/python -m invocations.
- Env management: each service expects a local .env (see README examples). Do not put secrets in repo; use .env or Docker secrets.
- Tests: pytest is the canonical runner for Python services. Use pytest -k or fully-qualified test nodeids for single-test runs.
- CI: GitHub Actions workflows exist per-service (.github/workflows/*). Follow workflow names when adding actions (backend.yml, mlops.yaml, llmops.yml, frontend.yml).
- Formatting/hooks: Python: Black + Ruff (ruff.toml at repo root). Frontend: Prettier + ESLint + Husky + lint-staged. Commits use Conventional Commits.
- Data directories: expect data and persistence under service-specific data/ directories (backend-service/data, mlops-service/data, llmops-service/data, etc.).

---

## Docs and AI agent files to consult

- Primary docs: README.md and docs/*.md (plan.md, llmops-plan.md, prediction-workflow.md).  
- AGENTS.md is referenced in README but not present in the repo root—check docs/ before assuming agent guidance exists.  
- No CLAUDE.md, .cursorrules, AGENTS.md, .windsurfrules, AIDER_CONVENTIONS.md or other assistant-specific rules were found during scanning.

---

If you want, configure MCP servers (e.g., Playwright for frontend E2E) for this repo. Otherwise, ask to add other guidance (CI shortcuts, test targets) or to include deeper per-service instructions.
