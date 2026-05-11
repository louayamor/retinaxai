# RetinaXAI

## Repository Map
- `backend-service/`, `mlops-service/`, `llmops-service/` are separate Python services.
- `frontend-service/` is the Next.js app.
- Python package roots are nested one level deeper: `backend-service/backend-service/`, `mlops-service/mlops-service/`, `llmops-service/llmops-service/`.
- `infra/infra/docker-compose.yml` is the fastest way to run the full stack.

## Verified Commands
- Backend: `cd backend-service/backend-service && ruff check app/ && pytest tests/ -v --tb=short`
- MLOps: `cd mlops-service && pip install -r requirements.txt && pip install --no-deps -e .`
- MLOps lint: `cd mlops-service/mlops-service && ruff check app/ monitoring/ main.py`
- MLOps tests: `cd mlops-service/mlops-service && pytest tests/ -v --tb=short`
- LLMOps lint: `cd llmops-service/llmops-service && ruff check app/`
- LLMOps tests: `cd llmops-service/llmops-service && pytest tests/ -v --tb=short`
- Frontend install: `cd frontend-service && bun install`
- Frontend lint: `cd frontend-service && bun run lint`
- Frontend test: `cd frontend-service && bun test`
- Frontend build: `cd frontend-service && bun run build`

## Important Quirks
- Backend tests expect a `.env` inside `backend-service/backend-service/`; CI writes `APP_ENV`, `APP_NAME`, `DATABASE_URL`, `SECRET_KEY`, `ML_SERVICE_URL`, `ML_SERVICE_API_KEY`, `LLM_SERVICE_URL`, and `LLM_SERVICE_API_KEY` there.
- MLOps `main.py` changes into `mlops-service/mlops-service/` before running CLI commands.
- LLMOps `main.py` changes into `llmops-service/llmops-service/` before running CLI commands.
- MLOps and LLMOps CI run `ruff` before `pytest`.
- Frontend CI runs `bun run lint` before `bun test`.
- MLOps DVC pipeline is opt-in in CI via `workflow_dispatch` or a commit message containing `[run-dvc]`.

## Docker / Runtime
- Backend image starts with `uvicorn app.main:app --host 0.0.0.0 --port 8000`.
- MLOps image starts with `python main.py serve`.
- LLMOps image starts with `python main.py serve`.
- Frontend image starts with `bun run server.js`.

## Source of Truth
- Prefer `ruff.toml`, service `Dockerfile`s, `Makefile`, and `.github/workflows/*.yml` over README snippets when they differ.
