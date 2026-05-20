# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- pytest test framework to LLMOps service

### Changed
- CI workflows now use full requirements.txt instead of stripped-down requirements-ci.txt for better test coverage
- Frontend build uses proper tsconfig without invalid ignoreDeprecations option

### Fixed
- CI test dependencies properly installed via `pip install -r requirements-ci.txt` then `pip install -e .` to ensure all test imports work

---

## [0.9.0] - 2026-04-11

### Added

#### Backend Service (Port 8000)
- JWT-based authentication with registration, login, and role-based access control
- Complete repository and service layers for Users, Patients, MRI Scans, Predictions, and Reports
- Full v1 REST API with all CRUD endpoints
- OCT report ingestion via `import_oct_reports.py`
- ML/LLM client request/response schemas
- SQLAlchemy 2.0 models with asyncpg, alembic migrations support
- Security middleware, CORS, rate limiting, exception handling

#### Frontend Service (Port 3001)
- Next.js 16 with App Router, React 19
- shadcn/ui component library with Tailwind CSS 4
- Dashboard pages: Overview, Patients, Predictions, Reports, Models, Visualise
- Dark theme with teal (#20bdbe) primary, gold (#c8a951) secondary
- Framer-motion animations on all dashboard pages
- JWT-based authentication with token persistence
- Real-time LLMOps operation status display
- Service health monitoring with animated "service down" UI
- Auth error boundaries for login page

#### MLOps Service (Port 8004)
- **Imaging Model**: EfficientNet-B3 (timm) for diabetic retinopathy grading
  - Input: 300x300 RGB fundus images
  - Output: 5-class probability (No DR, Mild, Moderate, Severe, Proliferative DR)
- **Clinical Model**: XGBoost for risk prediction from clinical features
- **OCR Pipeline**: Full OCT report extraction system
  - Tesseract OCR extractor
  - Image preprocessor for Topcon Triton reports
  - Layout-based region detector
  - Regex-based field parser
  - Per-region PNG export and base64 encoding
- MLflow integration with DagsHub
- Evidently reports for data drift detection
- Prometheus metrics endpoint

#### LLMOps Service (Port 8002)
- RAG pipeline with ChromaDB vector store
- Sentence-transformers embeddings (all-MiniLM-L6-v2)
- GitHub AI Inference (gpt-4o) client
- Async job queue for long-running report generation
- Singleton pattern for inference pipeline efficiency
- Configurable chunk sizes and overlap settings

#### Infrastructure
- Multi-stage Dockerfiles for all services
- Docker Compose orchestration
- GitHub Actions CI/CD workflows for each service
- PostgreSQL 16 with async driver

### Changed

#### Architecture
- Initialized monorepo with backend-service, mlops-service
- Added llmops-service for LLM-powered features
- Added frontend-service from next-shadcn-dashboard-starter

#### Frontend
- Replaced initial auth (Clerk) with custom JWT authentication
- Restructured prediction response to include separate left/right eye data with severity field

#### MLOps
- Restructured config system with config.yaml, params.yaml, schema.yaml
- Added configuration manager pattern
- Improved artifact isolation and pipeline logging

#### LLMOps
- Migrated from OpenAI to GitHub AI Inference
- Added configurable LLM provider support
- Refactored CLI into main.py with outer-level wrapper

#### Production Readiness
- All services now use service-relative paths via `Path(__file__).parent`
- Hardcoded `/home/louay` paths removed from `.env` files
- Error handling: try/except for model loading, image processing, LLM API calls
- Security: Removed hardcoded secrets, added auth to protected endpoints
- Performance: Database indexes, N+1 query fixes with selectinload
- Linting: Ruff fixes for unused imports, import order

### Fixed

#### Backend
- CORS middleware missing layer
- MRI scan schema definition
- Auth CORS, token persistence, email/OAuth2 field mapping
- Double-await bug in parallel route pages
- Pyright type corrections
- Duplicate `return patients` in routes
- Missing password validation (minimum 8 characters)
- Catch-all exception handler for unhandled errors

#### Frontend
- Auto-generated next-env.d.ts issues
- Hardcoded port display in models page
- Patients API response formatting
- Hardcoded localhost URLs
- Invalid tsconfig `ignoreDeprecations` option

#### MLOps
- Metrics route using is_file() instead of exists() to prevent IsADirectoryError
- Imaging model logging CPU safety
- Artifacts isolation
- Pyright type corrections
- CUDA availability checks with proper GPU detection
- GPU memory cleanup with torch.cuda.empty_cache()
- Model file existence and size validation
- Training crash handling for image loading and CSV reading

#### LLMOps
- LangChain imports (langchain-huggingface, langchain-chroma)
- /rag/manifest path usage
- Pipeline typing and config
- MockLLMClient bug
- Working directory for imports
- LLM client error handling for rate limits, timeouts, network failures
- Timeout configuration wired to manifest client
- ChromaDB resource cleanup with close() method

#### Security Vulnerabilities
- object-path upgrade to 0.11.5 (prototype pollution)
- flatted upgrade to 3.4.2 (prototype pollution)

#### Docker
- Dockerfile layer caching with deps stage
- .dockerignore to avoid excluding tsconfig
- Frontend build with Bun (matching local environment)
- CI=true and --no-cache for turbopack

### Deprecated

- **Clerk Authentication**: Removed in favor of custom JWT auth
- **OpenAI Client**: Removed from LLMOps, replaced with GitHub AI Inference