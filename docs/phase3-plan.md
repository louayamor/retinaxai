# Phase 3 Plan

## Goal
Add an isolated biomarker extraction service for vascular analysis without coupling it to the core prediction path.

## Workflow
1. `frontend-service` submits a prediction request to `backend-service`.
2. `backend-service` calls `mlops-service` for DR prediction.
3. `mlops-service` returns prediction, GradCAM, regions, and hotspots.
4. `backend-service` persists the prediction.
5. `backend-service` sends raw image bytes to `biomarker-service`.
6. `biomarker-service` computes vascular biomarkers.
7. `backend-service` persists the biomarker result.
8. `backend-service` calls `llmops-service` only after biomarker success.
9. `llmops-service` generates the final explanation using prediction + GradCAM + biomarkers.

## Architecture Decisions
- Keep `biomarker-service` isolated in its own Docker image and Python environment.
- Keep backend-owned persistence for biomarkers.
- Do not include embeddings in the public prediction payload.
- Gate XAI on biomarker completion.
- Pass raw image bytes to the biomarker service.

## Implementation Phases

### 1. Scaffold isolated biomarker service - Done (`#158`)
- Create `biomarker-service/`.
- Add Dockerfile, requirements, and FastAPI app.
- Add `/health` and `/ready` endpoints.
- Add a service wrapper and placeholder extractor interface.

### 2. Define biomarker contract - Done (`#160`)
- Add request/response schemas.
- Include identifiers, status, error handling, and service version.
- Keep the contract explicit and versioned.

### 3. Implement extraction pipeline - Done (`#156`)
- Add VascX integration behind a thin wrapper.
- Keep preprocessing and extraction isolated.
- Return normalized biomarker values.

### 4. Add backend persistence - Done (`#159`)
- Create `vascular_biomarkers` table in backend.
- Add repository/service methods for insert and fetch.

### 5. Wire async orchestration
- Trigger biomarker extraction after prediction succeeds.
- Block XAI until biomarker extraction is complete.
- Propagate failures into the explanation flow.

### 6. Add monitoring
- Expose Prometheus metrics for extraction latency and failures.
- Add structured logging with `prediction_id` correlation.

### 7. Add tests and validation
- Unit tests for service contract and extractor behavior.
- Integration tests for backend orchestration and persistence.
- End-to-end validation through Docker Compose.

## Exit Criteria
- Biomarker service runs independently.
- Backend can call biomarker-service using raw image bytes.
- Biomarker results are persisted by backend.
- XAI is blocked until biomarker success.
- The workflow is covered by tests and documented.
