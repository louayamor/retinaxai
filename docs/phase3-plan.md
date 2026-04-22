# Phase 3 Plan

## Goal
Add an isolated biomarker extraction service for vascular analysis without coupling it to the core prediction path.

## Workflow
1. `frontend-service` submits a prediction request to `backend-service`.
2. `backend-service` calls `mlops-service` for DR prediction.
3. `mlops-service` returns prediction, GradCAM, regions, and hotspots.
4. `backend-service` persists the prediction result immediately (always, even if biomarker extraction fails later).
5. `backend-service` starts event-driven biomarker extraction by calling `biomarker-service` and waiting for completion/failure in the same workflow transaction window.
6. `biomarker-service` computes vascular biomarkers and returns either success payload or structured failure.
7. `backend-service` persists biomarker success or biomarker failure metadata (`status`, standardized `error_code`, and message).
8. `backend-service` calls `llmops-service` only on biomarker success.
9. `llmops-service` generates the final explanation using prediction + GradCAM + biomarkers.
10. On biomarker failure, backend returns partial success: prediction + GradCAM with biomarker failure flag; no LLM explanation call is made.

### Failure-handling Contract
- Prediction persistence is mandatory once ML prediction succeeds, regardless of biomarker outcome.
- Biomarker failure is partial success, not full request failure, unless prediction itself failed earlier.
- llmops invocation is gated strictly on biomarker success.
- Backend returns standardized biomarker failure payload: `biomarker_status="FAILED"`, `error_code`, `error_message`.
- Timeout policy for biomarker extraction: 60s hard timeout per attempt.
- Retry policy: maximum 2 retries after initial failure (3 total attempts), exponential backoff with jitter.
- Backoff schedule: base 1s, multiplier 2, max backoff 8s, jitter +/-20%.
- Circuit breaker: open after 5 consecutive failures, half-open after 60s, close after 2 consecutive successful probes.

### Example API Outcomes
- Success: prediction + biomarkers + XAI explanation.
- Biomarker failure (partial): prediction + GradCAM + `biomarker_status=FAILED`, no XAI explanation payload.
- Full failure: prediction request fails only when prediction generation fails or a non-recoverable backend error occurs before persistence.

### Report Status Compatibility Plan
- Deployment order: apply enum migration before deploying model/API code that emits lowercase status values.
- Compatibility window: keep v1 response behavior mapped to uppercase report status values for legacy clients.
- Migration target: expose lowercase status values in v2 responses and deprecate uppercase mapping after clients migrate.
- Sunset criteria: remove uppercase mapping only after telemetry confirms no active v1 consumers.

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

### 5. Event-driven orchestration
- Trigger biomarker extraction after prediction succeeds.
- Block XAI until biomarker extraction completion event is received.
- Use explicit completion/failure events (or callback/message queue equivalent) so backend can deterministically continue/stop the flow.
- Apply biomarker extraction timeout of 60s per attempt.
- Retry failed biomarker calls up to 2 times with exponential backoff + jitter (1s, 2s, 4s; capped at 8s).
- Apply circuit breaker (open after 5 consecutive failures, retry after 60s half-open window).
- Fallback decision: proceed without XAI as partial success when biomarker step fails after retries/timeouts.
- Propagate failures into explanation flow by persisting failure state and skipping llmops trigger.

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
- Biomarker extraction failures are handled gracefully with explicit partial-success behavior.
- Timeout and retry policies are implemented and tested.
- Circuit-breaker behavior is implemented for repeated biomarker service failures.
- Tests cover failure scenarios: service down, timeout, malformed input.
