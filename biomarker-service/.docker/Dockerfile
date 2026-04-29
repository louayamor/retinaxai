# Build stage
FROM python:3.12-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for model caching
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    RTNLS_MODEL_RELEASES=/app/.cache/retinalysis-models

# Create cache directories
RUN mkdir -p $HF_HOME $TRANSFORMERS_CACHE $RTNLS_MODEL_RELEASES

# Dependencies stage
FROM base AS deps

# Copy requirements and install Python dependencies
COPY biomarker-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Build stage
FROM deps AS builder

# Copy application code
COPY biomarker-service/ .

# Pre-download model weights at build time for reproducible startup
RUN python -c "from retinalysis_vascx import VascX; VascX()"

# Create non-root user
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup appuser && \
    chown -R appuser:appgroup /app

USER appuser

# Expose port
EXPOSE 8010

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8010/health || exit 1

# Run command
CMD ["biomarker-service"]