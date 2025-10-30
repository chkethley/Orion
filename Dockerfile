# Multi-stage Dockerfile for Orion AGI Development Hub

# Base stage with Python and system dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Development stage
FROM base as development

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml setup.py ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -e ".[dev]"

# Copy application code
COPY . .

# Expose ports
EXPOSE 8000 8888

# Default command
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base as production

# Copy dependency files
COPY pyproject.toml setup.py ./

# Install production dependencies only
RUN pip install --upgrade pip setuptools wheel && \
    pip install .

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/

# Create non-root user
RUN useradd -m -u 1000 orion && \
    chown -R orion:orion /app

USER orion

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production command
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# GPU-enabled stage
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 as gpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install dependencies
COPY pyproject.toml setup.py ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install -e ".[dev]"

# Copy application code
COPY . .

EXPOSE 8000 8888

CMD ["python3", "-m", "src.core.main"]
