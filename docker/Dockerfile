# =============================================================================
# MMBT Fact-Checking Dockerfile
# Multi-stage build with CUDA support
# =============================================================================

# Stage 1: Base image with CUDA
FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Stage 2: Builder (installa dipendenze)
FROM base AS builder

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip wheel setuptools \
    && pip install --no-cache-dir -r requirements.txt

# Stage 3: Production
FROM builder AS production

# Copia tutto il progetto
COPY . .

# Configura PYTHONPATH per includere lib/
ENV PYTHONPATH="/app:/app/lib:${PYTHONPATH}"

# Crea directories necessarie
RUN mkdir -p outputs logs models/checkpoints data/results

# Default command
CMD ["python", "main.py", "-c", "configs/configurazione.conf"]

# Stage 4: Development (include tool di sviluppo)
FROM production AS development

RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    isort \
    flake8 \
    mypy \
    ipython \
    jupyter

CMD ["bash"]

# Stage 5: Testing
FROM production AS testing

RUN pip install --no-cache-dir pytest pytest-cov

CMD ["pytest", "tests/", "-v"]
