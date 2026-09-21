# syntax=docker/dockerfile:1.7
#
# Mega-Quantification GPU image: CUDA + PyTorch (sm_90/sm_120) + ModelOpt +
# llm-compressor + the megaquant CLI. Weights are NOT baked in; mount HF cache.
#
# Default stack (no NGC login):
#   docker compose build
#
# NGC PyTorch (often the most reliable RTX 5090 / sm_120 path):
#   BASE_IMAGE=nvcr.io/nvidia/pytorch:25.08-py3 INSTALL_TORCH=0 docker compose build

ARG BASE_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu24.04
FROM ${BASE_IMAGE}

ARG INSTALL_TORCH=1
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH} \
    HF_HOME=/cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/cache/huggingface/hub \
    TRANSFORMERS_CACHE=/cache/huggingface/transformers \
    HF_DATASETS_CACHE=/cache/huggingface/datasets \
    MEGAQUANT_ROOT=/opt/megaquant \
    PYTHONPATH=/opt/megaquant/src \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    TORCH_CUDA_ARCH_LIST="9.0;10.0;12.0" \
    MODELOPT_NVFP4_TRITON_SWEEP=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-dev \
        python3-pip \
        git \
        git-lfs \
        curl \
        ca-certificates \
        build-essential \
        ninja-build \
        pkg-config \
        libnuma1 \
        libnuma-dev \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && git lfs install --system

WORKDIR /opt/megaquant

# PyTorch CUDA wheel (skipped when the base image is already NGC PyTorch)
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "${INSTALL_TORCH}" = "1" ]; then \
      pip install --upgrade pip setuptools wheel \
      && pip install torch torchvision --index-url "${TORCH_INDEX}"; \
    else \
      pip install --upgrade pip setuptools wheel; \
    fi

COPY docker/requirements-gpu.txt /tmp/requirements-gpu.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements-gpu.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY recipes ./recipes
COPY docs ./docs
COPY docker ./docker

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e ".[dev]" \
    && mkdir -p /cache/huggingface /opt/megaquant/outputs /models \
    && chmod +x /opt/megaquant/docker/entrypoint.sh /opt/megaquant/docker/host-check.sh

WORKDIR /opt/megaquant
ENTRYPOINT ["/opt/megaquant/docker/entrypoint.sh"]
CMD ["plan", "-c", "recipes/qwen3.8-27b-nvfp4-w4a8.yaml"]
