FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# H-CoAtNet — Reproducible Environment (A* / TRIPOD-AI)
# Build: docker build -t hcoatnet:v1 -f Dockerfile .
# Run:   docker run --gpus all -it -v $(pwd):/workspace hcoatnet:v1 bash

ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=42 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && rm -rf /var/lib/apt/lists/*

# Python deps (pinned)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Alternative: H-CoAtNet/requirements.txt if root not present
COPY H-CoAtNet/requirements.txt ./H-CoAtNet/requirements.txt

# Copy codebase (except dataset)
COPY . .

# Verify
RUN python -c "import torch, timm, sklearn; print(f'torch {torch.__version__} timm {timm.__version__} sklearn {sklearn.__version__}')"
RUN nvidia-smi || echo "No GPU at build time (ok for CPU)"

CMD ["bash", "reproduce_all.sh"]
