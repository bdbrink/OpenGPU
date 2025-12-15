# ==========================================================
# KubeTrainer Infra Training & Inference Image
# Supports:
#   - Training (K8s Job)
#   - Inference (K8s Deployment)
# ==========================================================

############################
# Base GPU runtime
############################
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

############################
# System dependencies
############################
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    ca-certificates \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

############################
# Create app user
############################
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} kubetrainer \
 && useradd -m -u ${UID} -g ${GID} kubetrainer

############################
# App layout
############################
WORKDIR /app

# Where models, datasets, vectors live (PVC mount)
ENV DATA_DIR=/data
RUN mkdir -p /data && chown kubetrainer:kubetrainer /data

############################
# Python dependencies
############################
COPY requirements.txt .

RUN python3 -m pip install --upgrade pip \
 && pip install -r requirements.txt

############################
# Application code
############################
COPY infra_learning.py .
COPY training_loader.py .
COPY sre_rag_finetuning.py .
COPY interact.py .
COPY k8s_training_integration.py .

# Static assets baked into image (optional)
COPY knowledge_base ./knowledge_base

############################
# Entrypoint
############################
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

############################
# Permissions
############################
RUN chown -R kubetrainer:kubetrainer /app /data

USER kubetrainer

############################
# Default entrypoint
############################
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["serve"]
