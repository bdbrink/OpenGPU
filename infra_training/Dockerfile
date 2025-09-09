FROM rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0

# Set working directory
WORKDIR /dojo

# Update package lists and install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for ML and k8s integration
RUN pip install --no-cache-dir \
    transformers>=4.35.0 \
    accelerate \
    datasets \
    tokenizers \
    huggingface-hub \
    wandb \
    kubernetes \
    prometheus-client \
    pyyaml \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    tqdm \
    psutil

# Install additional dependencies for k8s telemetry
RUN pip install --no-cache-dir \
    requests \
    aiohttp \
    asyncio \
    python-dateutil

# Set environment variables for ROCm
ENV HSA_OVERRIDE_GFX_VERSION=10.3.0
ENV HIP_VISIBLE_DEVICES=0

# Create a non-root user (good practice)
RUN useradd -m -s /bin/bash trainer
RUN chown -R trainer:trainer /dojo
USER trainer

# Set Python path
ENV PYTHONPATH=/dojo:$PYTHONPATH

# Default command
CMD ["/bin/bash"]