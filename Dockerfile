# CUDA-enabled PyTorch base with toolchain for building CUDA extensions
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# System deps (OpenCV runtime libs + build tools for CUDA extension)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them, but replace DALI with a CUDA 12.x compatible build
# (requirements.txt may pin nvidia-dali-cuda120; we ensure the correct version is installed)
COPY requirements.txt ./requirements.txt
RUN grep -v '^nvidia-dali' requirements.txt | sed 's/^sklearn$/scikit-learn/' > requirements.nodali.txt \
 && pip install -r requirements.nodali.txt \
 && pip install nvidia-dali-cuda120

# Copy project files
COPY . .

# Build and install the custom CUDA extension used by data pipeline (my_interp)
# Set TORCH_CUDA_ARCH_LIST to target common GPU architectures (Pascal through Ada Lovelace)
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"
RUN pip install -v ./my_interp

# Default mount points
VOLUME ["/data", "/output"]
