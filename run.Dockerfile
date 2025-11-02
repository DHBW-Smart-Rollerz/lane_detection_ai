# CUDA-enabled PyTorch base with toolchain for building CUDA extensions
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

# Install Python and pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create symlink for python/pip if needed
RUN ln -s /usr/bin/python3 /usr/bin/python || true && \
    ln -s /usr/bin/pip3 /usr/bin/pip || true

# Set working directory
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
RUN grep -v '^nvidia-dali' requirements.txt | sed 's/^sklearn$/scikit-learn/' > requirements.nodali.txt && \
    pip install -r requirements.nodali.txt && \
    pip install nvidia-dali-cuda120

# Copy project files
COPY . /app

# Create output directory
RUN mkdir -p /app/result_inference
