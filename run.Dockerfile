# Plan (pseudocode):
# 1) Merge/normalize apt package installs so unzip is always installed in one deterministic layer.
# 2) Include both zip and unzip explicitly with --no-install-recommends.
# 3) Verify unzip exists during build to fail fast if installation did not succeed.
# 4) Keep existing behavior (python symlinks, env vars, requirements install, project copy).

# CUDA-enabled PyTorch base with toolchain for building CUDA extensions
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

# Set working directory
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    QT_QPA_PLATFORM=offscreen \
    DISPLAY=

# Install system deps (Python + OpenCV runtime libs + build tools + zip/unzip)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    build-essential \
    git \
    zip \
    unzip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && command -v unzip

# Copy requirements and install them, but replace DALI with a CUDA 12.x compatible build
# (requirements.txt may pin nvidia-dali-cuda120; we ensure the correct version is installed)
# Also force headless OpenCV in container to avoid Qt/xcb display crashes.
COPY requirements.txt ./requirements.txt
RUN grep -v '^nvidia-dali' requirements.txt \
    | sed -E 's/^sklearn([<>=!~].*)?$/scikit-learn\1/' \
    | sed -E 's/^opencv-python([<>=!~].*)?$/opencv-python-headless\1/' \
    | sed -E 's/^opencv-contrib-python([<>=!~].*)?$/opencv-contrib-python-headless\1/' \
    > requirements.nodali.txt && \
    pip install -r requirements.nodali.txt && \
    pip install nvidia-dali-cuda120

# Copy project files
COPY . /app

# Create output directory
RUN mkdir -p /app/result_inference
