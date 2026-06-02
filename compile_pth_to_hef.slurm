#!/bin/bash
#SBATCH --job-name=compile_hailo_hef
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/raid/homes/inf23054/sanel/lane_detection_ai/slurm-outs/slurm-compile-hef-%j.out

set -euo pipefail

CONTAINER_IMAGE="/raid/homes/inf23054/sanel/containers/lane_detection_ai_run.sqsh"
PROJECT_ROOT="/raid/homes/inf23054/sanel/lane_detection_ai"
CONTAINER_NAME="${CONTAINER_NAME:-lane_detection_ai_run_cached}"

# ---- Required inputs (override with sbatch --export=...) ----
MODEL_CONFIG="${MODEL_CONFIG:-config/21.04.2026_config.py}"
PTH_PATH="${PTH_PATH:-models/21.04.2026.pth}"
CALIB_DIR="${CALIB_DIR:-/raid/homes/inf23054/sanel/lane_detection_ai/hailo/opt_samples}"
OUT_DIR="${OUT_DIR:-/raid/homes/inf23054/sanel/lane_detection_ai/hailo/build/21.04.2026/}"
ONNX_OUT="${ONNX_OUT:-${OUT_DIR}/lane_detection_ai.onnx}"

# Directory that contains Hailo DFC wheels from official suite release
# (example content: hailo_sdk_client-*.whl + related suite wheels)
HAILO_WHEELS_DIR="${HAILO_WHEELS_DIR:-${PROJECT_ROOT}/hailo/dfc_wheels}"

# DFC env name expected by compile_onnx_hef.sh
HAILO_DFC_CONDA_ENV="${HAILO_DFC_CONDA_ENV:-hailo_env}"

mkdir -p "$OUT_DIR"
mkdir -p /raid/homes/inf23054/sanel/lane_detection_ai/slurm-outs

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-name="$CONTAINER_NAME" \
  --container-mounts="${PROJECT_ROOT}:${PROJECT_ROOT}" \
  --container-workdir="$PROJECT_ROOT" \
  bash -lc "
    set -euo pipefail

    echo '[INFO] GPU visibility check'
    nvidia-smi -L
    export CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-0}
    export TF_FORCE_GPU_ALLOW_GROWTH=true

    # ---------- fully writable local runtime state ----------
    LOCAL_STATE_ROOT='${PROJECT_ROOT}/.job_local'
    LOCAL_CONDA_ROOT='${PROJECT_ROOT}/.conda_local'
    mkdir -p \"\$LOCAL_STATE_ROOT\" \"\$LOCAL_CONDA_ROOT\"

    export HOME=\"\$LOCAL_STATE_ROOT/home\"
    export XDG_CACHE_HOME=\"\$LOCAL_STATE_ROOT/xdg_cache\"
    export XDG_CONFIG_HOME=\"\$LOCAL_STATE_ROOT/xdg_config\"
    export XDG_DATA_HOME=\"\$LOCAL_STATE_ROOT/xdg_data\"
    mkdir -p \"\$HOME\" \"\$XDG_CACHE_HOME\" \"\$XDG_CONFIG_HOME\" \"\$XDG_DATA_HOME\"
    mkdir -p \"\$XDG_CACHE_HOME/conda/notices\"

    export CONDA_ENVS_PATH=\"\$LOCAL_CONDA_ROOT/envs\"
    export CONDA_PKGS_DIRS=\"\$LOCAL_CONDA_ROOT/pkgs\"
    mkdir -p \"\$CONDA_ENVS_PATH\" \"\$CONDA_PKGS_DIRS\"

    export CONDA_NOTICES_DISABLED=true
    export PIP_CACHE_DIR=\"\$XDG_CACHE_HOME/pip\"
    mkdir -p \"\$PIP_CACHE_DIR\"

    export CONDARC=\"\$LOCAL_STATE_ROOT/condarc\"
    cat > \"\$CONDARC\" <<EOF
envs_dirs:
  - \$CONDA_ENVS_PATH
pkgs_dirs:
  - \$CONDA_PKGS_DIRS
auto_activate_base: false
notify_outdated_conda: false
EOF

    ENV_PREFIX=\"\$CONDA_ENVS_PATH/${HAILO_DFC_CONDA_ENV}\"

    if ! command -v conda >/dev/null 2>&1; then
      echo '[INFO] Installing Miniconda (user-local)'
      MINICONDA_DIR=\"\$LOCAL_CONDA_ROOT/miniconda3\"
      if [[ ! -d \"\$MINICONDA_DIR\" ]]; then
        wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        bash /tmp/miniconda.sh -b -p \"\$MINICONDA_DIR\"
      fi
      export PATH=\"\$MINICONDA_DIR/bin:\$PATH\"
    fi

    source \"\$(conda info --base)/etc/profile.d/conda.sh\"

    if [[ ! -d \"\$ENV_PREFIX\" ]]; then
      conda create -y -p \"\$ENV_PREFIX\" python=3.10 pip
    fi

    conda run -p \"\$ENV_PREFIX\" python -m pip install --upgrade pip setuptools wheel
    conda run -p \"\$ENV_PREFIX\" python -m pip install addict yapf onnx
    conda run -p \"\$ENV_PREFIX\" python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
    conda run -p \"\$ENV_PREFIX\" python -c \"import torch; print('[INFO] torch:', torch.__version__)\"

    conda install -y -p \"\$ENV_PREFIX\" -c conda-forge graphviz pygraphviz
    conda run -p \"\$ENV_PREFIX\" python -c \"import pygraphviz; print('[INFO] pygraphviz OK:', pygraphviz.__version__)\"

    conda run -p \"\$ENV_PREFIX\" python -m pip install --no-deps opencv-python-headless==4.10.0.84
    conda run -p \"\$ENV_PREFIX\" python -m pip install --no-deps numpy==1.26.4 scipy==1.12.0
    conda run -p \"\$ENV_PREFIX\" python -c \"import numpy as np, scipy; print('[INFO] numpy:', np.__version__); print('[INFO] scipy:', scipy.__version__)\"

    export HAILO_DFC_PREFIX="\$ENV_PREFIX"
    export HAILO_DFC_HAILO_BIN="\$HAILO_DFC_PREFIX/bin/hailo"

    have_dfc_cli() {
      [[ -x "\$HAILO_DFC_HAILO_BIN" ]] && "\$HAILO_DFC_HAILO_BIN" help 2>/dev/null | grep -E 'parser|optimize|compiler' >/dev/null
    }

    if ! have_dfc_cli; then
      if [[ ! -d '${HAILO_WHEELS_DIR}' ]]; then
        echo '[ERROR] HAILO_WHEELS_DIR not found: ${HAILO_WHEELS_DIR}'
        exit 2
      fi

      shopt -s nullglob
      wheels=(
        '${HAILO_WHEELS_DIR}'/hailo_dataflow_compiler-*.whl
        '${HAILO_WHEELS_DIR}'/hailo-dataflow-compiler-*.whl
        '${HAILO_WHEELS_DIR}'/hailo_sdk_client-*.whl
        '${HAILO_WHEELS_DIR}'/hailo-sdk-client-*.whl
      )
      shopt -u nullglob

      if [[ \${#wheels[@]} -eq 0 ]]; then
        echo '[ERROR] No Hailo DFC wheels found in ${HAILO_WHEELS_DIR}'
        echo '[INFO] Directory listing:'
        ls -lah '${HAILO_WHEELS_DIR}' || true
        echo '[INFO] Override path with: sbatch --export=HAILO_WHEELS_DIR=/abs/path/to/wheels ...'
        exit 2
      fi

      echo '[INFO] Installing Hailo wheels:'
      printf ' - %s\n' "\${wheels[@]}"
      conda run -p "\$ENV_PREFIX" env PIP_NO_BUILD_ISOLATION=1 PIP_PREFER_BINARY=1 \
        python -m pip install "\${wheels[@]}"
    fi

    if ! have_dfc_cli; then
      echo '[ERROR] Hailo DFC CLI not available after installation.'
      echo '[INFO] Expected binary:' "\$HAILO_DFC_HAILO_BIN"
      exit 2
    fi

    echo '[INFO] Verify DFC CLI'
    "\$HAILO_DFC_HAILO_BIN" help | grep -E 'parser|optimize|compiler' >/dev/null

    conda run -p \"\$ENV_PREFIX\" python '${PROJECT_ROOT}/scripts/convert_pth_onnx.py' \
      --base-path '${PROJECT_ROOT}' \
      --model-config '${MODEL_CONFIG}' \
      --pth '${PTH_PATH}' \
      --onnx-out '${ONNX_OUT}' \
      --opset 18

    chmod +x '${PROJECT_ROOT}/scripts/compile_onnx_hef.sh'
    export OPT_LEVEL=default
    export CALIB_SAMPLES=\${CALIB_SAMPLES:-1024}
    export HAILO_CALIBSET_SIZE=\${HAILO_CALIBSET_SIZE:-\${CALIB_SAMPLES}}
    export NET_NAME=\${NET_NAME:-lane_detection_ai}
    export HW_ARCH=\${HW_ARCH:-hailo8}
    export HAILO_DFC_CONDA_ENV='${HAILO_DFC_CONDA_ENV}'

    '${PROJECT_ROOT}/scripts/compile_onnx_hef.sh' \
      '${ONNX_OUT}' \
      '${CALIB_DIR}' \
      '${OUT_DIR}'

    echo '[DONE] HEF created at: ${OUT_DIR}'
  "