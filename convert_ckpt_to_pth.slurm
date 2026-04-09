#!/bin/bash
#SBATCH --job-name=convert_ckpt_to_pth
#SBATCH --partition=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-outs/slurm-convert-ckpt-%j.out

set -euo pipefail

USER="jan"

OWNER_PROJECT_DIR="${OWNER_PROJECT_DIR:-/raid/homes/inf23174/jan/lane_detection_ai_training}"
LOCAL_PROJECT_DIR="${LOCAL_PROJECT_DIR:-$HOME/$USER/lane_detection_ai_training}"
CONTAINER_APP_DIR="${CONTAINER_APP_DIR:-/app}"
OWNER_RESULTS_DIR="${OWNER_RESULTS_DIR:-$OWNER_PROJECT_DIR/results}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$HOME/$USER/containers/lane_detection_ai_training.sqsh}"

# Path relative to project root, usually under results/...
# Example:
# CKPT_REL_PATH="results/optuna/smartrollerz_lane_detection_optuna_bev_2026-03-16/trial_0015/best-epoch=XXX.ckpt"
CKPT_REL_PATH="${CKPT_REL_PATH:-results/optuna/smartrollerz_lane_detection_optuna_bev_2026-03-16/trial_0015/best-epoch=198.ckpt}"

# Optional output path relative to project root.
# If empty, output becomes <same-name>.pth next to ckpt.
OUT_REL_PATH="${OUT_REL_PATH:-}"

PROJECT_DIR_TO_MOUNT="$LOCAL_PROJECT_DIR"
echo "[INFO] Project mount: $PROJECT_DIR_TO_MOUNT -> $CONTAINER_APP_DIR"
echo "[INFO] Results mount: $OWNER_RESULTS_DIR -> $CONTAINER_APP_DIR/results"
echo "[INFO] CKPT_REL_PATH=$CKPT_REL_PATH"
[ -n "$OUT_REL_PATH" ] && echo "[INFO] OUT_REL_PATH=$OUT_REL_PATH"

export CKPT_REL_PATH OUT_REL_PATH CONTAINER_APP_DIR

srun --container-image="$CONTAINER_IMAGE" \
  --container-mounts="$PROJECT_DIR_TO_MOUNT:$CONTAINER_APP_DIR,$OWNER_RESULTS_DIR:$CONTAINER_APP_DIR/results" \
  bash -lc '
    set -euo pipefail
    CKPT_ABS="$CONTAINER_APP_DIR/$CKPT_REL_PATH"

    if [ -n "$OUT_REL_PATH" ]; then
      OUT_ABS="$CONTAINER_APP_DIR/$OUT_REL_PATH"
      python3 /app/scripts/convert_ckpt_to_pth.py --ckpt "$CKPT_ABS" --out "$OUT_ABS" --save_key model
    else
      python3 /app/scripts/convert_ckpt_to_pth.py --ckpt "$CKPT_ABS" --save_key model
    fi
  '
