#!/bin/bash
#SBATCH --job-name=inf23054_optuna_lightning
#SBATCH --partition=debug
#SBATCH --gres=gpu:user:6
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=100:00:00
#SBATCH --output=slurm-outs/slurm-optuna-%j.out

set -euo pipefail

USER="jan"

OWNER_PROJECT_DIR="${OWNER_PROJECT_DIR:-/raid/homes/inf23174/jan/lane_detection_ai_training}"
LOCAL_PROJECT_DIR="${LOCAL_PROJECT_DIR:-$HOME/$USER/lane_detection_ai_training}"
CONTAINER_APP_DIR="${CONTAINER_APP_DIR:-/app}"
OWNER_RESULTS_DIR="${OWNER_RESULTS_DIR:-$OWNER_PROJECT_DIR/results}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$HOME/$USER/containers/lane_detection_ai_training.sqsh}"

PROJECT_DIR_TO_MOUNT="$LOCAL_PROJECT_DIR"
echo "[INFO] Main project mount: $PROJECT_DIR_TO_MOUNT -> $CONTAINER_APP_DIR"
echo "[INFO] Shared results mount: $OWNER_RESULTS_DIR -> $CONTAINER_APP_DIR/results"

mkdir -p "$OWNER_RESULTS_DIR/optuna"
mkdir -p "$PROJECT_DIR_TO_MOUNT/.cache/torch"
mkdir -p "$PROJECT_DIR_TO_MOUNT/.config/matplotlib"

# Use a unique run tag per submission so Optuna does not resume old trials
# from an existing study DB (which can make a run appear to stop after only
# a few new trials when --optuna_n_trials is already nearly reached).
RUN_TAG="$(date +%Y-%m-%d_%H-%M-%S)"
OPTUNA_BASE="smartrollerz_optuna_bev_res18_aug_v2"
OPTUNA_STUDY_NAME="${OPTUNA_BASE}_${RUN_TAG}"
MLFLOW_EXPERIMENT="${OPTUNA_BASE}_${RUN_TAG}"
OPTUNA_STORAGE="sqlite:////app/results/optuna/${OPTUNA_STUDY_NAME}.db"
export OPTUNA_STUDY_NAME MLFLOW_EXPERIMENT OPTUNA_STORAGE

srun --container-image="$CONTAINER_IMAGE" \
  --container-mounts="$PROJECT_DIR_TO_MOUNT:$CONTAINER_APP_DIR,$OWNER_RESULTS_DIR:$CONTAINER_APP_DIR/results,$PROJECT_DIR_TO_MOUNT/configs:$CONTAINER_APP_DIR/configs" \
  bash -lc '
    export XDG_CACHE_HOME=/app/.cache &&
    export TORCH_HOME=/app/.cache/torch &&
    export MPLCONFIGDIR=/app/.config/matplotlib &&
    mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" &&
    nvidia-smi -L &&
    python3 /app/train_optuna_lightning.py \
      configs/smartrollerz_res18_bev.py \
      --dataset Smartrollerz \
      --data_root /app/dataset/smartrollerz_bev \
      --log_path /app/results \
      --mlflow_uri file:///app/results/mlruns \
      --mlflow_experiment "$MLFLOW_EXPERIMENT" \
      --eval_during_training True \
      --optuna_n_trials 24 \
      --optuna_study_name "$OPTUNA_STUDY_NAME" \
      --optuna_storage "$OPTUNA_STORAGE" \
      --optuna_direction maximize \
      --optuna_sampler tpe \
      --optuna_pruner median
  '

# srun --container-image="$CONTAINER_IMAGE" \
#   --container-mounts="$PROJECT_DIR_TO_MOUNT:$CONTAINER_APP_DIR,$OWNER_RESULTS_DIR:$CONTAINER_APP_DIR/results,$PROJECT_DIR_TO_MOUNT/configs:$CONTAINER_APP_DIR/configs" \
#   bash -lc '
#     export XDG_CACHE_HOME=/app/.cache &&
#     export TORCH_HOME=/app/.cache/torch &&
#     export MPLCONFIGDIR=/app/.config/matplotlib &&
#     mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" &&
#     nvidia-smi -L &&
#     python3 /app/train_optuna_lightning.py \
#       configs/smartrollerz_res18_bev.py \
#       --dataset Smartrollerz \
#       --data_root /app/dataset/smartrollerz_bev \
#       --log_path /app/results \
#       --mlflow_uri file:///app/results/mlruns \
#       --mlflow_experiment smartrollerz_lane_detection_optuna_bev_2026-03-16 \
#       --eval_during_training True \
#       --optuna_n_trials 14 \
#       --optuna_study_name smartrollerz_lane_detection_optuna_bev_2026-03-16 \
#       --optuna_storage sqlite:////app/results/optuna/smartrollerz_optuna_bev_2026-03-16.db \
#       --optuna_direction maximize \
#       --optuna_sampler tpe \
#       --optuna_pruner median
#   '