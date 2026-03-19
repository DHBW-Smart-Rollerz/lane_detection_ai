#!/bin/bash
#SBATCH --job-name=inf23054_optuna_lightning
#SBATCH --partition=debug
#SBATCH --gres=gpu:user:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=100:00:00
#SBATCH --output=slurm-outs/slurm-optuna-%j.out

set -euo pipefail

USER="sanel"

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

srun --container-image="$CONTAINER_IMAGE" \
  --container-mounts="$PROJECT_DIR_TO_MOUNT:$CONTAINER_APP_DIR,$OWNER_RESULTS_DIR:$CONTAINER_APP_DIR/results,$PROJECT_DIR_TO_MOUNT/configs:$CONTAINER_APP_DIR/configs" \
  bash -lc '
    export XDG_CACHE_HOME=/app/.cache &&
    export TORCH_HOME=/app/.cache/torch &&
    export MPLCONFIGDIR=/app/.config/matplotlib &&
    mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" &&
    nvidia-smi -L &&
    python3 /app/train_optuna_lightning.py \
      configs/smartrollerz_res18_finetune.py \
      --dataset Smartrollerz \
      --data_root /app/dataset/smartrollerz \
      --log_path /app/results \
      --mlflow_uri file:///app/results/mlruns \
      --mlflow_experiment smartrollerz_optuna_front_finetune_2026-03-18_discriminative_lr \
      --eval_during_training True \
      --optuna_n_trials 16 \
      --optuna_study_name smartrollerz_optuna_front_finetune_2026-03-18_discriminative_lr \
      --optuna_storage sqlite:////app/results/optuna/smartrollerz_optuna_front_finetune_2026-03-18_discriminative_lr.db \
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