#!/bin/bash

#SBATCH --job-name=inf23174_train_lightning 
#SBATCH --partition=debug                       
#SBATCH --gres=gpu:user:1                       
#SBATCH --cpus-per-task=24                       
#SBATCH --mem=128G                              
#SBATCH --time=08:00:00                         
#SBATCH --output=slurm-outs/slurm-%j.out        

set -euo pipefail

USER="sanel"

OWNER_PROJECT_DIR="${OWNER_PROJECT_DIR:-/raid/homes/inf23174/jan/lane_detection_ai_training}"
LOCAL_PROJECT_DIR="${LOCAL_PROJECT_DIR:-$HOME/$USER/lane_detection_ai_training}"
CONTAINER_APP_DIR="${CONTAINER_APP_DIR:-/app}"
OWNER_RESULTS_DIR="${OWNER_RESULTS_DIR:-$OWNER_PROJECT_DIR/results}"

CONTAINER_IMAGE="${CONTAINER_IMAGE:-$HOME/$USER/containers/lane_detection_ai_training.sqsh}"

# Always use your local project as the main mounted project directory.
PROJECT_DIR_TO_MOUNT="$LOCAL_PROJECT_DIR"
echo "[INFO] Main project mount: $PROJECT_DIR_TO_MOUNT -> $CONTAINER_APP_DIR"
echo "[INFO] Shared results mount: $OWNER_RESULTS_DIR -> $CONTAINER_APP_DIR/results"

if [[ ! -d "$OWNER_RESULTS_DIR" ]]; then
  echo "[ERROR] OWNER_RESULTS_DIR does not exist: $OWNER_RESULTS_DIR"
  exit 1
fi

# Umgebungsvariablen für Container-GPU-Support
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Umgebungsvariablen für Cache-Ordner (werden an den Container weitergereicht)
export TORCH_HOME=/app/.cache/torch
export MPLCONFIGDIR=/app/.config/matplotlib
export XDG_CACHE_HOME=/app/.cache

mkdir -p "$PROJECT_DIR_TO_MOUNT/.cache/torch"
mkdir -p "$PROJECT_DIR_TO_MOUNT/.config/matplotlib"

echo "[INFO] Starte Training..."
srun --container-image="$CONTAINER_IMAGE" \
  --container-mounts="$PROJECT_DIR_TO_MOUNT:$CONTAINER_APP_DIR,$OWNER_RESULTS_DIR:$CONTAINER_APP_DIR/results" \
     bash -lc '
       export XDG_CACHE_HOME=/app/.cache &&
       export TORCH_HOME=/app/.cache/torch &&
       export MPLCONFIGDIR=/app/.config/matplotlib &&
       mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" &&
       nvidia-smi -L &&
       python3 /app/train_lightning.py \
         configs/smartrollerz_res18_bev.py \
         --dataset Smartrollerz \
         --data_root /app/dataset/smartrollerz_bev \
         --log_path /app/results \
         --mlflow_uri file:///app/results/mlruns \
         --mlflow_experiment smartrollerz_lane_detection \
         --eval_during_training True
     '