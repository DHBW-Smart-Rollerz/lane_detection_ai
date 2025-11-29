#!/bin/bash
#SBATCH --job-name=stop_mlflow_ui
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-outs/slurm-stop-mlflow-%j.out

set -euo pipefail

echo "[INFO] Listing Docker containers before cleanup..."
srun docker ps || true

echo "[INFO] Checking for mlflow_ui container..."
if srun docker ps --format '{{.Names}}' | grep -qw mlflow_ui; then
    echo "[INFO] Stopping mlflow_ui container..."
    srun docker stop mlflow_ui || true
    echo "[INFO] Removing mlflow_ui container..."
    srun docker rm mlflow_ui || true
else
    echo "[INFO] No mlflow_ui container running."
fi

echo "[INFO] Listing Docker containers after cleanup..."
srun docker ps || true
