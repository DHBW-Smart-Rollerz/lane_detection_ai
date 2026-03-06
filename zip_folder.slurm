#!/bin/bash
#SBATCH --job-name=inference_run  # Name des Jobs
#SBATCH --gres=gpu:1                # Anzahl der benötigten GPUs
#SBATCH --cpus-per-task=32           # Anzahl der CPU-Kerne
#SBATCH --mem=64G                   # Arbeitsspeicher
#SBATCH --time=08:00:00             # Maximale Laufzeit
#SBATCH --output=slurm-outs/slurm-%j.out.      # Ausgabeprotokoll (%j = Job-ID)

USER="jan"

srun docker run --rm \
    --gpus all \
    --volume $HOME/$USER/lane_detection_ai_training/dataset:/app/dataset \
    lane_detection_ai:run \
    bash -lc "cd /app/dataset && unzip smartrollerz.zip -d smartrollerz && unzip smartrollerz_bev.zip -d smartrollerz_bev"