#!/bin/bash
#SBATCH --job-name=inference_run  # Name des Jobs
#SBATCH --gres=gpu:1                # Anzahl der benötigten GPUs
#SBATCH --cpus-per-task=4           # Anzahl der CPU-Kerne
#SBATCH --mem=24G                   # Arbeitsspeicher
#SBATCH --time=08:00:00             # Maximale Laufzeit
#SBATCH --output=slurm-outs/slurm-%j.out.      # Ausgabeprotokoll (%j = Job-ID)

USER="jan"

srun docker run --rm \
    --gpus all \
    --volume $HOME/$USER/lane_detection_ai_training/dataset:/app/dataset \
    --volume $HOME/$USER/lane_detection_ai_training/scripts/convert_cvat.py:/app/scripts/convert_cvat.py \
    lane_detection_ai:run \
    bash -lc "python3 /app/scripts/convert_cvat.py --label-files /app/dataset/smartrollerz/labels/*.xml --root /app/dataset/smartrollerz"