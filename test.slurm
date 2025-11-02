#!/bin/bash
#SBATCH --job-name=inference_run  # Name des Jobs
#SBATCH --gres=gpu:1                # Anzahl der benötigten GPUs
#SBATCH --cpus-per-task=4           # Anzahl der CPU-Kerne
#SBATCH --mem=24G                   # Arbeitsspeicher
#SBATCH --time=08:00:00             # Maximale Laufzeit
#SBATCH --output=slurm-outs/slurm-%j.out.      # Ausgabeprotokoll (%j = Job-ID)

USER="USER"

srun docker run --rm \
    --gpus all \
    --volume $HOME/$USER/lane_detection_ai_training/dataset:/app/dataset \
    --volume $HOME/$USER/lane_detection_ai_training/results:/app/results \
    --volume $HOME/$USER/lane_detection_ai_training/configs:/app/configs \
    --volume $HOME/$USER/lane_detection_ai_training/result_inference:/app/result_inference \
    lane_detection_ai:run \
    python3 /app/test.py \
    configs/smartrollerz_res18_bev.py