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
    python3 /app/inference.py \
    --image dataset/smartrollerz_bev/data/eval_2024-07-04/2024-07-04-19-30-57_frame000018.jpg \
    --weights /app/results/20251101_163846_lr_8e-03_b_32/best_model.pth \

