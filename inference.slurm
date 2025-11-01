#!/bin/bash
#SBATCH --job-name=inference_run  # Name des Jobs
#SBATCH --gres=gpu:1                # Anzahl der benötigten GPUs
#SBATCH --cpus-per-task=4           # Anzahl der CPU-Kerne
#SBATCH --mem=24G                   # Arbeitsspeicher
#SBATCH --time=08:00:00             # Maximale Laufzeit
#SBATCH --output=slurm-outs/slurm-%j.out.      # Ausgabeprotokoll (%j = Job-ID)

srun docker run --rm \
    --gpus all \
    --volume $HOME/lane_detection_ai_training_2/dataset:/app/dataset \
    --volume $HOME/lane_detection_ai_training_2/results:/app/results \
    --volume $HOME/lane_detection_ai_training_2/configs:/app/configs \
    --volume $HOME/lane_detection_ai_training_2/result_inference:/app/result_inference \
    inference_image \
    python3 /app/inference.py \
    --image dataset/smartrollerz_bev/data/eval_2024-07-04/2024-07-04-19-30-57_frame000018.jpg \
    --weights /app/results/20251101_163846_lr_8e-03_b_32/best_model.pth \

