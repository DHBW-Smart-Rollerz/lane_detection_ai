#!/bin/bash
#SBATCH --job-name=patch_gcl_train  # Name des Jobs
#SBATCH --gres=gpu:1                # Anzahl der benötigten GPUs
#SBATCH --cpus-per-task=4           # Anzahl der CPU-Kerne
#SBATCH --mem=24G                   # Arbeitsspeicher
#SBATCH --time=08:00:00             # Maximale Laufzeit
#SBATCH --output=slurm-outs/slurm-%j.out.      # Ausgabeprotokoll (%j = Job-ID)

srun docker run --rm \
    --gpus all \
    --env CUDA_VISIBLE_DEVICES=1 \
    --volume $HOME/lane_detection_ai_training_2/dataset:/app/dataset \
    --volume $HOME/lane_detection_ai_training_2/results:/app/results \
    --volume $HOME/lane_detection_ai_training_2/configs:/app/configs \
    --volume $HOME/lane_detection_ai_training_2/train.py:/app/train.py \
    training_image \
    python3 /app/train.py \
    configs/smartrollerz_res18_bev.py \
    --dataset Smartrollerz \
    --data_root /app/dataset/smartrollerz_bev \
    --log_path /app/results \