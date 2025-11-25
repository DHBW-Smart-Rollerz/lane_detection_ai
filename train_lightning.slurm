#!/bin/bash
#SBATCH --job-name=patch_gcl_train  # Name des Jobs
#SBATCH --gres=gpu:1                # Anzahl der benötigten GPUs
#SBATCH --cpus-per-task=4           # Anzahl der CPU-Kerne
#SBATCH --mem=24G                   # Arbeitsspeicher
#SBATCH --time=08:00:00             # Maximale Laufzeit
#SBATCH --output=slurm-outs/slurm-%j.out.      # Ausgabeprotokoll (%j = Job-ID)



srun docker run --rm \
        --gpus all \
        --env CUDA_VISIBLE_DEVICES=0 \
        --volume /raid/homes/inf23174/jan/lane_detection_ai_training/dataset:/app/dataset \
        --volume /raid/homes/inf23174/jan/lane_detection_ai_training/results:/app/results \
        --volume /raid/homes/inf23174/jan/lane_detection_ai_training/configs:/app/configs \
        --volume /raid/homes/inf23174/jan/lane_detection_ai_training/lightning:/app/lightning \
        --volume /raid/homes/inf23174/jan/lane_detection_ai_training/train_lightning.py:/app/train_lightning.py \
        lane_detection_ai:training \
        python3 /app/train_lightning.py \
        configs/smartrollerz_res18_bev.py \
        --dataset Smartrollerz \
        --data_root /app/dataset/smartrollerz_bev \
        --log_path /app/results \
        --mlflow_uri file:///app/results/mlruns \
        --mlflow_experiment smartrollerz_lane_detection \
        --eval_during_training True \