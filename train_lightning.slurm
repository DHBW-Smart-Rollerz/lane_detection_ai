#!/bin/bash

#SBATCH --job-name=inf23174_train_lightning 
#SBATCH --partition=debug                       
#SBATCH --gres=gpu:user:1                       
#SBATCH --cpus-per-task=24                       
#SBATCH --mem=128G                              
#SBATCH --time=08:00:00                         
#SBATCH --output=slurm-outs/slurm-%j.out        

USER="jan"

# Umgebungsvariablen für Container-GPU-Support
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Umgebungsvariablen für Cache-Ordner (werden an den Container weitergereicht)
export TORCH_HOME=/app/.cache/torch
export MPLCONFIGDIR=/app/.config/matplotlib
echo "Prüfe GPU und starte Training..."
srun --container-image=$HOME/$USER/containers/lane_detection_ai_training.sqsh \
     --container-mounts=$HOME/$USER/lane_detection_ai_training:/app \
     bash -lc '
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
