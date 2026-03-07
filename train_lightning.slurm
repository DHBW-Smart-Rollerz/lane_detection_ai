#!/bin/bash

#SBATCH --job-name=inf23054_patch_gcl_train     
#SBATCH --partition=debug                       
#SBATCH --gres=gpu:user:1                       
#SBATCH --cpus-per-task=24                       
#SBATCH --mem=128G                              
#SBATCH --time=08:00:00                         
#SBATCH --output=slurm-outs/slurm-%j.out        

USER="sanel"

# Umgebungsvariablen für Container-GPU-Support
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Umgebungsvariablen für Cache-Ordner (werden an den Container weitergereicht)
export TORCH_HOME=/app/.cache/torch
export MPLCONFIGDIR=/app/.config/matplotlib
<<<<<<< HEAD

echo "Überprüfe GPU-Sichtbarkeit im Container..."
srun --container-image=dockerd://lane_detection_ai:training nvidia-smi -L

# Eigentlicher Programmaufruf über Pyxis (jetzt ohne --env)
srun --container-image=dockerd://lane_detection_ai:training \
     --container-mounts=/raid/homes/inf23174/$USER/lane_detection_ai_training:/app \
     python3 /app/train_lightning.py \
     configs/smartrollerz_res18.py \
     --dataset Smartrollerz \
     --data_root /app/dataset/smartrollerz \
     --log_path /app/results \
     --mlflow_uri file:///app/results/mlruns \
     --mlflow_experiment smartrollerz_lane_detection \
     --eval_during_training True
=======
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
>>>>>>> 9aff7eb (Enhance Docker and SLURM scripts for improved dataset handling and training setup)
