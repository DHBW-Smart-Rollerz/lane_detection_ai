#!/bin/bash
#SBATCH --job-name=inference_build  # Name des Jobs
#SBATCH --gres=gpu:1                    # Anzahl der benötigten GPUs
#SBATCH --cpus-per-task=4               # Anzahl der CPU-Kerne
#SBATCH --mem=24G                       # Arbeitsspeicher
#SBATCH --time=08:00:00                 # Maximale Laufzeit
#SBATCH --output=slurm-outs/slurm-%j.out.          # Ausgabeprotokoll (%j = Job-ID)

srun docker build -f $HOME/lane_detection_ai_training_2/inference.Dockerfile . --network=host -t inference_image