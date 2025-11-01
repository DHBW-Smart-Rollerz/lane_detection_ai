#!/bin/bash
#SBATCH --job-name=training_image_build  # Name des Jobs
#SBATCH --cpus-per-task=4           # Anzahl der CPU-Kerne
#SBATCH --mem=16G                   # Arbeitsspeicher
#SBATCH --time=01:00:00             # Maximale Laufzeit
#SBATCH --output=slurm-outs/slurm-%j.out.      # Ausgabeprotokoll (%j = Job-ID)

srun docker build $HOME/lane_detection_ai_training_2/ --network=host -t training_image