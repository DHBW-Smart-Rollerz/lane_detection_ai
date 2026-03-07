#!/bin/bash

#SBATCH --job-name=inf23174_training_image_build # Zwingende Nomenklatur: USERNAME_JOBNAME 
#SBATCH --partition=debug                        # Gewünschte Partition [cite: 111]
#SBATCH --cpus-per-task=4                        # Anzahl der benötigten CPU-Kerne [cite: 112]
#SBATCH --mem=16G                                # Arbeitsspeicherbedarf [cite: 112]
#SBATCH --time=01:00:00                          # Maximale Laufzeit im Format HH:MM:SS [cite: 112]
#SBATCH --output=slurm-outs/slurm-%j.out         # Speicherort der Ausgabe [cite: 112]

USER="jan"

# Docker-Image bauen
srun docker build $HOME/$USER/lane_detection_ai_training/ --network=host -t lane_detection_ai:training
srun enroot import -o $HOME/$USER/containers/lane_detection_ai_training.sqsh dockerd://lane_detection_ai:training
