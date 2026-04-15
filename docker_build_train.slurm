#!/bin/bash

#SBATCH --job-name=inf23174_training_image_build # Zwingende Nomenklatur: USERNAME_JOBNAME 
#SBATCH --partition=debug                        # Gewünschte Partition 
#SBATCH --cpus-per-task=4                        # Anzahl der benötigten CPU-Kerne 
#SBATCH --mem=16G                                # Arbeitsspeicherbedarf 
#SBATCH --time=01:00:00                          # Maximale Laufzeit im Format HH:MM:SS
#SBATCH --output=slurm-outs/slurm-%j.out         # Speicherort der Ausgabe
USER="jan"
SQSH_PATH="$HOME/$USER/containers/lane_detection_ai_training.sqsh"

# Required for GPU visibility in Pyxis containers.
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Docker-Image bauen
srun docker build $HOME/$USER/lane_detection_ai_training/ --network=host -t lane_detection_ai:training

if [ -f "$SQSH_PATH" ]; then
	rm -f "$SQSH_PATH"
fi

srun enroot import -o "$SQSH_PATH" dockerd://lane_detection_ai:training
