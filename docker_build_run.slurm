#!/bin/bash
#SBATCH --job-name=inf23174_inference_build
#SBATCH --partition=debug
#SBATCH --gres=gpu:user:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-outs/slurm-%j.out

USER="jan"
SQSH_PATH="$HOME/$USER/containers/lane_detection_ai_run.sqsh"

# Required for GPU visibility in Pyxis containers.
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility

srun docker build -f $HOME/$USER/lane_detection_ai_training/run.Dockerfile . --network=host -t lane_detection_ai:run

if [ -f "$SQSH_PATH" ]; then
	rm -f "$SQSH_PATH"
fi

srun enroot import -o "$SQSH_PATH" dockerd://lane_detection_ai:run