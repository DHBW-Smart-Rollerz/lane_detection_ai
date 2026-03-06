#!/bin/bash
#SBATCH --job-name=inf23174_inference_build
#SBATCH --partition=debug
#SBATCH --gres=gpu:user:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-outs/slurm-%j.out

USER="jan"

srun docker build -f $HOME/$USER/lane_detection_ai_training/run.Dockerfile . --network=host -t lane_detection_ai:run