#!/bin/bash
#SBATCH --job-name=inference_run  # Name des Jobs
#SBATCH --gres=gpu:1                # Anzahl der benötigten GPUs
#SBATCH --cpus-per-task=4           # Anzahl der CPU-Kerne
#SBATCH --mem=24G                   # Arbeitsspeicher
#SBATCH --time=08:00:00             # Maximale Laufzeit
#SBATCH --output=slurm-outs/slurm-%j.out.      # Ausgabeprotokoll (%j = Job-ID)

USER="jan"

# Adjust these paths if your dataset/checkerboard location differs.
DATA_ROOT="/app/dataset/smartrollerz_bev_new"
LABEL_GLOB="$DATA_ROOT/labels/2026-01.xml"
CHECKERBOARD_IMAGE="$DATA_ROOT/labels/chessboard_2026-01_undistorted.png"

# Set to "--old-lens" if needed, otherwise leave empty.
OLD_LENS_FLAG=""

srun docker run --rm \
    --gpus all \
    --volume $HOME/$USER/lane_detection_ai_training/dataset:/app/dataset \
    --volume $HOME/$USER/lane_detection_ai_training/scripts/convert_cvat_bev.py:/app/scripts/convert_cvat_bev.py \
    lane_detection_ai:run \
    bash -lc "ls -a /app/dataset/smartrollerz_bev_new && python3 /app/scripts/convert_cvat_bev.py --label-files $LABEL_GLOB --root $DATA_ROOT --checkerboard-image $CHECKERBOARD_IMAGE $OLD_LENS_FLAG"