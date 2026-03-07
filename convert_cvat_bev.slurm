#!/bin/bash
#SBATCH --job-name=inference_run  # Name des Jobs
#SBATCH --gres=gpu:1                # Anzahl der benötigten GPUs
#SBATCH --cpus-per-task=4           # Anzahl der CPU-Kerne
#SBATCH --mem=24G                   # Arbeitsspeicher
#SBATCH --time=08:00:00             # Maximale Laufzeit
#SBATCH --output=slurm-outs/slurm-%j.out.      # Ausgabeprotokoll (%j = Job-ID)

USER="sanel"

# Adjust these paths if your dataset/checkerboard location differs.
DATA_ROOT="/app/dataset/smartrollerz_bev"
LABELS_DIR="$DATA_ROOT/labels"

# Set to "--old-lens" if needed, otherwise leave empty.
OLD_LENS_FLAG=""

srun docker run --rm \
    --gpus all \
    --volume $HOME/$USER/lane_detection_ai_training/dataset:/app/dataset \
    --volume $HOME/$USER/lane_detection_ai_training/scripts/convert_cvat_bev.py:/app/scripts/convert_cvat_bev.py \
    lane_detection_ai:run \
    bash -lc '
set -euo pipefail

for label_file in "'"$LABELS_DIR"'"/*.xml; do
    [ -e "$label_file" ] || continue

    base_name="$(basename "$label_file" .xml)"
    checkerboard_image="'"$LABELS_DIR"'/chessboard_${base_name}_undistorted.png"

    if [ ! -f "$checkerboard_image" ]; then
        echo "Skipping $label_file (missing checkerboard: $checkerboard_image)"
        continue
    fi

    echo "Converting $label_file with $checkerboard_image"
    python3 /app/scripts/convert_cvat_bev.py \
        --label-files "$label_file" \
        --root "'"$DATA_ROOT"'" \
        --checkerboard-image "$checkerboard_image" \
        '"$OLD_LENS_FLAG"'
done
'