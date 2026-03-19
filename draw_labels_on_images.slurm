#!/bin/bash
#SBATCH --job-name=draw_lane_labels
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-outs/slurm-%j.out

set -euo pipefail

USER="sanel"
PROJECT_ROOT="$HOME/$USER/lane_detection_ai_training"

# ------------------------------------------------------------
# Configure inputs/outputs here
# ------------------------------------------------------------
LABELS_JSON="/app/dataset/smartrollerz/labels/smartrollerz_anno_cache.json"
IMAGES_ROOT="/app/dataset/smartrollerz/"

# Optional fine-tuning
LINE_THICKNESS=2
POINT_RADIUS=2
FOLDER_PREFIX="data/rosbags/rosbag2_2025-11-11_course"  # Render one folder/bag
# IMAGE_NAME="data/rosbags/2023-05-24-17-46-49/2023-05-24-17-46-49_frame000110.png"  # Optional single image

FOLDER_TAG="$(basename "${FOLDER_PREFIX}")"
OUTPUT_DIR="/app/dataset/smartrollerz/debug_overlays/${FOLDER_TAG}"

CMD="python3 /app/scripts/draw_labels_on_images.py \
	--labels-json ${LABELS_JSON} \
	--images-root ${IMAGES_ROOT} \
	--output-dir ${OUTPUT_DIR} \
	--line-thickness ${LINE_THICKNESS} \
	--point-radius ${POINT_RADIUS}"

if [[ -n "${IMAGE_NAME:-}" ]]; then
	CMD+=" --image-name ${IMAGE_NAME}"
fi

if [[ -n "${FOLDER_PREFIX:-}" ]]; then
	CMD+=" --folder-prefix ${FOLDER_PREFIX}"
fi

srun docker run --rm \
	--volume "${PROJECT_ROOT}/dataset:/app/dataset" \
	--volume "${PROJECT_ROOT}/scripts:/app/scripts" \
	lane_detection_ai:run \
	bash -lc "${CMD}"
