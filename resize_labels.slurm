#!/bin/bash
#SBATCH --job-name=resize_labels
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-outs/slurm-%j.out

set -euo pipefail

USER="sanel"
PROJECT_ROOT="$HOME/$USER/lane_detection_ai_training"

# Input/resize settings
XML_INPUT="/app/dataset/smartrollerz/labels/2025-11.xml"
CURRENT_SIZE="800x640"
TARGET_SIZE="2064x1544"
OUTPUT_XML="/app/dataset/smartrollerz/labels/2025-11_resized.xml"

# Optional debug overlay (set DEBUG=1 to enable)
DEBUG=1
DEBUG_IMAGE="/app/dataset/smartrollerz/data/rosbags/rosbag2_2025-11-11_course/course_frame000035.jpg"
DEBUG_IMAGE_NAME="rosbags/rosbag2_2025-11-11_course/course_frame000035.jpg"
DEBUG_OUTPUT="/app/dataset/smartrollerz/labels/debug_overlay_2025-11.jpg"

CMD="python3 /app/scripts/resize_labels.py \
    --xml ${XML_INPUT} \
    --current-size ${CURRENT_SIZE} \
    --target-size ${TARGET_SIZE} \
    --output ${OUTPUT_XML}"

if [[ "${DEBUG}" == "1" ]]; then
    CMD+=" --debug --debug-image ${DEBUG_IMAGE} --debug-image-name ${DEBUG_IMAGE_NAME} --debug-output ${DEBUG_OUTPUT}"
fi

srun docker run --rm \
    --volume "${PROJECT_ROOT}/dataset:/app/dataset" \
    --volume "${PROJECT_ROOT}/scripts:/app/scripts" \
    lane_detection_ai:run \
    bash -lc "${CMD}"