#!/bin/bash
#SBATCH --job-name=batch_inference
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-outs/slurm-inference-%j.out

# ============================================================
# Batch Inference SLURM Script
# ============================================================
# Usage:
#   sbatch batch_inference.slurm
#
# Before running, update these variables:
# - TRAINING_RUN_DIR: The results folder from your training (e.g., results/20251129_143527_lr_8e-03_b_32)
# - INFERENCE_IMAGE_FOLDER: Folder with Inference images
# - CONFIG_FILE: Your config file
# ============================================================

USER="jan"

# Configuration - CHANGE THESE:
RESULTS_BASE_DIR="/raid/homes/inf23174/$USER/lane_detection_ai_training/results"
TRAINING_RUN_FOLDER="20260307_154142_lr_8e-03_b_32"  # Change this variable for different runs
TRAINING_RUN_DIR="$RESULTS_BASE_DIR/$TRAINING_RUN_FOLDER"

# Toggle BEV mode (true/false). This controls config and inference image folder.
USE_BEV=true

if [ "$USE_BEV" = true ]; then
    INFERENCE_IMAGE_FOLDER="/raid/homes/inf23174/$USER/lane_detection_ai_training/inference_images_bev"
    CONFIG_FILE="configs/smartrollerz_res18_bev.py"
else
    INFERENCE_IMAGE_FOLDER="/raid/homes/inf23174/$USER/lane_detection_ai_training/inference_images"
    CONFIG_FILE="configs/smartrollerz_res18.py"
fi

OUTPUT_FOLDER="batch_inference_results"
OUTPUT_FOLDER_HOST="/raid/homes/inf23174/$USER/lane_detection_ai_training/$OUTPUT_FOLDER"

# Create output folder on host
mkdir -p $OUTPUT_FOLDER_HOST

# Find the best checkpoint (highest epoch number)
BEST_CHECKPOINT=$(ls -1 ${TRAINING_RUN_DIR}/best-epoch*.ckpt 2>/dev/null | head -1)

if [ -z "$BEST_CHECKPOINT" ]; then
    echo "ERROR: No best checkpoint found in ${TRAINING_RUN_DIR}"
    echo "Looking for: best-epoch*.ckpt"
    ls -la ${TRAINING_RUN_DIR}/*.ckpt 2>/dev/null || echo "No checkpoints found"
    exit 1
fi

# Extract just the filename for use inside Docker
BEST_CHECKPOINT_BASENAME=$(basename ${BEST_CHECKPOINT})
TRAINING_RUN_BASENAME=$(basename ${TRAINING_RUN_DIR})

echo "=================================================="
echo "Batch Inference Setup"
echo "=================================================="
echo "BEV mode: $USE_BEV"
echo "Config: $CONFIG_FILE"
echo "Checkpoint (host): $BEST_CHECKPOINT"
echo "Checkpoint (docker): /app/results/${TRAINING_RUN_BASENAME}/${BEST_CHECKPOINT_BASENAME}"
echo "Inference images: $INFERENCE_IMAGE_FOLDER"
echo "Output folder (host): $OUTPUT_FOLDER_HOST"
echo "Output folder (docker): /app/$OUTPUT_FOLDER"
echo "=================================================="

USER="jan"

# Run Docker with batch inference - mount all necessary directories
srun docker run --rm \
    --gpus all \
    --env CUDA_VISIBLE_DEVICES=0 \
    --volume $HOME/$USER/lane_detection_ai_training/configs:/app/configs \
    --volume $HOME/$USER/lane_detection_ai_training/results:/app/results \
    --volume $INFERENCE_IMAGE_FOLDER:/app/inference_images \
    --volume $OUTPUT_FOLDER_HOST:/app/$OUTPUT_FOLDER \
    --volume $HOME/$USER/lane_detection_ai_training:/app \
    lane_detection_ai:run \
    python3 /app/batch_inference.py \
    --config /app/$CONFIG_FILE \
    --weights /app/results/${TRAINING_RUN_BASENAME}/${BEST_CHECKPOINT_BASENAME} \
    --image_folder /app/inference_images \
    --output_folder /app/$OUTPUT_FOLDER \
    --device cuda
    
echo ""
echo "=================================================="
echo "Inference complete!"
echo "Results saved to: $OUTPUT_FOLDER_HOST"
echo "=================================================="
ls -la $OUTPUT_FOLDER_HOST

USER="sanel" # default user for paths, can be overridden by environment variable USER

# Default absolute repo root (can be overridden by environment variable ROOT_DIR)
ROOT_DIR="${ROOT_DIR:-/raid/homes/inf23054}" #folder to change rights

ROOT_DIR="${ROOT_DIR}/$USER/lane_detection_ai_training"
# Which directories under ROOT_DIR to fix (space-separated, relative to ROOT_DIR)
# Example: TARGETS="USER inf23174" or TARGETS="dataset results"
TARGETS="${TARGETS:-results slurm-outs inference_images dataset batch_inference_results}" # user to give the rights to

# Image to run as root (use a local image to avoid pulls if possible).
IMAGE="${IMAGE:-lane_detection_ai:run}" 

# UID/GID to chown to (submitting user)
HOST_UID=$(id -u)
HOST_GID=$(id -g)

echo "ROOT_DIR=${ROOT_DIR}"
echo "TARGETS=${TARGETS}"
echo "Using IMAGE=${IMAGE}"
echo "Will chown to UID:GID=${HOST_UID}:${HOST_GID}"

# Dry run listing before changes
echo "Contents (before):"
for p in $TARGETS; do
  echo "---- ${ROOT_DIR}/${p} ----"
  ls -la "${ROOT_DIR}/${p}" || true
done

# Run a short root container to chown (no host sudo required)
srun docker run --rm \
  -v "${ROOT_DIR}":/data \
  --user root \
  "${IMAGE}" \
  bash -lc "for p in ${TARGETS}; do
              if [ -e /data/\$p ]; then
                echo chown -R ${HOST_UID}:${HOST_GID} /data/\$p;
                chown -R ${HOST_UID}:${HOST_GID} /data/\$p || true;
                chmod -R u+rwX /data/\$p || true;
              else
                echo '/data/'\$p 'does not exist';
              fi;
            done"

echo "Contents (after):"
for p in $TARGETS; do
  echo "---- ${ROOT_DIR}/${p} ----"
  ls -la "${ROOT_DIR}/${p}" || true
done