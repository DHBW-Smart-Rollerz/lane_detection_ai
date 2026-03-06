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

# Configuration - CHANGE THESE:
RESULTS_BASE_DIR="/raid/homes/inf23174/jan/lane_detection_ai_training/results"
TRAINING_RUN_FOLDER="20260306_135352_lr_8e-03_b_32"  # Change this variable for different runs
TRAINING_RUN_DIR="$RESULTS_BASE_DIR/$TRAINING_RUN_FOLDER"
INFERENCE_IMAGE_FOLDER="/raid/homes/inf23174/jan/lane_detection_ai_training/inference_images"
CONFIG_FILE="configs/smartrollerz_res18_bev.py"
OUTPUT_FOLDER="batch_inference_results"
OUTPUT_FOLDER_HOST="/raid/homes/inf23174/jan/lane_detection_ai_training/$OUTPUT_FOLDER"

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