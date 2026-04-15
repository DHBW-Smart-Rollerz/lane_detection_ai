#!/bin/bash
#SBATCH --job-name=inf23174_mlflow_ui
#SBATCH --partition=debug
#SBATCH --gres=gpu:0            # UI doesn’t need GPUs
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=99:10:00         # Keep it up to 4h; cancel when done
#SBATCH --output=slurm-outs/slurm-out-mlflow-%j.out

# MLRUNS_DIR=/raid/homes/inf23174/jan/lane_detection_ai_training/results/mlruns
# PORT=5050


# mlflow ui \
#    --backend-store-uri "$MLRUNS_DIR" \
#    --host 0.0.0.0 \
#    --port "$PORT"


# 1. Define Paths
# This is where your training data actually lives on the DGX
HOST_RESULTS_DIR=/raid/homes/inf23174/jan/lane_detection_ai_training/results

# This is where the container will see the data (must match training config)
CONTAINER_RESULTS_DIR=/app/results
MLRUNS_PATH="${CONTAINER_RESULTS_DIR}/mlruns"
USER="jan"
SQSH_PATH="$HOME/$USER/containers/lane_detection_ai_training.sqsh"

PORT=5050

# Required for GPU visibility in Pyxis containers.
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility

# 2. Print connection info for you
# Since this runs on a compute node, you need to know which node it is to tunnel.
NODE_HOSTNAME=$(hostname)
echo "======================================================"
echo "MLflow UI is starting on node: $NODE_HOSTNAME"
echo "To view this on your local laptop, run this in a NEW terminal on your laptop:"
echo ""
# set these (or export SSH_USER / DGX_SERVER before sbatch); defaults shown
SSH_USER="${SSH_USER:-<your_username>}"
DGX_SERVER="${DGX_SERVER:-<dgx_server_address>}"
DGX_SERVER=141.31.112.54

echo "ssh -N -L ${PORT}:${NODE_HOSTNAME}:${PORT} ${SSH_USER}@${DGX_SERVER}"
echo ""
echo "Then open: http://localhost:${PORT}"
echo "======================================================"

# 3. Run with Pyxis
# Uses the training sqsh image produced by docker_build_train.slurm.
srun --container-image="${SQSH_PATH}" \
    --container-mounts="${HOST_RESULTS_DIR}:${CONTAINER_RESULTS_DIR}" \
    --container-workdir=/app \
    mlflow ui \
    --host 0.0.0.0 \
    --port ${PORT} \
    --backend-store-uri ${MLRUNS_PATH}
    