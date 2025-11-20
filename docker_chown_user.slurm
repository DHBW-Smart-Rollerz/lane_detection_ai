#!/bin/bash
#SBATCH --job-name=chown_dataset
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-outs/slurm-%j.chown.out

# Configuration - you can edit these defaults directly in the file OR override
# them when submitting the job with sbatch using --export (example below).
#
# Examples:
#  1) Edit the variables below directly, then: sbatch docker_chown_user.slurm
#  2) Submit and override targets without editing the file:
#     sbatch --export=ROOT_DIR=/raid/homes/inf23174,TARGETS="USER inf23174" \
#           docker_chown_user.slurm
#  3) Submit for a single folder:
#     sbatch --export=ROOT_DIR=/raid/homes/inf23174,TARGETS="myfolder" \
#           docker_chown_user.slurm

# Default absolute repo root (can be overridden by environment variable ROOT_DIR)
ROOT_DIR="${ROOT_DIR:-/raid/homes/inf23174}" #folder to change rights

ROOT_DIR="${ROOT_DIR}/jan/lane_detection_ai_training"
# Which directories under ROOT_DIR to fix (space-separated, relative to ROOT_DIR)
# Example: TARGETS="USER inf23174" or TARGETS="dataset results"
TARGETS="${TARGETS:-results slurm-outs results/20251119_225212_lr_8e-03_b_32}" # user to give the rights to

# Image to run as root (use a local image to avoid pulls if possible).
IMAGE="${IMAGE:-lane_detection_ai:training}" 

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