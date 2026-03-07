#!/bin/bash
#SBATCH --job-name=add_user_folder_acl
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-outs/slurm-%j.out

# Important DO READ
# Slurmfile to permissions for one target user on the project folder and subfolders of another users home directory. 
# This is intended to be used for collaborating on one mlflow dashboard with different Linux accounts.

set -euo pipefail

TARGET_USER="inf23054"
TARGET_UID="$(getent passwd "$TARGET_USER" | cut -d: -f3 || true)"

if [[ -z "$TARGET_UID" ]]; then
    echo "[ERROR] Could not resolve UID for $TARGET_USER on host."
    exit 1
fi

if [[ "$(id -un)" != "inf23174" ]]; then
    echo "[WARN] This should be run as inf23174 (owner of /raid/homes/inf23174)."
fi

srun docker run --rm \
        --env TARGET_UID="$TARGET_UID" \
    --volume /raid/homes/inf23174:/raid/homes/inf23174 \
    lane_detection_ai:run \
    bash -lc "
set -e
setfacl -m u:$TARGET_UID:x /raid/homes/inf23174
setfacl -m u:$TARGET_UID:x /raid/homes/inf23174/jan
setfacl -R -m u:$TARGET_UID:rwx /raid/homes/inf23174/jan/lane_detection_ai_training
setfacl -R -d -m u:$TARGET_UID:rwx /raid/homes/inf23174/jan/lane_detection_ai_training
getfacl /raid/homes/inf23174/jan/lane_detection_ai_training | head -n 40
"