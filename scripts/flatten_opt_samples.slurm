#!/bin/bash
#SBATCH --job-name=flatten_opt_samples_pyxis
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-outs/slurm-flatten-opt-samples-%j.out

# Submit from the repository root (the slurm-outs/ directory must exist).

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$(dirname "${PROJECT_ROOT}")/containers/lane_detection_ai_run.sqsh}"
OPT_SAMPLES_ROOT="${OPT_SAMPLES_ROOT:-${PROJECT_ROOT}/hailo/opt_samples}"

# Set DRY_RUN=1 in sbatch --export to preview actions without moving files.
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$OPT_SAMPLES_ROOT"

echo "[INFO] Using container: $CONTAINER_IMAGE"
echo "[INFO] Flatten root: $OPT_SAMPLES_ROOT"
echo "[INFO] DRY_RUN: $DRY_RUN"

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts="${OPT_SAMPLES_ROOT}:${OPT_SAMPLES_ROOT}" \
  --container-workdir="${OPT_SAMPLES_ROOT}" \
  bash -lc "
    set -euo pipefail
    export OPT_SAMPLES_ROOT='$OPT_SAMPLES_ROOT'
    export DRY_RUN='$DRY_RUN'

    python3 - << 'PY'
import os
import shutil

root = os.environ['OPT_SAMPLES_ROOT']
dry_run = os.environ.get('DRY_RUN', '0') == '1'
img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

if not os.path.isdir(root):
    raise SystemExit(f'[ERROR] Folder not found: {root}')

moved = 0
skipped = 0
collisions = 0

for dirpath, dirnames, filenames in os.walk(root):
    # Never process files already in the base folder.
    if os.path.abspath(dirpath) == os.path.abspath(root):
        continue

    for name in filenames:
        src = os.path.join(dirpath, name)
        ext = os.path.splitext(name)[1].lower()
        if ext not in img_exts:
            skipped += 1
            continue

        dst = os.path.join(root, name)
        if os.path.exists(dst):
            base, ext2 = os.path.splitext(name)
            i = 1
            while True:
                candidate = os.path.join(root, f'{base}_{i:04d}{ext2}')
                if not os.path.exists(candidate):
                    dst = candidate
                    collisions += 1
                    break
                i += 1

        if dry_run:
            print(f'[DRY] {src} -> {dst}')
        else:
            shutil.move(src, dst)
            print(f'[MOVE] {src} -> {dst}')
        moved += 1

print(f'[INFO] Images moved: {moved}')
print(f'[INFO] Name collisions resolved: {collisions}')
print(f'[INFO] Non-image files skipped: {skipped}')
PY

    if [[ \"\$DRY_RUN\" == \"0\" ]]; then
      # Remove now-empty directories under opt_samples (keep root).
      find \"\$OPT_SAMPLES_ROOT\" -mindepth 1 -type d -empty -delete
      echo '[INFO] Removed empty subfolders.'
    fi

    echo '[DONE] Flattening finished.'
  "
