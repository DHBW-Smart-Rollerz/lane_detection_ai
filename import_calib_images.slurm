#!/bin/bash
#SBATCH --job-name=copy_resize_bev_pyxis
#SBATCH --partition=debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-outs/slurm-copy-resize-bev-%j.out

set -euo pipefail

# --- Config ---
CONTAINER_IMAGE="/raid/homes/inf23054/sanel/containers/lane_detection_ai_run.sqsh"
SRC_ROOT="/raid/homes/inf23054/sanel/lane_detection_ai_training/dataset/smartrollerz_bev/data/rosbags"
DST_ROOT="/raid/homes/inf23054/sanel/lane_detection_ai/hailo/opt_samples"

# Set target size here (or override via: sbatch --export=TARGET_WIDTH=...,TARGET_HEIGHT=... ...)
TARGET_WIDTH="${TARGET_WIDTH:-512}"
TARGET_HEIGHT="${TARGET_HEIGHT:-384}"

mkdir -p "$DST_ROOT"
mkdir -p "$(dirname "$SLURM_OUT_FILE" 2>/dev/null || echo slurm-outs)" || true

echo "[INFO] Using container: $CONTAINER_IMAGE"
echo "[INFO] Source: $SRC_ROOT"
echo "[INFO] Destination: $DST_ROOT"
echo "[INFO] Resize to: ${TARGET_WIDTH}x${TARGET_HEIGHT}"

# Run everything inside Pyxis container
srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts="/raid/homes/inf23054/sanel:/raid/homes/inf23054/sanel" \
  --container-workdir="/raid/homes/inf23054/sanel" \
  bash -lc "
    set -euo pipefail

    SRC_ROOT='$SRC_ROOT'
    DST_ROOT='$DST_ROOT'
    TARGET_WIDTH='$TARGET_WIDTH'
    TARGET_HEIGHT='$TARGET_HEIGHT'

    if [[ ! -d \"\$SRC_ROOT\" ]]; then
      echo \"[ERROR] Source directory not found: \$SRC_ROOT\" >&2
      exit 1
    fi

    echo \"[1/2] Copying files with '_bev' in filename...\"
    copied=0
    while IFS= read -r -d '' src; do
      rel=\"\${src#\${SRC_ROOT}/}\"
      dst=\"\${DST_ROOT}/\${rel}\"
      mkdir -p \"\$(dirname \"\$dst\")\"
      cp -f \"\$src\" \"\$dst\"
      copied=\$((copied + 1))
    done < <(find \"\$SRC_ROOT\" -type f \\( \
      -iname '*_bev*.jpg' -o -iname '*_bev*.jpeg' -o -iname '*_bev*.png' -o \
      -iname '*_bev*.bmp' -o -iname '*_bev*.tif' -o -iname '*_bev*.tiff' \
    \\) -print0)

    echo \"[INFO] Copied \$copied files.\"
    if [[ \"\$copied\" -eq 0 ]]; then
      echo \"[WARN] No matching '_bev' images found.\"
      exit 0
    fi

    echo \"[2/2] Resizing copied images...\"
    export DST_ROOT TARGET_WIDTH TARGET_HEIGHT
    python3 - << 'PY'
import os
import cv2

dst_root = os.environ['DST_ROOT']
w = int(os.environ['TARGET_WIDTH'])
h = int(os.environ['TARGET_HEIGHT'])
exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

ok = 0
fail = 0

for root, _, files in os.walk(dst_root):
    for fn in files:
        ext = os.path.splitext(fn)[1].lower()
        if ext not in exts:
            continue
        p = os.path.join(root, fn)
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f'[WARN] Could not read: {p}')
            fail += 1
            continue
        resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        if not cv2.imwrite(p, resized):
            print(f'[WARN] Could not write: {p}')
            fail += 1
            continue
        ok += 1

print(f'[INFO] Resized images: {ok}')
print(f'[INFO] Failed images:  {fail}')
if ok == 0:
    raise SystemExit('[ERROR] No images were resized.')
PY

    echo \"[DONE] Copy + resize finished.\"
  "