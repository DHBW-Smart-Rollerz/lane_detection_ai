#!/bin/bash
set -euo pipefail

echo '---CHECK: cache stat---'
stat -c '%n %s bytes' /app/dataset/smartrollerz_bev/labels/smartrollerz_anno_cache.json || true

echo '---CHECK: train list length---'
# get the line count as a number (avoid wc output formatting differences)
TRAIN_LIST=/app/dataset/smartrollerz_bev/labels/train_gt.txt
if [ -f "$TRAIN_LIST" ]; then
    count=$(wc -l < "$TRAIN_LIST" 2>/dev/null || echo 0)
else
    count=0
fi
echo "$count $TRAIN_LIST"

if [ "$count" -eq 0 ]; then
    echo "ERROR: in-container train list is empty (0 lines): $TRAIN_LIST"
    echo "Confirm host mount and that the host file is present at: $HOME/jan/lane_detection_ai_training/dataset/smartrollerz_bev/labels/train_gt.txt"
    echo "Aborting to avoid starting training with no data."
    echo
    echo '---DIAG: listing labels dir---'
    ls -la /app/dataset/smartrollerz_bev/labels || true
    echo
    echo '---DIAG: stat train_gt.txt---'
    stat -c '%n -> %N, size=%s, mode=%a, uid=%u,gid=%g' "$TRAIN_LIST" || true
    echo
    echo '---DIAG: readlink -f train_gt.txt---'
    readlink -f "$TRAIN_LIST" || true
    echo
    echo '---DIAG: wc -c and head---'
    wc -c "$TRAIN_LIST" || true
    head -n3 "$TRAIN_LIST" || true
    echo
    echo '---DIAG: file type (first bytes)---'
    # show NULs or binary markers if present
    od -An -t x1 -N 32 "$TRAIN_LIST" || true
    echo
    echo '---DIAG: mount info---'
    # show mount table entries for /app/dataset
    if [ -f /proc/mounts ]; then
        grep '/app/dataset' /proc/mounts || true
    fi
    mount | grep '/app/dataset' || true
    echo
    exit 1
fi

echo '---CHECK: head 200 bytes---'
head -c 200 /app/dataset/smartrollerz_bev/labels/smartrollerz_anno_cache.json || true
echo

echo '---CHECK: tail 200 bytes---'
tail -c 200 /app/dataset/smartrollerz_bev/labels/smartrollerz_anno_cache.json || true
echo

echo '---CHECK: attempt json load and key count---'
python3 - <<'PY'
import json,sys
p='/app/dataset/smartrollerz_bev/labels/smartrollerz_anno_cache.json'
try:
    with open(p,'r',encoding='utf-8') as f:
        d=json.load(f)
    print('JSON OK, keys=',len(d))
except Exception as e:
    print('JSON ERROR:',repr(e))
    try:
        from json import JSONDecodeError
        if isinstance(e, JSONDecodeError):
            print('pos',e.pos,'lineno',e.lineno,'colno',e.colno)
    except Exception:
        pass
    # do not exit here; let training decide
PY

echo '---Launching training---'
python3 /app/train.py configs/smartrollerz_res18_bev.py --dataset Smartrollerz --data_root /app/dataset/smartrollerz_bev --log_path /app/results
