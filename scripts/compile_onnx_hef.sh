#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./compile_onnx_hef.sh /path/to/model.onnx /path/to/calib_dir /path/to/out_dir

Inputs:
  - model.onnx: exported ONNX model
  - calib_dir: directory of representative images for quantization
  - out_dir: output directory where .har/.hef are written

Environment:
  NET_NAME: network name (default: lane_detection_ai)
  HW_ARCH:  hailo8 | hailo8l | ... (default: hailo8)
  OPT_LEVEL: fast | balanced | max (default: balanced)
  CALIB_SAMPLES: number of calibration samples to use (default: 256)

DFC CLI selection (local):
  HAILO_DFC_HAILO_BIN: absolute path to the DFC `hailo` binary (highest priority)
  HAILO_DFC_CONDA_ENV: conda env name to prefer if active (default: hailo_env)
  HAILO_DFC_PREFIX:    conda env prefix path to use (e.g. /home/.../envs/hailo_env)

Compiler availability:
  - If Hailo Dataflow Compiler (DFC) is installed locally and provides `hailo parser/optimize/compiler`, this script uses it.
  - Otherwise, set HAILO_DFC_DOCKER_IMAGE to a DFC docker image that contains the compiler CLI, e.g.:
      HAILO_DFC_DOCKER_IMAGE=<your_dfc_image> ./compile_onnx_hef.sh model.onnx calib out

Note:
  The `hailo` command from HailoRT (v4.x) includes `run/parse-hef` but does NOT include the compiler.
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 3 ]]; then
  usage
  exit 2
fi

# --------- User inputs ----------
ONNX_PATH="$1"
CALIB_DIR="$2"
OUT_DIR="$3"

NET_NAME="${NET_NAME:-lane_detection_ai}"
HW_ARCH="${HW_ARCH:-hailo8}"

# Optional: reduce effort for faster iteration vs max accuracy
OPT_LEVEL="${OPT_LEVEL:-balanced}"     # e.g. fast|balanced|max
CALIB_SAMPLES="${CALIB_SAMPLES:-256}"  # reasonable starting point
# --------------------------------

if [[ ! -f "${ONNX_PATH}" ]]; then
  echo "ERROR: ONNX file not found: ${ONNX_PATH}" >&2
  exit 2
fi
if [[ ! -d "${CALIB_DIR}" ]]; then
  echo "ERROR: Calibration dir not found: ${CALIB_DIR}" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"

ONNX_PATH_ABS="$(realpath "${ONNX_PATH}")"
CALIB_DIR_ABS="$(realpath "${CALIB_DIR}")"
OUT_DIR_ABS="$(realpath "${OUT_DIR}")"

HAR_PATH="${OUT_DIR_ABS}/${NET_NAME}.har"
HAR_OPT_PATH="${OUT_DIR_ABS}/${NET_NAME}_opt.har"
HEF_PATH="${OUT_DIR_ABS}/${NET_NAME}.hef"

have_dfc_hailo_cli() {
  # $1 must be a hailo binary/path
  local hailo_bin="${1:-}"
  [[ -n "${hailo_bin}" ]] || return 1
  [[ -x "${hailo_bin}" ]] || return 1
  "${hailo_bin}" help 2>/dev/null | grep -qE '(^|\s)parser(\s|$)' || return 1
  "${hailo_bin}" help 2>/dev/null | grep -qE '(^|\s)optimize(\s|$)' || return 1
  "${hailo_bin}" help 2>/dev/null | grep -qE '(^|\s)compiler(\s|$)' || return 1
}

select_local_hailo_dfc_bin() {
  # Priority:
  # 1) Explicit binary path
  # 2) Explicit prefix
  # 3) If the preferred conda env is active, use `command -v hailo`
  # 4) Common default conda env path
  local preferred_env="${HAILO_DFC_CONDA_ENV:-hailo_env}"

  if [[ -n "${HAILO_DFC_HAILO_BIN:-}" ]]; then
    echo "${HAILO_DFC_HAILO_BIN}"
    return 0
  fi

  if [[ -n "${HAILO_DFC_PREFIX:-}" ]]; then
    echo "${HAILO_DFC_PREFIX}/bin/hailo"
    return 0
  fi

  if [[ "${CONDA_DEFAULT_ENV:-}" == "${preferred_env}" ]]; then
    command -v hailo || true
    return 0
  fi

  if [[ -x "${HOME}/anaconda3/envs/${preferred_env}/bin/hailo" ]]; then
    echo "${HOME}/anaconda3/envs/${preferred_env}/bin/hailo"
    return 0
  fi

  echo ""
}

RUNNER=( )
WORKDIR="$(mktemp -d)"
cleanup() {
  rm -rf "${WORKDIR}" || true
}
trap cleanup EXIT

if [[ -n "${HAILO_DFC_DOCKER_IMAGE:-}" ]]; then
  if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker not found, but HAILO_DFC_DOCKER_IMAGE is set." >&2
    exit 2
  fi
  # Run compiler inside container; mount inputs/outputs.
  RUNNER=(docker run --rm \
    --user "$(id -u):$(id -g)" \
    -v "${ONNX_PATH_ABS}:/work/model.onnx:ro" \
    -v "${CALIB_DIR_ABS}:/work/calib:ro" \
    -v "${OUT_DIR_ABS}:/work/out" \
    -w /work \
    "${HAILO_DFC_DOCKER_IMAGE}" bash -lc)
  ONNX_PATH="/work/model.onnx"
  CALIB_DIR="/work/calib"
  HAR_PATH="/work/out/${NET_NAME}.har"
  HAR_OPT_PATH="/work/out/${NET_NAME}_opt.har"
  HEF_PATH="/work/out/${NET_NAME}.hef"
fi

# Prefer local DFC hailo binary if available (common when DFC is installed in a conda env)
HAILO_BIN="$(select_local_hailo_dfc_bin)"
if [[ -n "${HAILO_BIN}" ]] && have_dfc_hailo_cli "${HAILO_BIN}"; then
  # Avoid login shells here (they may trigger unrelated env scripts).
  RUNNER=(bash -c)
elif [[ -n "${HAILO_DFC_DOCKER_IMAGE:-}" ]]; then
  : # Docker runner already configured above
else
  echo "ERROR: Hailo Dataflow Compiler (DFC) CLI not found." >&2
  echo "This machine currently resolves `hailo` to the HailoRT utility (run/parse-hef), which cannot compile ONNX->HEF." >&2
  echo "" >&2
  echo "You said DFC is installed in a conda env; try ONE of:" >&2
  echo "  - Activate the env then re-run:  conda activate ${HAILO_DFC_CONDA_ENV:-hailo_env}" >&2
  echo "  - Or point directly at the env prefix: export HAILO_DFC_PREFIX=/home/.../anaconda3/envs/${HAILO_DFC_CONDA_ENV:-hailo_env}" >&2
  echo "  - Or point directly at the binary: export HAILO_DFC_HAILO_BIN=/home/.../anaconda3/envs/${HAILO_DFC_CONDA_ENV:-hailo_env}/bin/hailo" >&2
  echo "  - Or use Docker: export HAILO_DFC_DOCKER_IMAGE=<your_dfc_image>" >&2
  exit 2
fi

CALIB_NPY="${OUT_DIR_ABS}/${NET_NAME}_calib.npy"
PATCHED_ONNX="${OUT_DIR_ABS}/${NET_NAME}_hailo.onnx"
ONNX_FOR_PARSE="${ONNX_PATH}"

echo "[0/3] Build calibration set (.npy) from ${CALIB_DIR_ABS}"
# DFC expects a .npy file of pre-processed images shaped: (N, H, W, C)
# We'll derive H/W from the ONNX input shape and load RGB uint8 images.
conda_run_prefix=()
if command -v conda >/dev/null 2>&1; then
  # Prefer hailo_env if present; otherwise just use current python.
  conda_run_prefix=(conda run -n "${HAILO_DFC_CONDA_ENV:-hailo_env}")
fi

CALIB_SCRIPT="${WORKDIR}/build_calib_set.py"
cat >"${CALIB_SCRIPT}" <<'PY'
import glob
import os
import sys

import cv2
import numpy as np
import onnx


def main() -> int:
  if len(sys.argv) != 5:
    print("Usage: build_calib_set.py <onnx_path> <calib_dir> <out_npy> <max_samples>", file=sys.stderr)
    return 2

  onnx_path, calib_dir, out_path, max_samples_s = sys.argv[1:]
  max_samples = int(max_samples_s)

  m = onnx.load(onnx_path)
  inp = m.graph.input[0]
  dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]

  # Typical shapes: NCHW (1,3,H,W) or NHWC (1,H,W,3)
  if len(dims) == 4 and dims[1] == 3:
    h, w, c = int(dims[2]), int(dims[3]), 3
  elif len(dims) == 4 and dims[3] == 3:
    h, w, c = int(dims[1]), int(dims[2]), 3
  else:
    raise RuntimeError(f"Unsupported ONNX input shape: {dims}")

  patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
  paths = []
  for pat in patterns:
    paths.extend(glob.glob(os.path.join(calib_dir, pat)))
  paths = sorted(paths)[:max_samples]
  if not paths:
    raise RuntimeError(f"No calibration images found in {calib_dir}")

  arr = np.empty((len(paths), h, w, c), dtype=np.uint8)
  for i, p in enumerate(paths):
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    if img is None:
      raise RuntimeError(f"Failed to read image: {p}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    arr[i] = img

  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  np.save(out_path, arr)
  print(f"Wrote calib set: {out_path}")
  print(f"Calib shape: {arr.shape} dtype={arr.dtype}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
PY

"${conda_run_prefix[@]}" python "${CALIB_SCRIPT}" "${ONNX_PATH_ABS}" "${CALIB_DIR_ABS}" "${CALIB_NPY}" "${CALIB_SAMPLES}"

echo "[0/3] Patch ONNX for Hailo parser (add missing kernel_shape)"

PATCH_SCRIPT="${WORKDIR}/patch_onnx_for_hailo.py"
cat >"${PATCH_SCRIPT}" <<'PY'
import os
import sys

import onnx
from onnx import helper


def main() -> int:
  if len(sys.argv) != 3:
    print("Usage: patch_onnx_for_hailo.py <onnx_in> <onnx_out>", file=sys.stderr)
    return 2

  onnx_in, onnx_out = sys.argv[1:]
  m = onnx.load(onnx_in)
  init_map = {i.name: list(i.dims) for i in m.graph.initializer}

  patched = 0
  for node in m.graph.node:
    if node.op_type not in ("Conv", "ConvTranspose"):
      continue
    if any(a.name == "kernel_shape" for a in node.attribute):
      continue
    if len(node.input) < 2:
      continue
    w_name = node.input[1]
    w_dims = init_map.get(w_name)
    if not w_dims or len(w_dims) < 3:
      continue
    kernel = w_dims[-2:]
    node.attribute.extend([helper.make_attribute("kernel_shape", kernel)])
    patched += 1

  base = os.path.basename(onnx_out)
  location = base + ".data"
  onnx.save_model(
    m,
    onnx_out,
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location=location,
    size_threshold=1024,
  )

  print(f"Patched {patched} Conv/ConvTranspose nodes")
  print(f"Wrote patched ONNX: {onnx_out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
PY

"${conda_run_prefix[@]}" python "${PATCH_SCRIPT}" "${ONNX_PATH_ABS}" "${PATCHED_ONNX}"

ONNX_FOR_PARSE="${PATCHED_ONNX}"

echo "[1/3] Parse ONNX -> HAR"
"${RUNNER[@]}" "\"${HAILO_BIN:-hailo}\" parser onnx \"${ONNX_FOR_PARSE}\" --net-name \"${NET_NAME}\" --hw-arch \"${HW_ARCH}\" --har-path \"${HAR_PATH}\" -y"

echo "[2/3] Optimize (quantize) HAR with calibration"
if [[ "${OPT_LEVEL}" != "balanced" ]]; then
  echo "NOTE: OPT_LEVEL=${OPT_LEVEL} is not mapped 1:1 in DFC; running standard quantization." >&2
fi
"${RUNNER[@]}" "\"${HAILO_BIN:-hailo}\" optimize \"${HAR_PATH}\" --hw-arch \"${HW_ARCH}\" --calib-set-path \"${CALIB_NPY}\" --output-har-path \"${HAR_OPT_PATH}\""

echo "[3/3] Compile optimized HAR -> HEF"
"${RUNNER[@]}" "\"${HAILO_BIN:-hailo}\" compiler \"${HAR_OPT_PATH}\" --hw-arch \"${HW_ARCH}\" --output-dir \"${OUT_DIR_ABS}\""

# Normalize output filename to ${HEF_PATH} if the compiler produced a different name.
if [[ ! -f "${HEF_PATH}" ]]; then
  generated_hef=$(ls -1 "${OUT_DIR_ABS}"/*.hef 2>/dev/null | head -n 1 || true)
  if [[ -n "${generated_hef}" ]]; then
    mv -f "${generated_hef}" "${HEF_PATH}"
  fi
fi

if [[ ! -f "${HEF_PATH}" ]]; then
  echo "ERROR: HEF was not produced in ${OUT_DIR_ABS}" >&2
  echo "Check compiler output above for details." >&2
  exit 1
fi

echo "Done."
echo "HEF: ${HEF_PATH}"