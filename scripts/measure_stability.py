#!/usr/bin/env python3
"""Measure how stable the detected lanes are over a sequence of images (no labels needed).

For every model config the images are processed in sorted order. Per lane a 2nd degree
polynomial is fitted (in pixels, like the node) and evaluated at fixed image rows. The
frame-to-frame change of these x positions is the "jump" of a lane: small and rare jumps
mean a calm, stable detection.

Usage (from the package folder, ROS 2 workspace sourced):
    PYTHONPATH=. python3 scripts/measure_stability.py \
        config/model_bev_2026_config.py config/other_model_config.py \
        --images models/data/eval_2025-03-06
"""
import argparse
import glob
import os

import cv2
import numpy as np

from lane_detection_ai.model.model_wrapper import LaneDetectionAiModel

LANE_NAMES = ("left", "center", "right")


def measure(config: str, files: list[str], rows: tuple[int, ...], min_points: int,
            jump_threshold: float) -> None:
    """Run one model over all images and print the stability metrics."""
    model = LaneDetectionAiModel(".", config, log=lambda _: None)
    detected = np.zeros((len(files), len(LANE_NAMES)), dtype=bool)
    xs = np.full((len(files), len(LANE_NAMES), len(rows)), np.nan)

    for i, path in enumerate(files):
        lanes = model.predict(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
        for j, lane in enumerate(lanes):
            if lane is None or len(lane) < min_points:
                continue
            detected[i, j] = True
            coeff = np.polyfit(lane[:, 1], lane[:, 0], 2)
            xs[i, j] = np.polyval(coeff, rows)

    # Change of the x positions between consecutive frames (pixels)
    jumps = np.abs(np.diff(xs, axis=0))

    print(f"== {config} ({len(files)} images)")
    for j, name in enumerate(LANE_NAMES):
        lane_jumps = jumps[:, j][~np.isnan(jumps[:, j])]
        if len(lane_jumps) == 0:
            print(f"  {name:6s} detected {detected[:, j].mean() * 100:5.1f} %   no jumps")
            continue
        print(
            f"  {name:6s} detected {detected[:, j].mean() * 100:5.1f} %   "
            f"jump median {np.median(lane_jumps):5.1f} px   "
            f"p95 {np.percentile(lane_jumps, 95):6.1f} px   "
            f">{jump_threshold:.0f}px jumps {np.mean(lane_jumps > jump_threshold) * 100:4.1f} %"
        )
    print(f"  all 3 lanes in {detected.all(axis=1).mean() * 100:.1f} % of the images")


def main() -> None:
    """Parse the arguments and measure every given model config."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("configs", nargs="+", help="Model configs (relative to the package)")
    parser.add_argument("--images", default="models/data/eval_2025-03-06",
                        help="Folder with sequential images")
    parser.add_argument("--rows", type=int, nargs="+", default=[200, 400, 600],
                        help="Image rows (pixels) where the lane positions are compared")
    parser.add_argument("--min-points", type=int, default=5,
                        help="Minimum points for a lane to count as detected")
    parser.add_argument("--jump-threshold", type=float, default=30.0,
                        help="Jumps above this value (pixels) count as large")
    args = parser.parse_args()

    files = sorted(
        f for ext in ("*.jpg", "*.png") for f in glob.glob(os.path.join(args.images, ext))
    )
    if not files:
        raise FileNotFoundError(f"No images found in {args.images}")

    for config in args.configs:
        measure(config, files, tuple(args.rows), args.min_points, args.jump_threshold)


if __name__ == "__main__":
    main()
