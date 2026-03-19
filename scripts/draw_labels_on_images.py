#!/usr/bin/env python3
"""Draw cached lane labels from JSON onto images and save overlays.

Example:
	python scripts/draw_labels_on_images.py \
		--labels-json dataset/smartrollerz/labels/smartrollerz_anno_cache.json \
		--images-root dataset/smartrollerz \
		--output-dir dataset/smartrollerz/debug_overlays
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
from tqdm import tqdm


Point = Tuple[float, float]


def parse_cached_lane(raw_lane: Sequence[Sequence[float]]) -> List[Point]:
	"""Parse one cached lane from JSON as a list of (x, y) floats."""
	pts: List[Point] = []
	for pair in raw_lane:
		if not isinstance(pair, (list, tuple)) or len(pair) != 2:
			continue
		try:
			x = float(pair[0])
			y = float(pair[1])
		except (TypeError, ValueError):
			continue
		pts.append((x, y))
	return pts


def resolve_image_path(image_name: str, images_root: str) -> Optional[str]:
	"""Resolve image path from cache key against images root."""
	if os.path.isabs(image_name) and os.path.isfile(image_name):
		return image_name

	candidates = [
		os.path.join(images_root, image_name),
		os.path.join(images_root, image_name.lstrip("/")),
	]
	for candidate in candidates:
		if os.path.isfile(candidate):
			return candidate

	return None


def ensure_parent(path: str) -> None:
	parent = os.path.dirname(os.path.abspath(path))
	if parent:
		os.makedirs(parent, exist_ok=True)


def draw_labels_on_images(
	labels_json: str,
	images_root: str,
	output_dir: str,
	thickness: int,
	point_radius: int,
	only_image_name: Optional[str] = None,
	folder_prefix: Optional[str] = None,
) -> Tuple[int, int]:
	"""Draw lane annotations from cache JSON onto their corresponding images."""
	palette: Dict[int, Tuple[int, int, int]] = {
		0: (0, 255, 0),      # left
		1: (0, 255, 255),    # center
		2: (0, 128, 255),    # right
	}

	if not os.path.isfile(labels_json):
		raise RuntimeError(f"Labels JSON not found: {labels_json}")

	with open(labels_json, "r") as f:
		cached_labels = json.load(f)

	if not isinstance(cached_labels, dict):
		raise RuntimeError("Labels JSON must contain an object mapping image path -> lanes")

	os.makedirs(output_dir, exist_ok=True)

	total = 0
	saved = 0
	norm_prefix = (folder_prefix or "").strip().strip("/")
	for image_name, raw_lanes in tqdm(cached_labels.items(), desc="Drawing overlays"):
		image_name = (image_name or "").strip()
		if not image_name:
			continue

		if only_image_name and image_name != only_image_name:
			continue

		if norm_prefix:
			norm_image_name = image_name.strip("/")
			if not (
				norm_image_name == norm_prefix
				or norm_image_name.startswith(norm_prefix + "/")
			):
				continue

		total += 1
		image_path = resolve_image_path(image_name, images_root)
		if image_path is None:
			print(f"[WARN] Image not found for cache entry: {image_name}", file=sys.stderr)
			continue

		image = cv2.imread(image_path)
		if image is None:
			print(f"[WARN] Failed to read image: {image_path}", file=sys.stderr)
			continue

		if not isinstance(raw_lanes, list):
			print(f"[WARN] Invalid lane data for: {image_name}", file=sys.stderr)
			continue

		for lane_idx, raw_lane in enumerate(raw_lanes):
			color = palette.get(lane_idx, (255, 255, 255))
			points = parse_cached_lane(raw_lane)
			if len(points) < 2:
				continue

			# Draw only contiguous valid segments (x != -99999)
			segment: List[Tuple[int, int]] = []
			for x, y in points:
				if x == -99999.0:
					if len(segment) >= 2:
						for i in range(len(segment) - 1):
							cv2.line(
								image,
								segment[i],
								segment[i + 1],
								color,
								thickness,
								lineType=cv2.LINE_AA,
							)
					segment = []
					continue
				segment.append((int(round(x)), int(round(y))))

			if len(segment) >= 2:
				for i in range(len(segment) - 1):
					cv2.line(
						image,
						segment[i],
						segment[i + 1],
						color,
						thickness,
						lineType=cv2.LINE_AA,
					)

			if point_radius > 0:
				for x, y in points:
					if x == -99999.0:
						continue
					p = (int(round(x)), int(round(y)))
					cv2.circle(image, p, point_radius, color, -1, lineType=cv2.LINE_AA)

		out_path = os.path.join(output_dir, image_name)
		ensure_parent(out_path)
		ok = cv2.imwrite(out_path, image)
		if not ok:
			print(f"[WARN] Failed to write overlay: {out_path}", file=sys.stderr)
			continue
		saved += 1

	return total, saved


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Draw cached lane labels on corresponding images and save overlays."
	)
	parser.add_argument(
		"--labels-json",
		required=True,
		help="Path to lane cache JSON (e.g. smartrollerz_anno_cache.json).",
	)
	parser.add_argument(
		"--images-root",
		required=True,
		help="Root directory used to resolve image paths from JSON keys.",
	)
	parser.add_argument(
		"--output-dir",
		required=True,
		help="Output directory for overlay images.",
	)
	parser.add_argument(
		"--line-thickness",
		type=int,
		default=2,
		help="Polyline thickness in pixels (default: 2).",
	)
	parser.add_argument(
		"--point-radius",
		type=int,
		default=2,
		help="Point radius in pixels (0 disables points, default: 2).",
	)
	parser.add_argument(
		"--image-name",
		default=None,
		help="Optional single image name/key from JSON to render.",
	)
	parser.add_argument(
		"--folder-prefix",
		default=None,
		help=(
			"Optional folder prefix in JSON keys to render only a subset, "
			"e.g. 'data/rosbags/2023-05-24-17-46-49'."
		),
	)
	return parser


def main() -> None:
	args = build_parser().parse_args()

	total, saved = draw_labels_on_images(
		labels_json=args.labels_json,
		images_root=args.images_root,
		output_dir=args.output_dir,
		thickness=max(1, args.line_thickness),
		point_radius=max(0, args.point_radius),
		only_image_name=args.image_name,
		folder_prefix=args.folder_prefix,
	)

	print(f"Processed entries: {total}")
	print(f"Saved overlays:    {saved}")
	if saved == 0:
		raise RuntimeError("No overlay images were saved")


if __name__ == "__main__":
	main()
