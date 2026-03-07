#!/usr/bin/env python3
"""Resize CVAT XML lane labels from one image size to another.

Example:
	python scripts/resize_labels.py \
		--xml dataset/smartrollerz/labels/2026-01.xml \
		--current-size 800x640 \
		--target-size 2064x1544 \
		--output dataset/smartrollerz/labels/2026-01_resized.xml

Debug overlay example:
	python scripts/resize_labels.py \
		--xml dataset/smartrollerz/labels/2026-01.xml \
		--current-size 800x640 \
		--target-size 2064x1544 \
		--inplace \
		--debug \
		--debug-image dataset/smartrollerz/data/rosbags/rosbag2_2026_01_24-14_00_17/rosbag2_2026_01_24-14_00_17_frame000000.jpg \
		--debug-image-name rosbags/rosbag2_2026_01_24-14_00_17/rosbag2_2026_01_24-14_00_17_frame000000.jpg \
		--debug-output /tmp/lanes_overlay.jpg
"""

from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET
from typing import Iterable, List, Optional, Tuple


def parse_size(size_str: str) -> Tuple[float, float]:
	"""Parse size in the format WIDTHxHEIGHT (e.g., 800x640)."""
	normalized = size_str.strip().lower().replace(" ", "")
	if "x" not in normalized:
		raise ValueError(f"Invalid size '{size_str}'. Use WIDTHxHEIGHT, e.g. 800x640")

	w_str, h_str = normalized.split("x", 1)
	try:
		width = float(w_str)
		height = float(h_str)
	except ValueError as exc:
		raise ValueError(f"Invalid size '{size_str}'. Width/height must be numeric") from exc

	if width <= 0 or height <= 0:
		raise ValueError(f"Invalid size '{size_str}'. Width/height must be > 0")
	return width, height


def parse_points(points_str: str) -> List[Tuple[float, float]]:
	"""Parse CVAT polyline points string: 'x1,y1;x2,y2;...'"""
	points: List[Tuple[float, float]] = []
	for chunk in points_str.split(";"):
		chunk = chunk.strip()
		if not chunk:
			continue
		x_str, y_str = chunk.split(",")
		points.append((float(x_str), float(y_str)))
	return points


def points_to_string(points: Iterable[Tuple[float, float]]) -> str:
	"""Format points back to CVAT format with 2 decimal precision."""
	return ";".join(f"{x:.2f},{y:.2f}" for x, y in points)


def scale_and_clip_points(
	points: Iterable[Tuple[float, float]],
	scale_x: float,
	scale_y: float,
	target_w: float,
	target_h: float,
) -> List[Tuple[float, float]]:
	"""Scale points and clip them to target image bounds."""
	scaled: List[Tuple[float, float]] = []
	for x, y in points:
		nx = x * scale_x
		ny = y * scale_y
		nx = min(max(nx, 0.0), target_w)
		ny = min(max(ny, 0.0), target_h)
		scaled.append((nx, ny))
	return scaled


def resize_labels(
	xml_path: str,
	output_path: str,
	current_size: Tuple[float, float],
	target_size: Tuple[float, float],
) -> ET.ElementTree:
	"""Resize all image dimensions and polyline points in a CVAT XML file."""
	curr_w, curr_h = current_size
	target_w, target_h = target_size

	scale_x = target_w / curr_w
	scale_y = target_h / curr_h

	tree = ET.parse(xml_path)
	root = tree.getroot()

	image_nodes = root.findall("image")
	if not image_nodes:
		raise ValueError("No <image> nodes found in XML")

	for image_node in image_nodes:
		image_node.set("width", str(int(target_w)))
		image_node.set("height", str(int(target_h)))

		for polyline in image_node.findall("polyline"):
			raw_points = polyline.get("points", "")
			if not raw_points.strip():
				continue

			pts = parse_points(raw_points)
			resized_pts = scale_and_clip_points(pts, scale_x, scale_y, target_w, target_h)
			polyline.set("points", points_to_string(resized_pts))

	tree.write(output_path, encoding="utf-8", xml_declaration=True)
	return tree


def draw_debug_overlay(
	tree: ET.ElementTree,
	debug_image_path: str,
	debug_output_path: str,
	debug_image_name: Optional[str] = None,
	show_window: bool = False,
) -> None:
	"""Draw lane polylines on an image for visual verification."""
	try:
		import cv2
	except ImportError as exc:
		raise ImportError("Debug mode requires opencv-python (cv2)") from exc

	image = cv2.imread(debug_image_path)
	if image is None:
		raise FileNotFoundError(f"Could not read debug image: {debug_image_path}")

	root = tree.getroot()
	image_nodes = root.findall("image")
	if not image_nodes:
		raise ValueError("No <image> nodes found in XML")

	target_node = None
	if debug_image_name:
		for node in image_nodes:
			if node.get("name") == debug_image_name:
				target_node = node
				break
		if target_node is None:
			raise ValueError(f"Could not find <image name=\"{debug_image_name}\"> in XML")
	else:
		target_node = image_nodes[0]

	palette = {
		"left lane": (0, 255, 0),
		"center lane": (0, 255, 255),
		"right lane": (0, 128, 255),
	}

	for polyline in target_node.findall("polyline"):
		label = (polyline.get("label") or "").strip().lower()
		color = palette.get(label, (255, 0, 255))

		raw_points = polyline.get("points", "")
		if not raw_points.strip():
			continue

		pts = parse_points(raw_points)
		if len(pts) < 2:
			continue

		int_pts = [(int(round(x)), int(round(y))) for x, y in pts]
		for i in range(len(int_pts) - 1):
			cv2.line(image, int_pts[i], int_pts[i + 1], color, 2, cv2.LINE_AA)
		for p in int_pts:
			cv2.circle(image, p, 2, color, -1)

	out_dir = os.path.dirname(os.path.abspath(debug_output_path))
	if out_dir:
		os.makedirs(out_dir, exist_ok=True)
	ok = cv2.imwrite(debug_output_path, image)
	if not ok:
		raise RuntimeError(f"Failed to write debug image: {debug_output_path}")

	print(f"[DEBUG] Wrote overlay image to: {debug_output_path}")

	if show_window:
		cv2.imshow("resized-lanes-overlay", image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Resize CVAT lane labels from current image size to target image size.")
	parser.add_argument("--xml", required=True, help="Input CVAT XML label file")
	parser.add_argument("--current-size", required=True, help="Current size, e.g. 800x640")
	parser.add_argument("--target-size", required=True, help="Target size, e.g. 2064x1544")

	parser.add_argument(
		"--output",
		default=None,
		help="Output XML path. If omitted, writes '<input>_resized.xml' unless --inplace is set.",
	)
	parser.add_argument(
		"--inplace",
		action="store_true",
		help="Overwrite the input XML file in place.",
	)

	parser.add_argument(
		"--debug",
		action="store_true",
		help="Enable debug visualization by drawing resized lanes on an image.",
	)
	parser.add_argument(
		"--debug-image",
		default=None,
		help="Path to image used for debug overlay.",
	)
	parser.add_argument(
		"--debug-image-name",
		default=None,
		help="Optional XML <image name=...> to pick which lanes to draw (default: first image).",
	)
	parser.add_argument(
		"--debug-output",
		default="debug_overlay.jpg",
		help="Path for saved debug overlay image.",
	)
	parser.add_argument(
		"--show",
		action="store_true",
		help="Show debug overlay in an OpenCV window (requires GUI).",
	)
	return parser


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()

	if not os.path.isfile(args.xml):
		raise FileNotFoundError(f"Input XML not found: {args.xml}")

	current_size = parse_size(args.current_size)
	target_size = parse_size(args.target_size)

	if args.inplace:
		output_path = args.xml
	else:
		if args.output:
			output_path = args.output
		else:
			base, ext = os.path.splitext(args.xml)
			output_path = f"{base}_resized{ext or '.xml'}"

	os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

	tree = resize_labels(
		xml_path=args.xml,
		output_path=output_path,
		current_size=current_size,
		target_size=target_size,
	)

	print(f"Resized labels written to: {output_path}")

	if args.debug:
		if not args.debug_image:
			raise ValueError("--debug requires --debug-image")
		draw_debug_overlay(
			tree=tree,
			debug_image_path=args.debug_image,
			debug_output_path=args.debug_output,
			debug_image_name=args.debug_image_name,
			show_window=args.show,
		)


if __name__ == "__main__":
	main()
