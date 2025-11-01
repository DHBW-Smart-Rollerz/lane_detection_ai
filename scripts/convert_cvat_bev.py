import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from tqdm import tqdm


def calculate_bev_transform(
    checkerboard_path: str, checkerboard_dims: Tuple[int, int], old_lens: bool
) -> Optional[np.ndarray]:
    """
    Calculates the perspective transformation matrix (homography) for BEV.
    Uses checkerboard image dimensions to define the target BEV coordinate space.

    Args:
        checkerboard_path: Path to the checkerboard image.
        checkerboard_dims: Tuple of (inner_corners_width, inner_corners_height).
        dst_margin_px: Margin for the destination rectangle in the reference BEV image.

    Returns:
        The 3x3 perspective transformation matrix M, or None if calibration fails.
    """
    print(f"\nCalculating BEV transform using: {checkerboard_path}")
    print(f"Checkerboard dimensions (inner corners): {checkerboard_dims}")

    img = cv2.imread(checkerboard_path)
    if img is None:
        print(f"Error: Cannot load checkerboard image at {checkerboard_path}")
        return None

    ref_bev_h, ref_bev_w = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(img, checkerboard_dims, None)
    if not ret:
        print(
            f"Error: Checkerboard corners not found in {checkerboard_path}. "
            f"Ensure the dimensions {checkerboard_dims} are correct and the "
            f"checkerboard is fully visible.",
        )
        return None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)

    cols, rows = checkerboard_dims
    src = np.array(
        [
            corners[0][0],
            corners[cols - 1][0],
            corners[cols * rows - 1][0],
            corners[cols * (rows - 1)][0],
        ]
    )

    target_size = (ref_bev_w, ref_bev_h)

    # Calculate the width and height of the source rectangle
    width_src = np.linalg.norm(src[0] - src[1])
    height_src = np.linalg.norm(src[0] - src[3])

    # Calculate aspect ratio
    aspect_ratio = target_size[0] / target_size[1]

    # Calculate destination rectangle dimensions maintaining aspect ratio
    if width_src / height_src > aspect_ratio:
        # Width is the limiting dimension
        width_dst = width_src
        height_dst = width_dst / aspect_ratio
    else:
        # Height is the limiting dimension
        height_dst = height_src
        width_dst = height_dst * aspect_ratio

    # Calculate the center of the src_points
    center_src = np.mean(src, axis=0)

    # Define destination points centered around the center of src_points
    dst = np.float32(
        [
            [center_src[0] - width_dst * 0.4, center_src[1] - height_dst * 0.4],
            [center_src[0] + width_dst * 0.4, center_src[1] - height_dst * 0.4],
            [center_src[0] + width_dst * 0.4, center_src[1] + height_dst * 0.4],
            [center_src[0] - width_dst * 0.4, center_src[1] + height_dst * 0.4],
        ]
    )

    if old_lens:
        # Adjust the destination points for the old lens calibration
        dst[:, 1] += 1 * height_dst
    else:
        # Adjust the destination points for the new lens calibration
        dst[:, 1] += 1 * height_dst

    M = cv2.getPerspectiveTransform(src, dst)
    return M


def linear_interpolation(
    x: np.ndarray, y: np.ndarray, num_interp: int
) -> List[Tuple[int, int]]:
    """Linear interpolation helper."""
    if len(x) < 2:
        return []
    d = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    cd = np.zeros(len(x))
    cd[1:] = np.cumsum(d)
    u = cd / cd[-1] if cd[-1] > 0 else np.linspace(0, 1, len(x))
    interp_x = interpolate.interp1d(
        u, x, kind="linear", bounds_error=False, fill_value=(x[0], x[-1])
    )
    interp_y = interpolate.interp1d(
        u, y, kind="linear", bounds_error=False, fill_value=(y[0], y[-1])
    )
    u_new = np.linspace(0, 1, num_interp)
    return list(zip(map(int, interp_x(u_new)), map(int, interp_y(u_new))))


def interpolate_lanes(
    lane_polylines: List[List[Tuple[float, float]]],
) -> List[List[Tuple[int, int]]]:
    """Interpolates polylines in a lane definition."""
    interpolations = []
    for polyline in lane_polylines:
        seen = set()
        cleaned = [
            c for c in polyline if tuple(c) not in seen and not seen.add(tuple(c))
        ]
        if len(cleaned) < 2:
            continue
        x, y = map(np.array, zip(*cleaned))
        # Inlined compute_polyline_length:
        length = sum(
            np.sqrt((x[i + 1] - x[i]) ** 2 + (y[i + 1] - y[i]) ** 2)
            for i in range(len(cleaned) - 1)
        )
        num_interp = (
            max(2, int(length)) if length > 0 else (2 if len(cleaned) >= 2 else 0)
        )
        if num_interp > 0:
            # print(f"Interpolating {len(cleaned)} points to {num_interp} points.")
            interp_pts = linear_interpolation(x, y, num_interp)
            if interp_pts:
                interpolations.append(interp_pts)
    return interpolations


def get_sections(
    interp_lanes: List[List[Tuple[int, int]]], y_anchors: np.ndarray
) -> List[List[float]]:
    """Calculates x-coords for y-anchors."""
    cols = np.full(len(y_anchors), -99999.0)
    if interp_lanes:
        pts = np.vstack(interp_lanes)
        for i, y_anc in enumerate(y_anchors):
            matches = pts[pts[:, 1] == y_anc]
            if len(matches) > 0:
                cols[i] = np.mean(matches[:, 0])
    return np.vstack([cols, y_anchors]).T.tolist()


def load_xml_labels(xml_path: str, M: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """
    Loads annotations from one XML and transforms points to BEV using M.
    Stores original width/height.
    """
    labels = {}
    map_names = {
        "left lane": "left_lane",
        "center lane": "center_lane",
        "right lane": "right_lane",
    }
    print(f"Parsing and transforming: {xml_path}", end="")
    try:
        root = ET.parse(xml_path).getroot()
        count = 0
        for img_elem in root.findall(".//image"):
            img_path = img_elem.get("name")
            if not img_path:
                continue

            # Ensure width/height are stored, default to 0 if missing but skip adding if essential dims missing
            w = int(img_elem.get("width", 0))
            h = int(img_elem.get("height", 0))
            if not w or not h:
                print(
                    f"\nWarning: Skipping image {img_path} due to missing width/height attributes.",
                    file=sys.stderr,
                )
                continue
            img_data = {"lanes": {}, "width": w, "height": h}

            for poly_elem in img_elem.findall(".//polyline"):
                label, pts_str = poly_elem.get("label"), poly_elem.get("points")
                internal_name = map_names.get(label)
                if not internal_name or not pts_str:
                    continue

                original_polyline = [
                    tuple(map(float, p.split(",")))
                    for p in pts_str.strip().split(";")
                    if p
                ]
                if original_polyline:
                    interpolated_polyline = interpolate_lanes([original_polyline])
                    pts_to_transform = np.array(interpolated_polyline, dtype=np.float32)
                    transf_pts = cv2.perspectiveTransform(pts_to_transform, M)
                    transformed_polyline = [tuple(pt) for pt in transf_pts[0]]
                    filtered_polyline = [
                        pt
                        for pt in transformed_polyline
                        if 0 <= pt[0] < w and 0 <= pt[1] < h
                    ]
                    img_data["lanes"].setdefault(internal_name, []).append(
                        filtered_polyline
                    )

            # Only add if lanes were found (optional: add even if no lanes?)
            if img_data["lanes"]:
                labels[img_path] = img_data
                count += 1

        print(f" -> Found and transformed {count} images with lanes.")
        return labels
    except Exception as e:
        print(f"\nError processing {xml_path}: {e}", file=sys.stderr)
        return {}


def generate_outputs(root: str, labels_dict: dict, M, old_lens: bool, debug: bool):
    """
    Generates train_gt.txt, cache file, and placeholder segmentation masks.
    Uses ORIGINAL image dimensions for masks and anchors.
    """
    try:
        cache_path = os.path.join(root, "labels", "smartrollerz_anno_cache.json")
        with open(cache_path, "r") as file:
            cache_dict = json.load(file)
    except Exception:
        cache_dict = {}

    with open(os.path.join(root, "labels", "train_gt.txt"), "a") as train_gt_file:
        for rel_img_path, img_data in tqdm(labels_dict.items()):
            rel_img_path = os.path.join("data", rel_img_path)

            img_h = img_data.get("height")
            img_w = img_data.get("width")

            image_seg_path = rel_img_path[:-4] + "_seg" + rel_img_path[-4:]
            image_seg = np.zeros((1544, 2064), dtype=np.uint8)
            image_seg = image_seg[100 : 1544 - 100, 100 : 2064 - 100]
            cv2.imwrite(os.path.join(root, image_seg_path), image_seg)

            rel_imgbev_path = rel_img_path[:-4] + "_bev" + rel_img_path[-4:]
            img = cv2.imread(os.path.join(root, rel_img_path))
            img = cv2.warpPerspective(img, M, (img_w, img_h), flags=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if old_lens:
                img = img[200 : 1544 - 400, 350 : 2064 - 350]
            else:
                img = cv2.resize(img, (1364, 944), interpolation=cv2.INTER_LINEAR)

            cv2.imwrite(os.path.join(root, rel_imgbev_path), img)

            the_anno_row_anchor = np.array(range(0, 1544 - 0, 1))
            col_for_row_anchor = np.full((len(the_anno_row_anchor)), -99999)
            empty_lane = np.vstack([col_for_row_anchor, the_anno_row_anchor]).T.tolist()

            lane_data, lane_exists = [], []
            for key in ["left_lane", "center_lane", "right_lane"]:
                if key in img_data["lanes"]:
                    interp = interpolate_lanes(img_data["lanes"][key])
                    filtered_interp = []
                    for interp_poly in interp:
                        if old_lens:
                            filtered = [
                                (
                                    int((pt[0] - 350)),
                                    int((pt[1] - 200)),
                                )
                                for pt in interp_poly
                                if 350 <= pt[0] < img_w - 350
                                and 200 <= pt[1] < img_h - 400
                            ]
                        else:
                            scale_y = (img_h - 600) / img_h
                            scale_x = (img_w - 700) / img_w
                            filtered = [
                                (int((pt[0]) * scale_x), int((pt[1]) * scale_y))
                                for pt in interp_poly
                                if 0 <= pt[0] < img_w and 0 <= pt[1] < img_h
                            ]

                        if len(filtered) > 1:
                            filtered_interp.append(filtered)

                    if len(filtered_interp) > 0:
                        lane_data.append(
                            get_sections(filtered_interp, the_anno_row_anchor)
                        )
                        lane_exists.append(1)
                    else:
                        lane_data.append(empty_lane)
                        lane_exists.append(0)
                else:
                    lane_data.append(empty_lane)
                    lane_exists.append(0)

            if debug:
                # draw lanes on the image and show
                img_debug = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                for lane, color in zip(
                    lane_data, [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
                ):
                    for x, y in lane:
                        cv2.circle(img_debug, (int(x), int(y)), 5, color, -1)
                cv2.imshow("Lanes", img_debug)
                cv2.waitKey(0)

            if not debug:
                cache_dict[rel_imgbev_path] = lane_data
                train_gt_file.write(
                    f"{rel_imgbev_path} {image_seg_path} {' '.join(map(str, lane_exists))}\n"
                )

    if not debug:
        with open(cache_path, "w") as file:
            json.dump(cache_dict, file)


def get_args():
    parser = argparse.ArgumentParser(
        description="Convert XML lane annotations to BEV format (using original image size)."
    )
    parser.add_argument(
        "--label-files", required=True, nargs="+", help="Input XML annotation file(s)."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Root directory.",
    )
    parser.add_argument(
        "--checkerboard-image",
        required=True,
        help="Path to the checkerboard image for BEV calibration.",
    )
    parser.add_argument(
        "--old-lens",
        action="store_true",
        help="Use new lens calibration.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for additional output.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    checkerboard_dims = 8, 6
    M = calculate_bev_transform(
        args.checkerboard_image, checkerboard_dims, args.old_lens
    )
    if M is None:
        print("Error: BEV calibration failed. Exiting.", file=sys.stderr)
        sys.exit(1)

    combined_labels = {}
    print("\nLoading annotations and transforming to BEV...")
    for file_path in args.label_files:
        if not os.path.isfile(file_path):
            print(f"Skip missing file: {file_path}", file=sys.stderr)
            continue

        new_labels = load_xml_labels(file_path, M)
        combined_labels.update(new_labels)

    if combined_labels:
        generate_outputs(args.root, combined_labels, M, args.old_lens, args.debug)
    else:
        print("\nError: No valid labels loaded.", file=sys.stderr)
        sys.exit(1)
    print("\nProcessing complete.")
