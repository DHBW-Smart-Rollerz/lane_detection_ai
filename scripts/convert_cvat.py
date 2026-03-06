import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Tuple

import cv2  # Re-added for imwrite
import numpy as np
from scipy import interpolate
from tqdm import tqdm


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


def load_xml_labels(xml_path: str) -> Dict[str, Dict[str, Any]]:
    """Loads annotations from one XML."""
    labels = {}
    map_names = {
        "left lane": "left_lane",
        "center lane": "center_lane",
        "right lane": "right_lane",
    }
    print(f"Parsing: {xml_path}", end="")
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

                polyline = [
                    tuple(map(float, p.split(",")))
                    for p in pts_str.strip().split(";")
                    if p
                ]
                if polyline:
                    img_data["lanes"].setdefault(internal_name, []).append(polyline)

            if img_data["lanes"]:
                labels[img_path] = img_data
                count += 1

        print(f" -> Found {count} images with lanes.")
        return labels
    except Exception as e:
        print(f"\nError processing {xml_path}: {e}", file=sys.stderr)
        return {}  # Return empty on file error


def generate_outputs(root: str, labels_dict: dict):
    """
    Generates train_gt.txt, cache file, and placeholder segmentation masks.
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

            rel_imgcrop_path = rel_img_path[:-4] + "_crop" + rel_img_path[-4:]
            img = cv2.imread(os.path.join(root, rel_img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # img = img[100 : 1544 - 400, 250 : 2064 - 250]
            img = cv2.resize(img, (1044, 1364), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(root, rel_imgcrop_path), img)

            the_anno_row_anchor = np.array(range(150, 1544 - 500, 1))
            col_for_row_anchor = np.full((len(the_anno_row_anchor)), -99999)
            empty_lane = np.vstack([col_for_row_anchor, the_anno_row_anchor]).T.tolist()

            scale_y = 1  # img_h / (img_h - 600)
            scale_x = 1  # img_w / (img_w - 700)

            lane_data, lane_exists = [], []
            for key in ["left_lane", "center_lane", "right_lane"]:
                if key in img_data["lanes"]:
                    interp = interpolate_lanes(img_data["lanes"][key])
                    filtered_interp = []
                    for interp_poly in interp:
                        filtered = [
                            (int((pt[0] - 250) * scale_x), int((pt[1] - 100) * scale_y))
                            for pt in interp_poly
                            if 250 <= pt[0] < img_w - 250 and 100 <= pt[1] < img_h - 400
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

            # draw lanes on the image and show
            img_debug = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            for lane, color in zip(lane_data, [(0, 255, 0), (0, 0, 255), (255, 0, 0)]):
                for x, y in lane:
                    cv2.circle(img_debug, (int(x), int(y)), 5, color, -1)
            # cv2.imshow("Lanes", img_debug)
            # cv2.waitKey(0)

            cache_dict[rel_imgcrop_path] = lane_data
            train_gt_file.write(
                f"{rel_imgcrop_path} {image_seg_path} {' '.join(map(str, lane_exists))}\n"
            )

    with open(cache_path, "w") as file:
        json.dump(cache_dict, file)

def get_args():
    parser = argparse.ArgumentParser(
        description="Convert XML lane annotations (very short, with seg output)."
    )
    parser.add_argument(
        "--label-files", required=True, nargs="+", help="Input XML annotation file(s)."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Root directory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    combined_labels = {}
    print("\nLoading annotations ...")
    for file_path in args.label_files:
        if not os.path.isfile(file_path):
            print(f"Skip missing file: {file_path}", file=sys.stderr)
            continue

        new_labels = load_xml_labels(xml_path=file_path)
        combined_labels.update(new_labels)

    if combined_labels:
        generate_outputs(args.root, combined_labels)
    else:
        print("\nError: No valid labels loaded.", file=sys.stderr)
        sys.exit(1)
    print("\nProcessing complete.")
