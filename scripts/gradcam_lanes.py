#!/usr/bin/env python3
"""
Grad-CAM for the UFLDv2 lane detection model (offline, no ROS required).

Shows which image regions the network uses to place a lane. For every selected
image an overlay (heatmap + predicted lane points) is written to the output folder.

Target score per lane:
  loc   -- sum of the winning cell logits of all existing row/col anchors of the lane
           ("why is the lane placed HERE?")
  exist -- sum of the "lane exists" logits of all row/col anchors of the lane
           ("why does the network think the lane EXISTS?")
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch

from lane_detection_ai.model.checkpoint import load_checkpoint
from lane_detection_ai.model.decode import decode_lanes, merge_lanes
from lane_detection_ai.model.utils.common import (
    IMAGE_MEAN,
    IMAGE_STD,
    get_config,
    get_model,
)

PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LANE_NAMES = ["left", "center", "right"]
LANE_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR: blue, green, red


class GradCam:
    """Minimal Grad-CAM implementation based on forward/backward hooks."""

    def __init__(self, net: torch.nn.Module, layer: torch.nn.Module):
        """
        Initialize the hooks.

        Arguments:
            net -- The model.
            layer -- The layer whose activations are explained.
        """
        self.net = net
        self.activations = None
        self.gradients = None
        layer.register_forward_hook(self._save_activation)
        layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, score: torch.Tensor) -> np.ndarray:
        """
        Compute the Grad-CAM heatmap for a scalar score.

        Arguments:
            score -- Scalar computed from the network output of the last forward pass.

        Returns:
            np.ndarray -- Heatmap in [0, 1] with the spatial size of the layer.
        """
        if not score.requires_grad:
            # Lane not detected on any anchor -> nothing to explain
            return np.zeros(self.activations.shape[2:], dtype=np.float32)
        self.net.zero_grad()
        score.backward(retain_graph=True)
        # Channel weights = global average of the gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1))[0]
        cam = cam.detach().numpy()
        return cam / cam.max() if cam.max() > 0 else cam


def lane_score(pred: dict, lane: int, target: str) -> torch.Tensor:
    """
    Build the scalar score that should be explained for one lane.

    Arguments:
        pred -- Network output dictionary.
        lane -- Lane index (0 left, 1 center, 2 right).
        target -- "loc" or "exist".

    Returns:
        torch.Tensor -- Scalar score.
    """
    score = torch.zeros(())
    for kind in ("row", "col"):
        loc = pred[f"loc_{kind}"][0, :, :, lane]  # [num_cells, num_anchors]
        exist = pred[f"exist_{kind}"][0, :, :, lane]  # [2, num_anchors]
        valid = exist.argmax(0) == 1  # [num_anchors]
        if target == "exist":
            score = score + exist[1].sum()
        elif valid.any():
            score = score + loc.max(0).values[valid].sum()
    return score


def preprocess(image: np.ndarray, cfg) -> torch.Tensor:
    """Apply the same preprocessing as LaneDetectionAiModel._preprocess()."""
    image = cv2.resize(
        image, (cfg.train_width, int(cfg.train_height / cfg.crop_ratio))
    )
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = (image.astype(np.float32) / 255.0 - IMAGE_MEAN) / IMAGE_STD
    tensor = torch.from_numpy(image.transpose(2, 0, 1).copy())
    return tensor[None, :, -cfg.train_height :, :]


def overlay(image: np.ndarray, cam: np.ndarray, points, lane: int) -> np.ndarray:
    """Draw the heatmap and the predicted points of a lane onto the image."""
    h, w = image.shape[:2]
    cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    out = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), 0.55, heat, 0.45, 0)
    for x, y in points:
        cv2.circle(out, (int(x), int(y)), 4, LANE_COLORS[lane], -1)
        cv2.circle(out, (int(x), int(y)), 4, (255, 255, 255), 1)
    cv2.putText(
        out, LANE_NAMES[lane], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
    )
    return out


def main():
    """Run Grad-CAM on a set of images and save the overlays."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/model_bev_config.py")
    parser.add_argument("--images", default="models/eval_2025-03-06")
    parser.add_argument("--frames", default="0,100,200,400,600,800,1000,1100",
                        help="Comma separated indices into the sorted image list, or 'all'")
    parser.add_argument("--layer", default="layer3", choices=["layer2", "layer3", "layer4"])
    parser.add_argument("--target", default="loc", choices=["loc", "exist"])
    parser.add_argument("--output", default="gradcam_output")
    args = parser.parse_args()

    cfg = get_config(os.path.join(PACKAGE_DIR, args.config))
    if not cfg.test_model.endswith(".pth"):
        raise ValueError("Grad-CAM needs gradients -> use a .pth model, not ONNX/HEF")
    net = get_model(cfg)
    load_checkpoint(net, os.path.join(PACKAGE_DIR, cfg.test_model))
    net.eval()

    grad_cam = GradCam(net, getattr(net.model, args.layer))

    files = sorted(glob.glob(os.path.join(PACKAGE_DIR, args.images, "*.jpg")))
    indices = range(len(files)) if args.frames == "all" else [int(i) for i in args.frames.split(",")]
    os.makedirs(args.output, exist_ok=True)

    for idx in indices:
        image = cv2.imread(files[idx], cv2.IMREAD_GRAYSCALE)
        pred = net(preprocess(image, cfg).requires_grad_(True))

        lanes = merge_lanes(
            *decode_lanes(
                {k: v.detach().numpy() for k, v in pred.items()},
                cfg.row_anchor,
                cfg.col_anchor,
                image.shape[1],
                image.shape[0],
            )
        )

        tiles = []
        for lane in range(cfg.num_lanes):
            cam = grad_cam(lane_score(pred, lane, args.target))
            points = lanes[lane] if lanes[lane] is not None else []
            tiles.append(overlay(image, cam, points, lane))

        name = f"{idx:05d}_{args.layer}_{args.target}.jpg"
        cv2.imwrite(os.path.join(args.output, name), np.hstack(tiles))
        print(f"[{idx}] {os.path.basename(files[idx])} -> {name}")

    print(f"Done. Overlays in {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
