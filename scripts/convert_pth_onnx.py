import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from lane_detection_ai.model.utils.common import get_config, get_model


class PreprocessAndModel(nn.Module):
    """
    Wraps the lane model with preprocessing so the exported graph can take uint8 input.

    Input:
      - uint8 tensor in NCHW, range [0..255], RGB
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x_uint8: torch.Tensor):
        x = x_uint8.to(torch.float32) / 255.0
        x = (x - self.mean) / self.std
        pred = self.model(x)
        # Keep a stable output order for downstream compilation/runtime
        return pred["loc_row"], pred["exist_row"], pred["loc_col"], pred["exist_col"]


def _load_checkpoint_model_state(pth_path: str) -> dict:
    ckpt = torch.load(pth_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    # fallback: sometimes the checkpoint is directly a state_dict
    return ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-path", required=True, help="Package base path (same as used by the node)")
    ap.add_argument("--model-config", required=True, help="e.g. config.py")
    ap.add_argument("--pth", required=True, help="Path to .pth checkpoint (absolute or relative to base-path)")
    ap.add_argument("--onnx-out", required=True, help="Output .onnx path")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    base_path = args.base_path
    cfg = get_config(os.path.join(base_path, args.model_config))

    net = get_model(cfg)
    net.eval()

    pth_path = args.pth
    if not os.path.isabs(pth_path):
        pth_path = os.path.join(base_path, pth_path)

    state = _load_checkpoint_model_state(pth_path)

    compatible = {}
    for k, v in state.items():
        # mirrors logic in LaneDetectionAiModel._load_pytorch_model
        if "lane_detection_ai.module." in k:
            compatible[k[7:]] = v
        else:
            compatible[k] = v

    net.load_state_dict(compatible, strict=False)

    wrapped = PreprocessAndModel(net).eval()

    h = int(cfg.train_height)
    w = int(cfg.train_width)

    dummy = torch.zeros((1, 3, h, w), dtype=torch.uint8)

    onnx_out = args.onnx_out
    # Allow passing a directory (common CLI usage)
    if onnx_out.endswith(os.sep) or (os.path.exists(onnx_out) and os.path.isdir(onnx_out)):
        os.makedirs(onnx_out, exist_ok=True)
        onnx_out = os.path.join(onnx_out, "lane_detection_ai.onnx")
    else:
        out_dir = os.path.dirname(onnx_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    torch.onnx.export(
        wrapped,
        dummy,
        onnx_out,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["loc_row", "exist_row", "loc_col", "exist_col"],
        dynamic_axes=None,  # keep fixed shapes for easier Hailo compilation
    )

    print(f"Exported ONNX to: {onnx_out}")
    print(f"Input shape: (1,3,{h},{w}) uint8 RGB")


if __name__ == "__main__":
    main()