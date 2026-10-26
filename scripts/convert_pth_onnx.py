"""Export a trained model (.pth) to ONNX with the normalization in the graph.

The exported network takes a uint8 RGB image [1, 3, H, W] (0..255) as input, so
the Hailo compiler can be calibrated with raw images and the node doesn't have to
normalize on the host. The outputs are loc_row, exist_row, loc_col and exist_col.

Usage:
    python3 scripts/convert_pth_onnx.py --base-path . \
        --model-config config/model_config.py --pth models/model.pth \
        --onnx-out build/lane_detection_ai.onnx
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lane_detection_ai.model.checkpoint import load_checkpoint
from lane_detection_ai.model.utils.common import (
    IMAGE_MEAN,
    IMAGE_STD,
    get_config,
    get_model,
)


class PreprocessAndModel(torch.nn.Module):
    """Wraps the model with the normalization of uint8 RGB input."""

    def __init__(self, model: torch.nn.Module) -> None:
        """Initialize the wrapper.

        Args:
            model: The lane detection network.
        """
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.from_numpy(IMAGE_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.from_numpy(IMAGE_STD).view(1, 3, 1, 1))

    def forward(
        self, x_uint8: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize the image and run the network (outputs in a fixed order)."""
        x = (x_uint8.to(torch.float32) / 255.0 - self.mean) / self.std
        pred = self.model(x)
        return pred["loc_row"], pred["exist_row"], pred["loc_col"], pred["exist_col"]


def _resolve(base_path: Path, path: Path) -> Path:
    return path if path.is_absolute() else base_path / path


def main() -> None:
    """Export the model given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--base-path", type=Path, required=True, help="Package path (as the node)"
    )
    parser.add_argument(
        "--model-config", type=Path, required=True, help="Model config path"
    )
    parser.add_argument(
        "--pth", type=Path, required=True, help="Checkpoint, relative to base path"
    )
    parser.add_argument(
        "--onnx-out",
        type=Path,
        required=True,
        help="Output .onnx file or directory, relative to base path",
    )
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    cfg = get_config(str(_resolve(args.base_path, args.model_config)))
    net = get_model(cfg)
    load_checkpoint(net, str(_resolve(args.base_path, args.pth)))
    wrapped = PreprocessAndModel(net.eval()).eval()

    onnx_out = _resolve(args.base_path, args.onnx_out)
    if onnx_out.is_dir() or str(args.onnx_out).endswith("/"):
        onnx_out /= "lane_detection_ai.onnx"
    onnx_out.parent.mkdir(parents=True, exist_ok=True)

    height, width = int(cfg.train_height), int(cfg.train_width)
    torch.onnx.export(
        wrapped,
        torch.zeros((1, 3, height, width), dtype=torch.uint8),
        str(onnx_out),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["loc_row", "exist_row", "loc_col", "exist_col"],
        dynamic_axes=None,  # fixed shapes for the Hailo compiler
        dynamo=False,
    )

    print(f"Exported ONNX to: {onnx_out}")
    print(f"Input shape: (1, 3, {height}, {width}) uint8 RGB")


if __name__ == "__main__":
    main()
