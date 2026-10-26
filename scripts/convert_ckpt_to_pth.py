"""Convert a training checkpoint (.ckpt) into a plain state dict (.pth).

Checkpoints of the training scripts also pickle the training config, whose classes
(`utils.config.Config`) are not available here. They are replaced by a stand-in
class while loading, and only the network weights are saved.

Usage:
    python3 scripts/convert_ckpt_to_pth.py models/model.ckpt models/model.pth
"""

import argparse
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lane_detection_ai.model.checkpoint import extract_state_dict


class _ConfigStandIn(dict):
    """Replaces the pickled config classes of the training code."""


def _register_config_stand_ins() -> None:
    utils_module = types.ModuleType("utils")
    config_module = types.ModuleType("utils.config")
    config_module.Config = _ConfigStandIn
    config_module.ConfigDict = _ConfigStandIn
    utils_module.config = config_module
    sys.modules.setdefault("utils", utils_module)
    sys.modules.setdefault("utils.config", config_module)
    sys.modules.setdefault("config", config_module)


def main() -> None:
    """Convert the checkpoint given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("ckpt", type=Path, help="Training checkpoint (.ckpt)")
    parser.add_argument("pth", type=Path, help="Output file (.pth)")
    args = parser.parse_args()

    _register_config_stand_ins()
    # The checkpoint contains pickled objects: only convert trusted files
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state_dict = extract_state_dict(checkpoint)

    args.pth.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, args.pth)
    print(f"Saved {len(state_dict)} tensors to {args.pth}")


if __name__ == "__main__":
    main()
