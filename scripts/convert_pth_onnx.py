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
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    # fallback: sometimes the checkpoint is directly a state_dict
    return ckpt


def _adapt_state_dict_for_model(
    state_dict: dict, target_state_dict: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], tuple[int, int, int]]:
    """Best-effort key adaptation for checkpoints trained with wrappers/prefixes.

    Returns:
        (best_candidate_state, (matched, missing, unexpected))
    """
    raw_items = {k: v for k, v in state_dict.items() if torch.is_tensor(v)}
    target_keys = set(target_state_dict.keys())

    prefixes = [
        "module.",
        "model.",
        "net.",
        "network.",
        "lane_detection_ai.module.",
        "lane_detection_ai.",
    ]

    def _strip_once(key: str, pfx: str) -> str:
        return key[len(pfx):] if key.startswith(pfx) else key

    def _strip_iterative(key: str) -> str:
        out = key
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if out.startswith(p):
                    out = out[len(p) :]
                    changed = True
        return out

    candidates: list[dict[str, torch.Tensor]] = [dict(raw_items)]

    for p in prefixes:
        remap = {_strip_once(k, p): v for k, v in raw_items.items()}
        candidates.append(remap)

    candidates.append({_strip_iterative(k): v for k, v in raw_items.items()})

    def _score(cand: dict[str, torch.Tensor]) -> tuple[int, int, int]:
        cand_keys = set(cand.keys())
        matched = len(cand_keys & target_keys)
        missing = len(target_keys - cand_keys)
        unexpected = len(cand_keys - target_keys)
        return matched, missing, unexpected

    best = None
    best_score = (-1, 10**9, 10**9)
    for cand in candidates:
        s = _score(cand)
        if s[0] > best_score[0] or (s[0] == best_score[0] and (s[1] + s[2]) < (best_score[1] + best_score[2])):
            best = cand
            best_score = s

    assert best is not None
    return best, best_score


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

    if not isinstance(state, dict):
        raise RuntimeError(
            f"Unsupported checkpoint structure in {pth_path}: expected dict-like state_dict, got {type(state).__name__}"
        )

    best_state, score = _adapt_state_dict_for_model(state, net.state_dict())
    missing, unexpected = net.load_state_dict(best_state, strict=False)

    total_target = max(1, len(net.state_dict()))
    coverage = 1.0 - (len(missing) / total_target)
    if coverage < 0.90:
        raise RuntimeError(
            "Checkpoint/model mismatch while preparing ONNX export.\n"
            f"file={pth_path}\n"
            f"matched={score[0]}, missing={len(missing)}, unexpected={len(unexpected)}, coverage={coverage:.3f}\n"
            f"sample_missing={list(missing)[:12]}\n"
            f"sample_unexpected={list(unexpected)[:12]}\n"
            "Likely causes: wrong backbone/head config, wrong checkpoint file, or incompatible training code."
        )

    wrapped = PreprocessAndModel(net).eval()

    h = int(cfg.train_height)
    w = int(cfg.train_width)

    dummy = torch.zeros((1, 3, h, w), dtype=torch.uint8)

    onnx_out = args.onnx_out
    # Match --pth behavior: resolve relative paths against --base-path.
    if not os.path.isabs(onnx_out):
        onnx_out = os.path.join(base_path, onnx_out)

    # Allow passing a directory (common CLI usage)
    if onnx_out.endswith(os.sep) or (os.path.exists(onnx_out) and os.path.isdir(onnx_out)):
        os.makedirs(onnx_out, exist_ok=True)
        onnx_out = os.path.join(onnx_out, "lane_detection_ai.onnx")
    else:
        out_dir = os.path.dirname(onnx_out)
        if out_dir:
            if os.path.exists(out_dir) and not os.path.isdir(out_dir):
                raise RuntimeError(
                    "Cannot create output directory because a file already exists at that path:\n"
                    f"  {out_dir}\n\n"
                    "Choose a different --onnx-out path or rename/delete the conflicting file."
                )
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