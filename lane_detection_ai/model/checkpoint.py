"""Loading of trained PyTorch checkpoints into the lane detection network."""

from __future__ import annotations

import pickle
from collections.abc import Callable, Mapping

import torch

# Prefixes that training wrappers (DataParallel, Lightning, ...) add to the keys.
_KEY_PREFIXES = (
    "module.",
    "model.",
    "net.",
    "network.",
    "lane_detection_ai.module.",
    "lane_detection_ai.",
)

# Minimal share of the network weights that must be found in the checkpoint.
MIN_COVERAGE = 0.9


def extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    """Return the tensors of a checkpoint as state dict.

    Supports plain state dicts and checkpoints that store the state dict under
    "model" (UFLD training) or "state_dict" (PyTorch Lightning).

    Args:
        checkpoint: The object loaded with `torch.load`.

    Returns:
        The tensor entries of the state dict.
    """
    state_dict = checkpoint
    if isinstance(checkpoint, Mapping):
        for key in ("model", "state_dict"):
            if isinstance(checkpoint.get(key), Mapping):
                state_dict = checkpoint[key]
                break

    if not isinstance(state_dict, Mapping):
        msg = (
            "Unsupported checkpoint structure: expected a state dict, got "
            f"{type(state_dict).__name__}"
        )
        raise TypeError(msg)

    return {k: v for k, v in state_dict.items() if torch.is_tensor(v)}


def _strip_prefixes(key: str) -> str:
    stripped = True
    while stripped:
        stripped = False
        for prefix in _KEY_PREFIXES:
            if key.startswith(prefix):
                key = key.removeprefix(prefix)
                stripped = True
    return key


def adapt_state_dict(
    state_dict: Mapping[str, torch.Tensor], target_keys: set[str]
) -> dict[str, torch.Tensor]:
    """Rename the keys of a state dict to match the keys of the network.

    Tries the original keys, the keys with each known prefix removed once and the
    keys with all known prefixes removed and returns the candidate with the most
    matching keys (ties are broken by the fewest missing and unexpected keys).

    Args:
        state_dict: The state dict of the checkpoint.
        target_keys: The state dict keys of the network.

    Returns:
        The state dict with the best matching keys.
    """
    candidates = [dict(state_dict)]
    candidates.extend(
        {k.removeprefix(prefix): v for k, v in state_dict.items()}
        for prefix in _KEY_PREFIXES
    )
    candidates.append({_strip_prefixes(k): v for k, v in state_dict.items()})

    def score(candidate: dict[str, torch.Tensor]) -> tuple[int, int]:
        keys = set(candidate)
        matched = len(keys & target_keys)
        mismatched = len(target_keys - keys) + len(keys - target_keys)
        return matched, -mismatched

    return max(candidates, key=score)


def _torch_load(path: str, log: Callable[[str], None]) -> object:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        # Checkpoints of the training scripts also pickle their config object.
        # Prefer converting them with scripts/convert_ckpt_to_pth.py.
        log(
            f"[LaneDetectionAiModel] {path} contains non-tensor objects, loading it "
            "with weights_only=False. Only load checkpoints from trusted sources."
        )
        return torch.load(path, map_location="cpu", weights_only=False)


def load_checkpoint(
    net: torch.nn.Module, path: str, log: Callable[[str], None] = print
) -> None:
    """Load the weights of a checkpoint file into the network.

    Args:
        net: The network to load the weights into.
        path: Path of the checkpoint (.pth/.ckpt).
        log: Function to log messages with.

    Raises:
        RuntimeError: If the checkpoint does not fit the network.
    """
    state_dict = extract_state_dict(_torch_load(path, log))
    target_state = net.state_dict()
    adapted = adapt_state_dict(state_dict, set(target_state))

    shape_mismatches = [
        f"{k}: checkpoint {tuple(v.shape)} != model {tuple(target_state[k].shape)}"
        for k, v in adapted.items()
        if k in target_state and v.shape != target_state[k].shape
    ]
    if shape_mismatches:
        msg = (
            f"Checkpoint {path} does not fit the model config (different layer "
            f"sizes): {shape_mismatches[:5]}. Check the backbone, the anchor/grid "
            "sizes and mlp_mid_dim in the model config."
        )
        raise RuntimeError(msg)

    missing, unexpected = net.load_state_dict(adapted, strict=False)
    coverage = 1.0 - len(missing) / max(1, len(target_state))
    if coverage < MIN_COVERAGE:
        msg = (
            f"Checkpoint {path} does not fit the model: {len(missing)} missing and "
            f"{len(unexpected)} unexpected keys (coverage {coverage:.3f}).\n"
            f"Missing (sample): {list(missing)[:12]}\n"
            f"Unexpected (sample): {list(unexpected)[:12]}\n"
            f"Checkpoint keys (sample): {list(state_dict)[:12]}\n"
            "Likely causes: wrong backbone/head config, wrong checkpoint file or "
            "incompatible training code."
        )
        raise RuntimeError(msg)
