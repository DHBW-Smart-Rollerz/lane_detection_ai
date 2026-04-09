import argparse
import os
import sys
from typing import Any, Dict

import torch


def _extract_state_dict(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
	if not isinstance(ckpt_obj, dict):
		raise ValueError("Unsupported checkpoint format: expected a dict-like checkpoint.")

	if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
		return ckpt_obj["state_dict"]
	if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
		return ckpt_obj["model"]
	if all(torch.is_tensor(v) for v in ckpt_obj.values()):
		return ckpt_obj

	raise ValueError(
		"Could not find model weights in checkpoint. Expected one of: "
		"{'state_dict': ...}, {'model': ...}, or a raw state_dict."
	)


def _load_checkpoint_safely(ckpt_path: str) -> Any:
	# Ensure project root is on sys.path so pickled config objects like
	# utils.config.Config can be imported during torch.load(..., weights_only=False).
	script_dir = os.path.dirname(os.path.abspath(__file__))
	project_root = os.path.dirname(script_dir)
	if project_root not in sys.path:
		sys.path.insert(0, project_root)

	# Prefer safer loading when available; if unsupported or insufficient,
	# fall back to full checkpoint loading.
	try:
		return torch.load(ckpt_path, map_location="cpu", weights_only=True)
	except TypeError:
		# Older torch without weights_only argument
		pass
	except Exception:
		# Some Lightning checkpoints still require full load for state extraction
		pass

	return torch.load(ckpt_path, map_location="cpu")


def main() -> None:
	parser = argparse.ArgumentParser(description="Convert Lightning .ckpt to .pth state_dict file")
	parser.add_argument("--ckpt", required=True, type=str, help="Path to source .ckpt")
	parser.add_argument(
		"--out",
		default=None,
		type=str,
		help="Path to output .pth (default: same folder/name as ckpt)",
	)
	parser.add_argument(
		"--save_key",
		default="model",
		choices=["model", "state_dict", "raw"],
		help="How to store weights in output file",
	)
	args = parser.parse_args()

	ckpt_path = os.path.abspath(args.ckpt)
	if not os.path.isfile(ckpt_path):
		raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

	if args.out is None:
		base, _ = os.path.splitext(ckpt_path)
		out_path = base + ".pth"
	else:
		out_path = os.path.abspath(args.out)

	os.makedirs(os.path.dirname(out_path), exist_ok=True)

	ckpt_obj = _load_checkpoint_safely(ckpt_path)
	state_dict = _extract_state_dict(ckpt_obj)

	if args.save_key == "raw":
		to_save: Any = state_dict
	elif args.save_key == "state_dict":
		to_save = {"state_dict": state_dict}
	else:
		to_save = {"model": state_dict}

	torch.save(to_save, out_path)
	print(f"[convert_ckpt_to_pth] source: {ckpt_path}")
	print(f"[convert_ckpt_to_pth] output: {out_path}")
	print(f"[convert_ckpt_to_pth] params: {len(state_dict)} tensors")


if __name__ == "__main__":
	main()
