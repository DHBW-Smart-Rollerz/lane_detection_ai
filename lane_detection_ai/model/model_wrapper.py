#!/usr/bin/env python3
import os
import time
from typing import List

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torchvision.transforms as transforms
from camera_preprocessing.transformation.calibration import Calibration
from timing.timer import Timer

from lane_detection_ai.model.utils.common import get_config, get_model

# --- NEW: optional HailoRT import ---
try:
    from hailo_platform import (  # type: ignore
        HEF,
        ConfigureParams,
        FormatType,
        HailoStreamInterface,
        InferVStreams,
        InputVStreamParams,
        OutputVStreamParams,
        VDevice,
    )

    _HAILO_AVAILABLE = True
except Exception as e:
    _HAILO_AVAILABLE = False
    _HAILO_IMPORT_ERROR = e


class _HailoSession:
    """
    Minimal HailoRT session for single-network inference.

    Expects the compiled HEF to expose outputs compatible with:
      loc_row, exist_row, loc_col, exist_col
    """

    def __init__(self, hef_path: str):
        if not _HAILO_AVAILABLE:
            import sys

            details = ""
            try:
                details = f" (import error: {_HAILO_IMPORT_ERROR!r})"
            except Exception:
                details = ""

            raise RuntimeError(
                "HailoRT python bindings are not available in the current Python interpreter."
                f"\n- Python: {sys.executable}"
                f"\n- Missing module: hailo_platform{details}"
                "\n\nFix: install HailoRT Python bindings for THIS interpreter, or run the node inside the environment where they are installed."
                "\nFor many Hailo installs this is not on PyPI; you typically need to install the HailoRT .whl provided by Hailo, or use the vendor install scripts."
            )

        self._hef = HEF(hef_path)

        # VDevice allocation can fail if another process holds the device.
        # Retry briefly to handle the common case of a previous node still shutting down.
        retries = 30
        delay_s = 0.1
        last_exc: Exception | None = None
        for _ in range(retries):
            try:
                self._vdevice = VDevice()
                last_exc = None
                break
            except Exception as e:
                last_exc = e
                time.sleep(delay_s)

        if last_exc is not None:
            raise RuntimeError(
                (
                    "Failed to create Hailo VDevice (device busy/unavailable).\n"
                    "This usually means another process is using the Hailo device.\n\n"
                    "Try: `sudo lsof /dev/hailo*` or `sudo fuser -v /dev/hailo*`\n"
                    "Then stop that process (or unplug/replug / restart service)."
                )
            ) from last_exc

        configure_params = ConfigureParams.create_from_hef(
            self._hef, interface=HailoStreamInterface.PCIe
        )
        self._network_groups = self._vdevice.configure(self._hef, configure_params)
        self._network_group = self._network_groups[0]

        self._input_infos = self._hef.get_input_vstream_infos()
        self._output_infos = self._hef.get_output_vstream_infos()

        # Prefer explicit UINT8 to avoid runtime AUTO conversion overhead.
        # Fall back to AUTO if UINT8 is unsupported for this HEF/runtime.
        preferred_format = getattr(FormatType, "UINT8", FormatType.AUTO)
        self._vstream_format = preferred_format
        try:
            self._in_params = InputVStreamParams.make_from_network_group(
                self._network_group, format_type=preferred_format
            )
            self._out_params = OutputVStreamParams.make_from_network_group(
                self._network_group, format_type=preferred_format
            )
        except Exception:
            self._vstream_format = FormatType.AUTO
            self._in_params = InputVStreamParams.make_from_network_group(
                self._network_group, format_type=FormatType.AUTO
            )
            self._out_params = OutputVStreamParams.make_from_network_group(
                self._network_group, format_type=FormatType.AUTO
            )

        self._in_name = self._input_infos[0].name
        self._out_names = [o.name for o in self._output_infos]

        # Keep a persistent activation/infer pipeline to avoid per-frame setup overhead.
        self._activation_cm = None
        self._infer_cm = None
        self._infer_pipeline = None

    def _ensure_pipeline(self) -> None:
        if self._infer_pipeline is not None:
            return

        self._activation_cm = self._network_group.activate()
        self._activation_cm.__enter__()

        self._infer_cm = InferVStreams(
            self._network_group, self._in_params, self._out_params
        )
        self._infer_pipeline = self._infer_cm.__enter__()

    def infer(self, input_tensor: np.ndarray) -> dict[str, np.ndarray]:
        """
        input_tensor:
          - numpy array shaped to the HEF input vstream.
          - commonly NHWC uint8 or NCHW uint8 depending on compile.
        """
        self._ensure_pipeline()
        return self._infer_pipeline.infer({self._in_name: input_tensor})

    def close(self) -> None:
        # Close in reverse order of enter.
        if self._infer_cm is not None:
            try:
                self._infer_cm.__exit__(None, None, None)
            finally:
                self._infer_cm = None
                self._infer_pipeline = None

        if self._activation_cm is not None:
            try:
                self._activation_cm.__exit__(None, None, None)
            finally:
                self._activation_cm = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @property
    def input_shape(self) -> tuple[int, ...]:
        # Hailo vstream shape is commonly (H,W,C) or (N,H,W,C) depending on compile
        return tuple(getattr(self._input_infos[0], "shape", ()))

    @property
    def vstream_format(self) -> str:
        return str(getattr(self._vstream_format, "name", self._vstream_format))


class LaneDetectionAiModel:
    """LaneDetectionAiModel class."""

    def __init__(self, base_path: str, model_config_path: str):
        """
        Initialize the LaneDetectionAiModel class.

        Arguments:
            base_path -- The base path to the model.
            model_config_path -- The path to the model configuration file.
        """
        torch.backends.cudnn.benchmark = True

        self.camera_calibration = Calibration()
        self.camera_calibration.setup()
        self.config = get_config(os.path.join(base_path, model_config_path))
        self.config.test_model = os.path.join(base_path, self.config.test_model)

        # Default preprocessing (PyTorch/ONNX path): normalize float32
        self.image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        # NEW: backend selection
        self._backend: str = "cpu_pytorch"
        self._hailo_session: _HailoSession | None = None
        self._numpy_decode_cache: dict[
            tuple[object, ...], dict[str, np.ndarray | float]
        ] = {}
        self._hailo_output_map_cache: dict[str, object] | None = None
        self.load_model()

    def load_model(self):
        """
        Load the model.

        Returns:
            torch.nn.Module -- The model.
        """
        self.config.batch_size = 1

        assert self.config.backbone in [
            "9",
            "18",
            "34",
            "50",
            "101",
            "152",
            "50next",
            "101next",
            "50wide",
            "101wide",
        ]

        if self.config.test_model.endswith(".pth"):
            self._backend = "cpu_pytorch"
            self._load_pytorch_model()
        elif self.config.test_model.endswith(".onnx"):
            self._backend = "onnxruntime_cpu"
            self._load_onnx_model()
        elif self.config.test_model.endswith(".hef"):
            self._backend = "hailo"
            self._load_hailo_model()
        else:
            raise ValueError(f"Unsupported model file format: {self.config.test_model}")

        self._log_startup_configuration()

    def _log_startup_configuration(self) -> None:
        """Emit one-time startup diagnostics for model/backend verification."""
        model_path = str(self.config.test_model)
        model_exists = os.path.exists(model_path)
        model_size_mb = (
            os.path.getsize(model_path) / (1024.0 * 1024.0) if model_exists else -1.0
        )

        msg_parts = [
            "[LaneDetectionAiModel][startup]",
            f"backend={self._backend}",
            f"backbone={getattr(self.config, 'backbone', 'unknown')}",
            f"model={model_path}",
            f"exists={model_exists}",
        ]

        if model_exists:
            msg_parts.append(f"size_mb={model_size_mb:.2f}")

        if self._backend == "hailo":
            hailo_preprocess = getattr(
                self.config,
                "hailo_preprocess",
                getattr(self.config, "hailo_preprocessor", "baked"),
            )
            hailo_decode_mode = str(
                getattr(self.config, "hailo_decode_mode", "softmax")
            ).lower()
            msg_parts.append(f"hailo_preprocess={hailo_preprocess}")
            msg_parts.append(f"hailo_decode_mode={hailo_decode_mode}")

            if self._hailo_session is not None:
                msg_parts.append(f"hailo_input_shape={self._hailo_session.input_shape}")
                msg_parts.append(
                    f"hailo_vstream_format={self._hailo_session.vstream_format}"
                )

        print(" ".join(msg_parts))

    def _load_pytorch_model(self):
        """
        Load the PyTorch model.
        """
        self.net = get_model(self.config)

        ckpt = torch.load(self.config.test_model, map_location="cpu", weights_only=False)

        if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        if not isinstance(state_dict, dict):
            raise RuntimeError(
                f"Unsupported checkpoint structure in {self.config.test_model}: "
                f"expected dict-like state_dict, got {type(state_dict).__name__}"
            )

        # Keep tensor entries only (ignore metadata/optimizer artifacts).
        raw_items = {k: v for k, v in state_dict.items() if torch.is_tensor(v)}

        target_state = self.net.state_dict()
        target_keys = set(target_state.keys())

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
            changed = True
            out = key
            while changed:
                changed = False
                for p in prefixes:
                    if out.startswith(p):
                        out = out[len(p):]
                        changed = True
            return out

        candidates: list[dict[str, torch.Tensor]] = []

        # Candidate 1: original keys
        candidates.append(dict(raw_items))

        # Candidate 2..N: single-prefix strip
        for p in prefixes:
            remap = {}
            for k, v in raw_items.items():
                nk = _strip_once(k, p)
                remap[nk] = v
            candidates.append(remap)

        # Candidate N+1: iterative strip-until-stable
        remap_iter = {}
        for k, v in raw_items.items():
            remap_iter[_strip_iterative(k)] = v
        candidates.append(remap_iter)

        def _score(cand: dict[str, torch.Tensor]) -> tuple[int, int, int]:
            cand_keys = set(cand.keys())
            matched = len(cand_keys & target_keys)
            missing = len(target_keys - cand_keys)
            unexpected = len(cand_keys - target_keys)
            return matched, missing, unexpected

        best = None
        best_score = (-1, 10**9, 10**9)  # maximize matched, minimize missing/unexpected
        for cand in candidates:
            s = _score(cand)
            if s[0] > best_score[0] or (s[0] == best_score[0] and (s[1] + s[2]) < (best_score[1] + best_score[2])):
                best = cand
                best_score = s

        assert best is not None
        missing, unexpected = self.net.load_state_dict(best, strict=False)

        total_target = max(1, len(target_keys))
        coverage = 1.0 - (len(missing) / total_target)

        # Require strong match; otherwise this is likely wrong architecture/checkpoint.
        if coverage < 0.90:
            ckpt_key_sample = list(raw_items.keys())[:12]
            missing_sample = list(missing)[:12]
            unexpected_sample = list(unexpected)[:12]
            raise RuntimeError(
                "Checkpoint/model mismatch while loading PyTorch weights.\n"
                f"file={self.config.test_model}\n"
                f"matched={best_score[0]}, missing={len(missing)}, unexpected={len(unexpected)}, coverage={coverage:.3f}\n"
                f"sample_missing={missing_sample}\n"
                f"sample_unexpected={unexpected_sample}\n"
                f"sample_ckpt_keys={ckpt_key_sample}\n"
                "Likely causes: wrong backbone/head config, wrong checkpoint file, or incompatible training code."
            )

        self.net.eval()

    def _load_onnx_model(self):
        """
        Load the ONNX model.
        """
        self.ort_session = ort.InferenceSession(self.config.test_model)

    def _load_hailo_model(self):
        """
        Load the Hailo HEF model.
        """
        self._hailo_session = _HailoSession(self.config.test_model)

        # If your HEF does NOT include preprocessing, keep PyTorch normalization.
        # Default stays "baked" to preserve current behavior unless configured.
        # Support both keys; many configs in this repo still use `hailo_preprocessor`.
        hailo_preprocess = getattr(
            self.config,
            "hailo_preprocess",
            getattr(self.config, "hailo_preprocessor", "baked"),
        )
        if hailo_preprocess == "baked":
            self.image_transform = None  # feed uint8
        elif hailo_preprocess == "pytorch":
            pass  # keep default Normalize pipeline
        else:
            raise ValueError(f"Unknown hailo_preprocess={hailo_preprocess!r}")

    def _prepare_hailo_input(self, image: np.ndarray) -> np.ndarray:
        assert self._hailo_session is not None

        # Optional explicit BGR->RGB correction (enable via config if needed)
        if (
            getattr(self.config, "hailo_assume_bgr", False)
            and image.ndim == 3
            and image.shape[2] == 3
        ):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # If we're using PyTorch-style preprocessing, feed float32 NCHW (like ONNXRuntime path)
        if self.image_transform is not None:
            x_t = self.image_transform(image)[None, :, :, :]
            return x_t.cpu().numpy()

        # Otherwise feed uint8 in the layout the HEF expects.
        shape = self._hailo_session.input_shape

        # Heuristic: if last dim is 3, treat as NHWC; else treat as NCHW.
        if len(shape) >= 3 and shape[-1] == 3:
            return image[None, ...].astype(np.uint8)  # NHWC
        return np.transpose(image, (2, 0, 1))[None, ...].astype(np.uint8)  # NCHW

    def predict(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Predict the lanes in the image.

        Arguments:
            image -- The image.

        Returns:
            List[np.ndarray] -- The lanes.
        """
        original_image_height = image.shape[0]
        original_image_width = image.shape[1]

        with Timer(name="predict_total", filter_strength=40):
            with Timer(name="image_transform", filter_strength=40):
                image = cv2.resize(
                    image,
                    (
                        self.config.train_width,
                        int(self.config.train_height / self.config.crop_ratio),
                    ),
                )
                if len(image.shape) != 3:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

                # Crop last train_height rows (keeps behavior identical)
                image = image[-self.config.train_height :, :, :]

                if self._backend == "hailo":
                    x = self._prepare_hailo_input(image)
                else:
                    x_t = self.image_transform(image)
                    x_t = x_t[None, :, :, :]
                    x = x_t

            hailo_out: dict[str, np.ndarray] | None = None

            with Timer(name="inference", filter_strength=5):
                if self.config.test_model.endswith(".pth"):
                    with torch.inference_mode():
                        pred = self.net(x)
                elif self.config.test_model.endswith(".onnx"):
                    x_np = x.cpu().numpy()
                    out = self.ort_session.run(None, {"input": x_np})
                    pred = {
                        "loc_row": torch.from_numpy(out[0]),
                        "exist_row": torch.from_numpy(out[1]),
                        "loc_col": torch.from_numpy(out[2]),
                        "exist_col": torch.from_numpy(out[3]),
                    }
                elif self.config.test_model.endswith(".hef"):
                    assert self._hailo_session is not None
                    with Timer(name="inference_call", filter_strength=5):
                        hailo_out = self._hailo_session.infer(x)
                    pred = None
                else:
                    raise ValueError(
                        f"Unsupported model file format: {self.config.test_model}"
                    )

            with Timer(name="pred2coords", filter_strength=40):
                if self.config.test_model.endswith(".hef"):
                    assert hailo_out is not None
                    with Timer(name="inference_postmap", filter_strength=5):
                        pred_np = self._hailo_outputs_to_pred(hailo_out, as_torch=False)
                    hailo_decode_mode = str(
                        getattr(self.config, "hailo_decode_mode", "softmax")
                    ).lower()
                    if hailo_decode_mode == "argmax":
                        with Timer(name="decode_argmax", filter_strength=40):
                            coords = self.pred2coords_argmax_numpy(
                                pred_np,
                                self.config.row_anchor,
                                self.config.col_anchor,
                                original_image_width=original_image_width,
                                original_image_height=original_image_height,
                            )
                    elif hailo_decode_mode == "softmax":
                        with Timer(name="decode_softmax", filter_strength=40):
                            coords = self.pred2coords_optimized_numpy(
                                pred_np,
                                self.config.row_anchor,
                                self.config.col_anchor,
                                original_image_width=original_image_width,
                                original_image_height=original_image_height,
                            )
                    else:
                        raise ValueError(
                            f"Unsupported hailo_decode_mode={hailo_decode_mode!r}. "
                            "Use 'softmax' or 'argmax'."
                        )
                else:
                    with Timer(name="decode_torch", filter_strength=40):
                        coords = self.pred2coords_optimized(
                            pred,
                            self.config.row_anchor,
                            self.config.col_anchor,
                            original_image_width=original_image_width,
                            original_image_height=original_image_height,
                        )

            with Timer(name="lane_merge", filter_strength=40):
                if len(coords[0]) > 0 and len(coords[3]) > 0:
                    left_lane = np.vstack([coords[0], coords[3]])
                elif len(coords[0]) == 0 and len(coords[3]) > 0:
                    left_lane = np.array(coords[3])
                elif len(coords[0]) > 0 and len(coords[3]) == 0:
                    left_lane = np.array(coords[0])
                else:
                    left_lane = None

                if len(coords[1]) > 0 and len(coords[4]) > 0:
                    center_lane = np.vstack([coords[1], coords[4]])
                elif len(coords[1]) == 0 and len(coords[4]) > 0:
                    center_lane = np.array(coords[4])
                elif len(coords[1]) > 0 and len(coords[4]) == 0:
                    center_lane = np.array(coords[1])
                else:
                    center_lane = None

                if len(coords[2]) > 0 and len(coords[5]) > 0:
                    right_lane = np.vstack([coords[2], coords[5]])
                elif len(coords[2]) == 0 and len(coords[5]) > 0:
                    right_lane = np.array(coords[5])
                elif len(coords[2]) > 0 and len(coords[5]) == 0:
                    right_lane = np.array(coords[2])
                else:
                    right_lane = None

        return [left_lane, center_lane, right_lane]

    def _hailo_outputs_to_pred(
        self, out: dict[str, np.ndarray], as_torch: bool = True
    ) -> dict[str, torch.Tensor] | dict[str, np.ndarray]:
        """Map and reshape Hailo outputs into the tensors expected by post-processing.

        The post-processing expects:
          - loc_row:   [B, grid, num_row, num_lanes]
          - exist_row: [B, 2,    num_row, num_lanes]
          - loc_col:   [B, grid, num_col, num_lanes]
          - exist_col: [B, 2,    num_col, num_lanes]

        Some HEF pipelines output flattened tensors (e.g. [1, N] or [N]).
        This function matches outputs primarily by element-count derived from config and
        reshapes to the expected 4D tensors.
        """

        def _numel(arr: np.ndarray) -> int:
            try:
                return int(arr.size)
            except Exception:
                return int(np.prod(arr.shape))

        def _reshape_to(
            arr: np.ndarray, shape: tuple[int, ...]
        ) -> torch.Tensor | np.ndarray:
            # Always reshape via numpy to avoid torch view/contiguity surprises.
            reshaped = np.reshape(arr, shape)
            if as_torch:
                t = torch.from_numpy(reshaped)
                if t.dtype != torch.float32:
                    t = t.to(torch.float32)
                return t
            return reshaped.astype(np.float32, copy=False)

        # Fast-path: reuse output->tensor mapping learned on first valid frame.
        map_cache = self._hailo_output_map_cache
        if map_cache is not None:
            try:
                loc_row_arr = out[str(map_cache["loc_row_name"])]
                exist_row_arr = out[str(map_cache["exist_row_name"])]
                loc_col_arr = out[str(map_cache["loc_col_name"])]
                exist_col_arr = out[str(map_cache["exist_col_name"])]

                if (
                    _numel(loc_row_arr) == int(map_cache["loc_row_numel"])
                    and _numel(exist_row_arr) == int(map_cache["exist_row_numel"])
                    and _numel(loc_col_arr) == int(map_cache["loc_col_numel"])
                    and _numel(exist_col_arr) == int(map_cache["exist_col_numel"])
                ):
                    return {
                        "loc_row": _reshape_to(
                            loc_row_arr,
                            tuple(map_cache["loc_row_shape"]),
                        ),
                        "exist_row": _reshape_to(
                            exist_row_arr,
                            tuple(map_cache["exist_row_shape"]),
                        ),
                        "loc_col": _reshape_to(
                            loc_col_arr,
                            tuple(map_cache["loc_col_shape"]),
                        ),
                        "exist_col": _reshape_to(
                            exist_col_arr,
                            tuple(map_cache["exist_col_shape"]),
                        ),
                    }
            except Exception:
                # Fall back to robust remapping if any mismatch occurs.
                pass

        num_lanes = int(getattr(self.config, "num_lanes", 3))
        num_row = int(
            getattr(self.config, "num_row", len(getattr(self.config, "row_anchor", [])))
        )
        num_col = int(
            getattr(self.config, "num_col", len(getattr(self.config, "col_anchor", [])))
        )
        griding_num = int(getattr(self.config, "griding_num", 200))
        num_cell_row = int(getattr(self.config, "num_cell_row", 0) or 0)
        num_cell_col = int(getattr(self.config, "num_cell_col", 0) or 0)

        # Different exports use different "grid" conventions:
        # - some use griding_num (often 200/201)
        # - sparse variants often use num_cell_row/num_cell_col (often 100)
        row_grid_candidates = [griding_num, griding_num + 1]
        col_grid_candidates = [griding_num, griding_num + 1]
        if num_cell_row > 0:
            row_grid_candidates.extend([num_cell_row, num_cell_row + 1])
        if num_cell_col > 0:
            col_grid_candidates.extend([num_cell_col, num_cell_col + 1])

        # De-duplicate while preserving order
        def _uniq(seq: list[int]) -> list[int]:
            seen: set[int] = set()
            out_list: list[int] = []
            for v in seq:
                if v not in seen:
                    seen.add(v)
                    out_list.append(v)
            return out_list

        row_grid_candidates = _uniq(row_grid_candidates)
        col_grid_candidates = _uniq(col_grid_candidates)

        # Expected element counts
        exist_row_elems = 2 * num_row * num_lanes
        exist_col_elems = 2 * num_col * num_lanes
        loc_row_elems_candidates = [
            g * num_row * num_lanes for g in row_grid_candidates
        ]
        loc_col_elems_candidates = [
            g * num_col * num_lanes for g in col_grid_candidates
        ]

        # Work on a mutable list of outputs
        remaining: list[tuple[str, np.ndarray]] = list(out.items())

        def _pop_by_name(substrs: list[str]) -> tuple[str, np.ndarray] | None:
            for i, (k, v) in enumerate(remaining):
                lk = k.lower()
                if any(s in lk for s in substrs):
                    return remaining.pop(i)
            return None

        def _pop_by_numel(expected: int) -> tuple[str, np.ndarray] | None:
            for i, (k, v) in enumerate(remaining):
                if _numel(v) == expected:
                    return remaining.pop(i)
            return None

        # Prefer semantic names first (safer when multiple outputs share the same numel).
        exist_row_item = _pop_by_name(["exist_row", "existrow"]) or _pop_by_numel(exist_row_elems)  # type: ignore[assignment]
        exist_col_item = _pop_by_name(["exist_col", "existcol"]) or _pop_by_numel(exist_col_elems)  # type: ignore[assignment]

        # Loc outputs: match by candidate element-counts (grid may be g or g+1)
        def _pop_by_numel_any(
            expected_list: list[int],
        ) -> tuple[str, np.ndarray] | None:
            for exp in expected_list:
                hit = _pop_by_numel(exp)
                if hit is not None:
                    return hit
            return None

        loc_row_item = _pop_by_name(["loc_row", "locrow"]) or _pop_by_numel_any(
            loc_row_elems_candidates
        )
        loc_col_item = _pop_by_name(["loc_col", "loccol"]) or _pop_by_numel_any(
            loc_col_elems_candidates
        )

        if (
            exist_row_item is None
            or exist_col_item is None
            or loc_row_item is None
            or loc_col_item is None
        ):
            shapes = {k: list(v.shape) for k, v in out.items()}
            numels = {k: _numel(v) for k, v in out.items()}
            raise ValueError(
                "Failed to map HEF outputs to expected tensors. "
                f"Got outputs: shapes={shapes}, numels={numels}. "
                f"Expected exist_row={exist_row_elems}, exist_col={exist_col_elems}, "
                f"loc_row one of {loc_row_elems_candidates}, loc_col one of {loc_col_elems_candidates}."
            )

        exist_row_name, exist_row_arr = exist_row_item
        exist_col_name, exist_col_arr = exist_col_item
        loc_row_name, loc_row_arr = loc_row_item
        loc_col_name, loc_col_arr = loc_col_item

        # Infer grid sizes for row/col from element counts.
        loc_row_numel = _numel(loc_row_arr)
        loc_col_numel = _numel(loc_col_arr)

        grid_row = loc_row_numel // (num_row * num_lanes)
        grid_col = loc_col_numel // (num_col * num_lanes)

        if grid_row not in row_grid_candidates or grid_col not in col_grid_candidates:
            # Still try to proceed, but raise a clearer error.
            raise ValueError(
                "HEF output sizes do not match expected grid sizes. "
                f"grid_row={grid_row} (candidates={row_grid_candidates}), "
                f"grid_col={grid_col} (candidates={col_grid_candidates}), "
                f"loc_row_numel={loc_row_numel}, loc_col_numel={loc_col_numel}."
            )

        pred = {
            "loc_row": _reshape_to(loc_row_arr, (1, grid_row, num_row, num_lanes)),
            "exist_row": _reshape_to(exist_row_arr, (1, 2, num_row, num_lanes)),
            "loc_col": _reshape_to(loc_col_arr, (1, grid_col, num_col, num_lanes)),
            "exist_col": _reshape_to(exist_col_arr, (1, 2, num_col, num_lanes)),
        }

        # Cache mapping metadata for subsequent frames.
        self._hailo_output_map_cache = {
            "loc_row_name": loc_row_name,
            "exist_row_name": exist_row_name,
            "loc_col_name": loc_col_name,
            "exist_col_name": exist_col_name,
            "loc_row_numel": int(loc_row_numel),
            "exist_row_numel": int(_numel(exist_row_arr)),
            "loc_col_numel": int(loc_col_numel),
            "exist_col_numel": int(_numel(exist_col_arr)),
            "loc_row_shape": (1, grid_row, num_row, num_lanes),
            "exist_row_shape": (1, 2, num_row, num_lanes),
            "loc_col_shape": (1, grid_col, num_col, num_lanes),
            "exist_col_shape": (1, 2, num_col, num_lanes),
        }

        return pred

    @staticmethod
    def _build_center_lookup_tables(
        grid_size: int, local_width: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Precompute window index tables for each possible center index."""
        win = 2 * local_width + 1
        centers = np.arange(grid_size, dtype=np.int64)[:, None]  # [G,1]
        offsets = np.arange(win, dtype=np.int64)[None, :]  # [1,win]

        start = np.maximum(centers - local_width, 0)
        end = np.minimum(centers + local_width, grid_size - 1)

        inds = start + offsets  # [G,win]
        valid_mask = inds <= end
        inds_clamped = np.minimum(inds, grid_size - 1)
        inds_float = inds_clamped.astype(np.float32, copy=False)

        return inds_clamped, valid_mask, inds_float

    def _get_numpy_decode_cache(
        self,
        row_anchor: np.ndarray,
        col_anchor: np.ndarray,
        local_width: int,
        original_image_width: int,
        original_image_height: int,
        num_grid_row: int,
        num_grid_col: int,
    ) -> dict[str, np.ndarray | float]:
        row_anchor_arr = np.asarray(row_anchor, dtype=np.float32)
        col_anchor_arr = np.asarray(col_anchor, dtype=np.float32)

        key = (
            local_width,
            original_image_width,
            original_image_height,
            num_grid_row,
            num_grid_col,
            row_anchor_arr.tobytes(),
            col_anchor_arr.tobytes(),
        )

        hit = self._numpy_decode_cache.get(key)
        if hit is not None:
            return hit

        row_scale = (
            original_image_width / (num_grid_row - 1)
            if num_grid_row > 1
            else float(original_image_width)
        )
        col_scale = (
            original_image_height / (num_grid_col - 1)
            if num_grid_col > 1
            else float(original_image_height)
        )

        row_anchor_coords = (row_anchor_arr * float(original_image_height)).astype(
            np.int32
        )
        col_anchor_coords = (col_anchor_arr * float(original_image_width)).astype(
            np.int32
        )

        row_inds, row_mask, row_inds_float = self._build_center_lookup_tables(
            num_grid_row, local_width
        )
        col_inds, col_mask, col_inds_float = self._build_center_lookup_tables(
            num_grid_col, local_width
        )

        cache_entry: dict[str, np.ndarray | float] = {
            "row_scale": float(row_scale),
            "col_scale": float(col_scale),
            "row_anchor_coords": row_anchor_coords,
            "col_anchor_coords": col_anchor_coords,
            "row_inds": row_inds,
            "row_mask": row_mask,
            "row_inds_float": row_inds_float,
            "col_inds": col_inds,
            "col_mask": col_mask,
            "col_inds_float": col_inds_float,
        }
        self._numpy_decode_cache[key] = cache_entry
        return cache_entry

    def pred2coords_optimized_numpy(
        self,
        pred: dict[str, np.ndarray],
        row_anchor: np.ndarray,
        col_anchor: np.ndarray,
        local_width: int = 1,
        original_image_width: int = 1640,
        original_image_height: int = 590,
    ) -> List[np.ndarray]:
        """NumPy-based coordinate decoding for Hailo path (no Torch overhead)."""
        loc_row = np.asarray(pred["loc_row"][0], dtype=np.float32)  # [G_row, C_row, L_row]
        loc_col = np.asarray(pred["loc_col"][0], dtype=np.float32)  # [G_col, C_col, L_col]

        num_grid_row, _, num_lane_row = loc_row.shape
        num_grid_col, _, num_lane_col = loc_col.shape

        max_indices_row = loc_row.argmax(axis=0)  # [C_row, L_row]
        valid_row = pred["exist_row"][0].argmax(axis=0).astype(bool)  # [C_row, L_row]

        max_indices_col = loc_col.argmax(axis=0)  # [C_col, L_col]
        valid_col = pred["exist_col"][0].argmax(axis=0).astype(bool)  # [C_col, L_col]

        cache = self._get_numpy_decode_cache(
            row_anchor=row_anchor,
            col_anchor=col_anchor,
            local_width=local_width,
            original_image_width=original_image_width,
            original_image_height=original_image_height,
            num_grid_row=num_grid_row,
            num_grid_col=num_grid_col,
        )

        row_scale = float(cache["row_scale"])
        col_scale = float(cache["col_scale"])
        row_anchor_coords = np.asarray(cache["row_anchor_coords"], dtype=np.int32)
        col_anchor_coords = np.asarray(cache["col_anchor_coords"], dtype=np.int32)
        row_inds = np.asarray(cache["row_inds"], dtype=np.int64)
        row_mask = np.asarray(cache["row_mask"], dtype=bool)
        row_inds_float = np.asarray(cache["row_inds_float"], dtype=np.float32)
        col_inds = np.asarray(cache["col_inds"], dtype=np.int64)
        col_mask = np.asarray(cache["col_mask"], dtype=bool)
        col_inds_float = np.asarray(cache["col_inds_float"], dtype=np.float32)

        def _decode_axis(
            loc_axis: np.ndarray,      # [G, C, L]
            valid_axis: np.ndarray,    # [C, L] bool
            max_axis: np.ndarray,      # [C, L]
            anchor_axis: np.ndarray,   # [C]
            scale: float,
            lane_count: int,
            row_mode: bool,
            center_inds: np.ndarray,   # [G, win]
            center_mask: np.ndarray,   # [G, win]
            center_inds_float: np.ndarray,  # [G, win]
        ) -> List[np.ndarray]:
            coords_axis: List[np.ndarray] = []

            for lane_i in range(lane_count):
                k_idx = np.flatnonzero(valid_axis[:, lane_i])
                if k_idx.size == 0:
                    coords_axis.append(np.empty((0, 2), dtype=np.int32))
                    continue

                center = max_axis[k_idx, lane_i]
                inds_clamped = center_inds[center]            # [M, win]
                valid_mask = center_mask[center]              # [M, win]
                inds_clamped_float = center_inds_float[center]  # [M, win]

                # [G, M] -> [M, G]
                logits_mg = np.transpose(loc_axis[:, k_idx, lane_i], (1, 0))
                logits_win = np.take_along_axis(logits_mg, inds_clamped, axis=1)

                # Mask invalid positions before softmax
                logits_win[~valid_mask] = np.float32(-1e30)
                logits_win = logits_win - np.max(logits_win, axis=1, keepdims=True)
                np.exp(logits_win, out=logits_win)
                logits_win /= np.sum(logits_win, axis=1, keepdims=True)

                pos = np.sum(logits_win * inds_clamped_float, axis=1) + 0.5
                pix = np.rint(pos * scale).astype(np.int32)
                anc = anchor_axis[k_idx].astype(np.int32, copy=False)

                if row_mode:
                    lane_coords = np.empty((pix.shape[0], 2), dtype=np.int32)
                    lane_coords[:, 0] = pix
                    lane_coords[:, 1] = anc
                else:
                    lane_coords = np.empty((pix.shape[0], 2), dtype=np.int32)
                    lane_coords[:, 0] = anc
                    lane_coords[:, 1] = pix

                coords_axis.append(lane_coords)

            return coords_axis

        coords_row = _decode_axis(
            loc_axis=loc_row,
            valid_axis=valid_row,
            max_axis=max_indices_row,
            anchor_axis=row_anchor_coords,
            scale=row_scale,
            lane_count=num_lane_row,
            row_mode=True,
            center_inds=row_inds,
            center_mask=row_mask,
            center_inds_float=row_inds_float,
        )

        coords_col = _decode_axis(
            loc_axis=loc_col,
            valid_axis=valid_col,
            max_axis=max_indices_col,
            anchor_axis=col_anchor_coords,
            scale=col_scale,
            lane_count=num_lane_col,
            row_mode=False,
            center_inds=col_inds,
            center_mask=col_mask,
            center_inds_float=col_inds_float,
        )

        return coords_row + coords_col

    def pred2coords_argmax_numpy(
        self,
        pred: dict[str, np.ndarray],
        row_anchor: np.ndarray,
        col_anchor: np.ndarray,
        original_image_width: int = 1640,
        original_image_height: int = 590,
    ) -> List[np.ndarray]:
        """Fastest NumPy decode: argmax-only (no local softmax refinement)."""
        loc_row = np.asarray(pred["loc_row"][0], dtype=np.float32)  # [G_row, C_row, L_row]
        loc_col = np.asarray(pred["loc_col"][0], dtype=np.float32)  # [G_col, C_col, L_col]

        num_grid_row, _, num_lane_row = loc_row.shape
        num_grid_col, _, num_lane_col = loc_col.shape

        max_indices_row = loc_row.argmax(axis=0)  # [C_row, L_row]
        valid_row = pred["exist_row"][0].argmax(axis=0).astype(bool)  # [C_row, L_row]

        max_indices_col = loc_col.argmax(axis=0)  # [C_col, L_col]
        valid_col = pred["exist_col"][0].argmax(axis=0).astype(bool)  # [C_col, L_col]

        cache = self._get_numpy_decode_cache(
            row_anchor=row_anchor,
            col_anchor=col_anchor,
            local_width=1,
            original_image_width=original_image_width,
            original_image_height=original_image_height,
            num_grid_row=num_grid_row,
            num_grid_col=num_grid_col,
        )

        row_scale = float(cache["row_scale"])
        col_scale = float(cache["col_scale"])
        row_anchor_coords = np.asarray(cache["row_anchor_coords"], dtype=np.int32)
        col_anchor_coords = np.asarray(cache["col_anchor_coords"], dtype=np.int32)

        coords_row: List[np.ndarray] = []
        for lane_i in range(num_lane_row):
            k_idx = np.flatnonzero(valid_row[:, lane_i])
            if k_idx.size == 0:
                coords_row.append(np.empty((0, 2), dtype=np.int32))
                continue

            center = max_indices_row[k_idx, lane_i].astype(np.float32)
            pix = np.rint((center + 0.5) * row_scale).astype(np.int32)
            anc = row_anchor_coords[k_idx].astype(np.int32, copy=False)

            lane_coords = np.empty((pix.shape[0], 2), dtype=np.int32)
            lane_coords[:, 0] = pix
            lane_coords[:, 1] = anc
            coords_row.append(lane_coords)

        coords_col: List[np.ndarray] = []
        for lane_i in range(num_lane_col):
            k_idx = np.flatnonzero(valid_col[:, lane_i])
            if k_idx.size == 0:
                coords_col.append(np.empty((0, 2), dtype=np.int32))
                continue

            center = max_indices_col[k_idx, lane_i].astype(np.float32)
            pix = np.rint((center + 0.5) * col_scale).astype(np.int32)
            anc = col_anchor_coords[k_idx].astype(np.int32, copy=False)

            lane_coords = np.empty((pix.shape[0], 2), dtype=np.int32)
            lane_coords[:, 0] = anc
            lane_coords[:, 1] = pix
            coords_col.append(lane_coords)

        return coords_row + coords_col

    @staticmethod
    def pred2coords(
        pred,
        row_anchor,
        col_anchor,
        local_width=1,
        original_image_width=1640,
        original_image_height=590,
    ):
        """
        Convert the prediction to coordinates.

        Arguments:
            pred -- Prediction.
            row_anchor -- Row anchor.
            col_anchor -- Column anchor.

        Keyword Arguments:
            local_width -- Local Width (default: {1})
            original_image_width -- Original Image width (default: {1640})
            original_image_height -- Original Image height (default: {590})

        Returns:
            List[np.ndarray] -- The coordinates.
        """
        batch_size, num_grid_row, num_cls_row, num_lane_row = pred["loc_row"].shape
        batch_size, num_grid_col, num_cls_col, num_lane_col = pred["loc_col"].shape

        max_indices_row = pred["loc_row"].argmax(1).cpu()
        # n , num_cls, num_lanes
        valid_row = pred["exist_row"].argmax(1).cpu()
        # n, num_cls, num_lanes

        max_indices_col = pred["loc_col"].argmax(1).cpu()
        # n , num_cls, num_lanes
        valid_col = pred["exist_col"].argmax(1).cpu()

        # n, num_cls, num_lanes

        pred["loc_row"] = pred["loc_row"].cpu()
        pred["loc_col"] = pred["loc_col"].cpu()

        coords = []

        row_lane_idx = [0, 1, 2]
        col_lane_idx = [0, 1, 2]

        for i in row_lane_idx:
            tmp = []
            for k in range(valid_row.shape[1]):
                if valid_row[0, k, i]:
                    all_ind = torch.tensor(
                        list(
                            range(
                                max(0, max_indices_row[0, k, i] - local_width),
                                min(
                                    num_grid_row - 1,
                                    max_indices_row[0, k, i] + local_width,
                                )
                                + 1,
                            )
                        )
                    )

                    out_tmp = (
                        pred["loc_row"][0, all_ind, k, i].softmax(0) * all_ind.float()
                    ).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_row - 1) * original_image_width
                    tmp.append(
                        (int(out_tmp), int(row_anchor[k] * original_image_height))
                    )
            coords.append(tmp)

        for i in col_lane_idx:
            tmp = []
            for k in range(valid_col.shape[1]):
                if valid_col[0, k, i]:
                    all_ind = torch.tensor(
                        list(
                            range(
                                max(0, max_indices_col[0, k, i] - local_width),
                                min(
                                    num_grid_col - 1,
                                    max_indices_col[0, k, i] + local_width,
                                )
                                + 1,
                            )
                        )
                    )

                    out_tmp = (
                        pred["loc_col"][0, all_ind, k, i].softmax(0) * all_ind.float()
                    ).sum() + 0.5

                    out_tmp = out_tmp / (num_grid_col - 1) * original_image_height
                    tmp.append(
                        (int(col_anchor[k] * original_image_width), int(out_tmp))
                    )
            coords.append(tmp)

        return coords

    @staticmethod
    def pred2coords_optimized(
        pred: dict[str, torch.Tensor],
        row_anchor: np.ndarray,
        col_anchor: np.ndarray,
        local_width: int = 1,
        original_image_width: int = 1640,
        original_image_height: int = 590,
    ) -> List[List[tuple[int, int]]]:
        """
        Convert the prediction to coordinates (Optimized Version).

        Keeps the same decoding rule as the original implementation:
        - argmax for center index
        - local window softmax-weighted average
        """
        with torch.inference_mode():
            loc_row = pred["loc_row"][0]   # [G_row, C_row, L_row]
            loc_col = pred["loc_col"][0]   # [G_col, C_col, L_col]

            device = loc_row.device
            _, num_cls_row, num_lane_row = loc_row.shape
            num_grid_row = loc_row.shape[0]

            _, num_cls_col, num_lane_col = loc_col.shape
            num_grid_col = loc_col.shape[0]

            max_indices_row = loc_row.argmax(dim=0)                 # [C_row, L_row]
            valid_row = pred["exist_row"][0].argmax(dim=0).bool()   # [C_row, L_row]

            max_indices_col = loc_col.argmax(dim=0)                 # [C_col, L_col]
            valid_col = pred["exist_col"][0].argmax(dim=0).bool()   # [C_col, L_col]

            row_scale = (
                original_image_width / (num_grid_row - 1)
                if num_grid_row > 1
                else float(original_image_width)
            )
            col_scale = (
                original_image_height / (num_grid_col - 1)
                if num_grid_col > 1
                else float(original_image_height)
            )

            # Keep truncation semantics close to old code (int()).
            row_anchor_coords = (
                torch.as_tensor(row_anchor, dtype=torch.float32, device=device)
                * float(original_image_height)
            ).to(torch.int32)
            col_anchor_coords = (
                torch.as_tensor(col_anchor, dtype=torch.float32, device=device)
                * float(original_image_width)
            ).to(torch.int32)

            win = 2 * local_width + 1
            offsets = torch.arange(win, device=device, dtype=torch.long)

            def _decode_axis(
                loc_axis: torch.Tensor,            # [G, C, L]
                valid_axis: torch.Tensor,          # [C, L] bool
                max_axis: torch.Tensor,            # [C, L]
                anchor_axis: torch.Tensor,         # [C]
                grid_size: int,
                scale: float,
                lane_count: int,
                row_mode: bool,                    # True -> (x,y), False -> (x,y) swapped
            ) -> List[List[tuple[int, int]]]:
                coords_axis: List[List[tuple[int, int]]] = []

                for lane_i in range(lane_count):
                    k_idx = torch.nonzero(valid_axis[:, lane_i], as_tuple=False).squeeze(1)
                    if k_idx.numel() == 0:
                        coords_axis.append([])
                        continue

                    center = max_axis[k_idx, lane_i]  # [M]
                    start = (center - local_width).clamp_min(0)
                    end = (center + local_width).clamp_max(grid_size - 1)

                    # Build fixed window per anchor, then mask invalid tail near borders
                    inds = start[:, None] + offsets[None, :]         # [M, win]
                    valid_mask = inds <= end[:, None]                # [M, win]
                    inds_clamped = inds.clamp_max(grid_size - 1)     # [M, win]

                    # Gather logits for each anchor/window
                    # loc_axis[:, k_idx, lane_i] -> [G, M] -> transpose -> [M, G]
                    logits_mg = loc_axis[:, k_idx, lane_i].transpose(0, 1).contiguous()
                    logits_win = logits_mg.gather(dim=1, index=inds_clamped)  # [M, win]

                    # Ignore out-of-range positions in softmax
                    min_val = torch.finfo(logits_win.dtype).min
                    logits_win = logits_win.masked_fill(~valid_mask, min_val)

                    probs = torch.softmax(logits_win, dim=1)
                    pos = (probs * inds_clamped.to(probs.dtype)).sum(dim=1) + 0.5
                    pix = torch.round(pos * scale).to(torch.int32)

                    anc = anchor_axis[k_idx].to(torch.int32)

                    if row_mode:
                        lane_coords = list(zip(pix.tolist(), anc.tolist()))
                    else:
                        lane_coords = list(zip(anc.tolist(), pix.tolist()))

                    coords_axis.append(lane_coords)

                return coords_axis

            coords_row = _decode_axis(
                loc_axis=loc_row,
                valid_axis=valid_row,
                max_axis=max_indices_row,
                anchor_axis=row_anchor_coords,
                grid_size=num_grid_row,
                scale=row_scale,
                lane_count=num_lane_row,
                row_mode=True,
            )

            coords_col = _decode_axis(
                loc_axis=loc_col,
                valid_axis=valid_col,
                max_axis=max_indices_col,
                anchor_axis=col_anchor_coords,
                grid_size=num_grid_col,
                scale=col_scale,
                lane_count=num_lane_col,
                row_mode=False,
            )

            return coords_row + coords_col
