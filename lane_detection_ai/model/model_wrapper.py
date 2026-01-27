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

        # Use AUTO formats; runtime will accept numpy arrays matching the vstream params.
        self._in_params = InputVStreamParams.make_from_network_group(
            self._network_group, format_type=FormatType.AUTO
        )
        self._out_params = OutputVStreamParams.make_from_network_group(
            self._network_group, format_type=FormatType.AUTO
        )

        self._in_name = self._input_infos[0].name
        self._out_names = [o.name for o in self._output_infos]

    def infer(self, input_tensor: np.ndarray) -> dict[str, np.ndarray]:
        """
        input_tensor:
          - numpy array shaped to the HEF input vstream.
          - commonly NHWC uint8 or NCHW uint8 depending on compile.
        """
        with self._network_group.activate():
            with InferVStreams(self._network_group, self._in_params, self._out_params) as infer_pipeline:
                outputs = infer_pipeline.infer({self._in_name: input_tensor})

        # outputs is already a dict[name] -> np.ndarray
        return outputs

    @property
    def input_shape(self) -> tuple[int, ...]:
        # Hailo vstream shape is commonly (H,W,C) or (N,H,W,C) depending on compile
        return tuple(getattr(self._input_infos[0], "shape", ()))


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

    def _load_pytorch_model(self):
        """
        Load the PyTorch model.
        """
        self.net = get_model(self.config)

        state_dict = torch.load(
            self.config.test_model, map_location="cpu", weights_only=True
        )["model"]
        compatible_state_dict = {}
        for k, v in state_dict.items():
            if "lane_detection_ai.module." in k:
                compatible_state_dict[k[7:]] = v
            else:
                compatible_state_dict[k] = v

        self.net.load_state_dict(compatible_state_dict, strict=False)
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
        hailo_preprocess = getattr(self.config, "hailo_preprocess", "baked")
        if hailo_preprocess == "baked":
            self.image_transform = None  # feed uint8
        elif hailo_preprocess == "pytorch":
            pass  # keep default Normalize pipeline
        else:
            raise ValueError(f"Unknown hailo_preprocess={hailo_preprocess!r}")

    def _prepare_hailo_input(self, image: np.ndarray) -> np.ndarray:
        assert self._hailo_session is not None

        # Optional explicit BGR->RGB correction (enable via config if needed)
        if getattr(self.config, "hailo_assume_bgr", False) and image.ndim == 3 and image.shape[2] == 3:
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
        with Timer(name="image_transform", filter_strength=40):
            image = cv2.resize(
                image,
                (
                    self.config.train_width,
                    int(self.config.train_height / self.config.crop_ratio),
                ),
            )
            if image.shape[0] != 3:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # Crop last train_height rows (keeps behavior identical)
            image = image[-self.config.train_height:, :, :]

            if self._backend == "hailo":
                x = self._prepare_hailo_input(image)
            else:
                x_t = self.image_transform(image)
                x_t = x_t[None, :, :, :]
                x = x_t

        with Timer(name="inference", filter_strength=40):
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
                out = self._hailo_session.infer(x)

                pred = self._hailo_outputs_to_pred(out)
            else:
                raise ValueError(f"Unsupported model file format: {self.config.test_model}")

        with Timer(name="pred2coords", filter_strength=40):
            coords = self.pred2coords_optimized(
                pred,
                self.config.row_anchor,
                self.config.col_anchor,
                original_image_width=self.camera_calibration.target_size[0],
                original_image_height=self.camera_calibration.target_size[1],
            )

        if len(coords[0]) > 0 and len(coords[3]) > 0:
            left_lane = np.vstack([coords[0], coords[3]])
        elif len(coords[0]) == 0 and len(coords[3]) > 0:
            left_lane = coords[3]
        elif len(coords[0]) > 0 and len(coords[3]) == 0:
            left_lane = coords[0]
        else:
            left_lane = None

        if len(coords[1]) > 0 and len(coords[4]) > 0:
            center_lane = np.vstack([coords[1], coords[4]])
        elif len(coords[1]) == 0 and len(coords[4]) > 0:
            center_lane = coords[4]
        elif len(coords[1]) > 0 and len(coords[4]) == 0:
            center_lane = coords[1]
        else:
            center_lane = None

        if len(coords[2]) > 0 and len(coords[5]) > 0:
            right_lane = np.vstack([coords[2], coords[5]])
        elif len(coords[2]) == 0 and len(coords[5]) > 0:
            right_lane = coords[5]
        elif len(coords[2]) > 0 and len(coords[5]) == 0:
            right_lane = coords[2]
        else:
            right_lane = None

        return [left_lane, center_lane, right_lane]

    def _hailo_outputs_to_pred(self, out: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
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

        def _to_torch(arr: np.ndarray) -> torch.Tensor:
            t = torch.from_numpy(arr)
            if t.dtype != torch.float32:
                t = t.to(torch.float32)
            return t

        def _numel(arr: np.ndarray) -> int:
            try:
                return int(arr.size)
            except Exception:
                return int(np.prod(arr.shape))

        def _reshape_to(arr: np.ndarray, shape: tuple[int, ...]) -> torch.Tensor:
            # Always reshape via numpy to avoid torch view/contiguity surprises.
            reshaped = np.reshape(arr, shape)
            return _to_torch(reshaped)

        num_lanes = int(getattr(self.config, "num_lanes", 3))
        num_row = int(getattr(self.config, "num_row", len(getattr(self.config, "row_anchor", []))))
        num_col = int(getattr(self.config, "num_col", len(getattr(self.config, "col_anchor", []))))
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
        loc_row_elems_candidates = [g * num_row * num_lanes for g in row_grid_candidates]
        loc_col_elems_candidates = [g * num_col * num_lanes for g in col_grid_candidates]

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

        # Prefer exact element-count matches; fall back to semantic substrings if present.
        exist_row_item = _pop_by_numel(exist_row_elems) or _pop_by_name(["exist_row", "existrow"])  # type: ignore[assignment]
        exist_col_item = _pop_by_numel(exist_col_elems) or _pop_by_name(["exist_col", "existcol"])  # type: ignore[assignment]

        # Loc outputs: match by candidate element-counts (grid may be g or g+1)
        def _pop_by_numel_any(expected_list: list[int]) -> tuple[str, np.ndarray] | None:
            for exp in expected_list:
                hit = _pop_by_numel(exp)
                if hit is not None:
                    return hit
            return None

        loc_row_item = _pop_by_numel_any(loc_row_elems_candidates) or _pop_by_name(
            ["loc_row", "locrow"]
        )
        loc_col_item = _pop_by_numel_any(loc_col_elems_candidates) or _pop_by_name(
            ["loc_col", "loccol"]
        )

        if exist_row_item is None or exist_col_item is None or loc_row_item is None or loc_col_item is None:
            shapes = {k: list(v.shape) for k, v in out.items()}
            numels = {k: _numel(v) for k, v in out.items()}
            raise ValueError(
                "Failed to map HEF outputs to expected tensors. "
                f"Got outputs: shapes={shapes}, numels={numels}. "
                f"Expected exist_row={exist_row_elems}, exist_col={exist_col_elems}, "
                f"loc_row one of {loc_row_elems_candidates}, loc_col one of {loc_col_elems_candidates}."
            )

        _, exist_row_arr = exist_row_item
        _, exist_col_arr = exist_col_item
        _, loc_row_arr = loc_row_item
        _, loc_col_arr = loc_col_item

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

        return pred

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

        Arguments:
            pred -- Prediction dictionary containing tensors ('loc_row', 'exist_row', 'loc_col', 'exist_col').
                    Assumes tensors are on the appropriate device (CPU or GPU).
            row_anchor -- Row anchor positions (normalized 0-1 or similar). NumPy array.
            col_anchor -- Column anchor positions (normalized 0-1 or similar). NumPy array.

        Keyword Arguments:
            local_width -- Local width for averaging (default: {1})
            original_image_width -- Original Image width (default: {1640})
            original_image_height -- Original Image height (default: {590})

        Returns:
            List[List[Tuple[int, int]]] -- The coordinates for each lane.
                                           Structure: [[(x,y), ...], [(x,y), ...], ...]
        """
        loc_row_tensor = pred["loc_row"]
        loc_col_tensor = pred["loc_col"]
        device = loc_row_tensor.device

        batch_size, num_grid_row, num_cls_row, num_lane_row = loc_row_tensor.shape
        batch_size, num_grid_col, num_cls_col, num_lane_col = loc_col_tensor.shape

        max_indices_row = loc_row_tensor[0].argmax(
            0
        )  # Shape: [num_cls_row, num_lane_row]
        valid_row = pred["exist_row"][0].argmax(0)  # Shape: [num_cls_row, num_lane_row]
        max_indices_col = loc_col_tensor[0].argmax(
            0
        )  # Shape: [num_cls_col, num_lane_col]
        valid_col = pred["exist_col"][0].argmax(0)  # Shape: [num_cls_col, num_lane_col]

        row_scale = (
            original_image_width / (num_grid_row - 1)
            if num_grid_row > 1
            else original_image_width
        )
        col_scale = (
            original_image_height / (num_grid_col - 1)
            if num_grid_col > 1
            else original_image_height
        )

        coords = []
        row_anchor_coords = (
            torch.from_numpy(row_anchor).float() * original_image_height
        ).int()
        col_anchor_coords = (
            torch.from_numpy(col_anchor).float() * original_image_width
        ).int()

        # Row Processing
        for i in range(num_lane_row):
            tmp = []
            valid_k_indices = torch.where(valid_row[:, i].cpu())[0]

            for k in valid_k_indices:
                max_idx = max_indices_row[k, i].item()

                start = max(0, max_idx - local_width)
                end = min(num_grid_row - 1, max_idx + local_width)

                all_ind = torch.arange(
                    start, end + 1, device=device, dtype=torch.float32
                )
                locs = loc_row_tensor[0, start : end + 1, k, i]

                # Softmax and weighted sum
                probs = locs.softmax(0)
                out_tmp = torch.sum(probs * all_ind) + 0.5

                # Scale to image coordinates and get y-coordinate from anchor
                x_coord_tensor = out_tmp * row_scale
                y_coord = row_anchor_coords[k].item()

                tmp.append((x_coord_tensor.round().int().item(), y_coord))
            coords.append(tmp)

        # Column Processing
        for i in range(num_lane_col):
            tmp = []
            valid_k_indices = torch.where(valid_col[:, i].cpu())[0]

            for k in valid_k_indices:
                max_idx = max_indices_col[k, i].item()

                start = max(0, max_idx - local_width)
                end = min(num_grid_col - 1, max_idx + local_width)

                all_ind = torch.arange(
                    start, end + 1, device=device, dtype=torch.float32
                )
                locs = loc_col_tensor[0, start : end + 1, k, i]

                # Softmax and weighted sum
                probs = locs.softmax(0)
                out_tmp = torch.sum(probs * all_ind) + 0.5

                # Scale to image coordinates and get x-coordinate from anchor
                y_coord_tensor = out_tmp * col_scale
                x_coord = col_anchor_coords[k].item()

                tmp.append((x_coord, y_coord_tensor.round().int().item()))
            coords.append(tmp)

        return coords
