"""Lane detection model with PyTorch, ONNX Runtime and Hailo-8 backends."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import onnxruntime as ort
import torch
from timing.timer import Timer

from lane_detection_ai.model.checkpoint import load_checkpoint
from lane_detection_ai.model.decode import decode_lanes, merge_lanes
from lane_detection_ai.model.hailo import OUTPUT_KEYS, HailoOutputMapper, HailoSession
from lane_detection_ai.model.utils.common import (
    IMAGE_MEAN,
    IMAGE_STD,
    get_config,
    get_model,
)

if TYPE_CHECKING:
    from collections.abc import Callable

SUPPORTED_BACKBONES = {
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
}
DECODE_MODES = {"softmax", "argmax"}


class LaneDetectionAiModel:
    """Lane detection on images, backend chosen by the model file extension.

    - `.pth`: PyTorch on the CPU (convert training checkpoints with
      scripts/convert_ckpt_to_pth.py)
    - `.onnx`: ONNX Runtime on the CPU (float32 input, or uint8 input if the
      normalization is part of the exported graph)
    - `.hef`: Hailo-8 accelerator with HailoRT

    Optional model config keys:
        assume_bgr: Convert color images from BGR to RGB (default False, also
            read from the old key hailo_assume_bgr).
        decode_mode: "softmax" (precise, default) or "argmax" (faster), also read
            from the old key hailo_decode_mode.
        hailo_preprocess: "baked" if the HEF contains the normalization and expects
            uint8 RGB input (default, what scripts/convert_pth_onnx.py exports),
            "pytorch" to normalize on the host. Also read from hailo_preprocessor.
        hailo_output_names: Dict from "loc_row", "exist_row", "loc_col" and
            "exist_col" to the HEF output names, if they can't be found
            automatically.
    """

    def __init__(
        self,
        base_path: str,
        model_config_path: str,
        log: Callable[[str], None] = print,
    ) -> None:
        """Load the model config and the model.

        Args:
            base_path: The base path of the package; the config path and the
                test_model path of the config are relative to it.
            model_config_path: The path of the model config.
            log: Function to log messages with.
        """
        self._log = log
        self.config = get_config(str(Path(base_path) / model_config_path))
        self.config.test_model = str(Path(base_path) / self.config.test_model)
        self.config.batch_size = 1

        if self.config.backbone not in SUPPORTED_BACKBONES:
            msg = f"Unsupported backbone {self.config.backbone!r}"
            raise ValueError(msg)

        cfg = self.config
        self._assume_bgr = bool(
            getattr(cfg, "assume_bgr", getattr(cfg, "hailo_assume_bgr", False))
        )
        self._decode_mode = str(
            getattr(cfg, "decode_mode", getattr(cfg, "hailo_decode_mode", "softmax"))
        ).lower()
        if self._decode_mode not in DECODE_MODES:
            msg = f"Unsupported decode_mode {self._decode_mode!r}, use {DECODE_MODES}"
            raise ValueError(msg)

        self._net: torch.nn.Module | None = None
        self._ort_session = None
        self._hailo_session: HailoSession | None = None
        self._hailo_outputs: HailoOutputMapper | None = None
        self._uint8_input = False
        self._channels_last = False

        model_path = self.config.test_model
        suffix = Path(model_path).suffix
        if suffix == ".pth":
            self.backend = "pytorch"
            self._load_pytorch_model(model_path)
        elif suffix == ".onnx":
            self.backend = "onnxruntime"
            self._load_onnx_model(model_path)
        elif suffix == ".hef":
            self.backend = "hailo"
            self._load_hailo_model(model_path)
        else:
            msg = f"Unsupported model file format: {model_path}"
            raise ValueError(msg)

        self._log_startup_configuration()

    def _load_pytorch_model(self, model_path: str) -> None:
        self._net = get_model(self.config)
        load_checkpoint(self._net, model_path, log=self._log)
        self._net.eval()

    def _load_onnx_model(self, model_path: str) -> None:
        self._ort_session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        model_input = self._ort_session.get_inputs()[0]
        self._ort_input_name = model_input.name
        # Exports of scripts/convert_pth_onnx.py contain the normalization
        self._uint8_input = model_input.type == "tensor(uint8)"

        output_names = [o.name for o in self._ort_session.get_outputs()]
        if set(OUTPUT_KEYS) <= set(output_names):
            self._ort_output_names = list(OUTPUT_KEYS)
        else:
            # Unnamed outputs are expected in the export order of convert_pth_onnx.py
            self._ort_output_names = output_names[: len(OUTPUT_KEYS)]

    def _load_hailo_model(self, model_path: str) -> None:
        preprocess = getattr(
            self.config,
            "hailo_preprocess",
            getattr(self.config, "hailo_preprocessor", "baked"),
        )
        if preprocess not in {"baked", "pytorch"}:
            msg = f"Unknown hailo_preprocess={preprocess!r}, use 'baked' or 'pytorch'"
            raise ValueError(msg)
        self._uint8_input = preprocess == "baked"

        self._hailo_session = HailoSession(
            model_path, input_format="uint8" if self._uint8_input else "float32"
        )
        self._channels_last = self._hailo_session.channels_last

        cfg = self.config
        self._hailo_outputs = HailoOutputMapper(
            {
                "loc_row": (cfg.num_cell_row, cfg.num_row, cfg.num_lanes),
                "exist_row": (2, cfg.num_row, cfg.num_lanes),
                "loc_col": (cfg.num_cell_col, cfg.num_col, cfg.num_lanes),
                "exist_col": (2, cfg.num_col, cfg.num_lanes),
            },
            output_names=getattr(cfg, "hailo_output_names", None),
            log=self._log,
        )

    def _log_startup_configuration(self) -> None:
        model_path = Path(self.config.test_model)
        parts = [
            "[LaneDetectionAiModel]",
            f"backend={self.backend}",
            f"backbone={self.config.backbone}",
            f"model={model_path}",
            f"size_mb={model_path.stat().st_size / 2**20:.2f}",
            f"input={'uint8' if self._uint8_input else 'float32 normalized'}",
            f"decode_mode={self._decode_mode}",
        ]
        if self._hailo_session is not None:
            parts.append(f"hef_input_shape={self._hailo_session.input_shape}")
        self._log(" ".join(parts))

    def close(self) -> None:
        """Release the Hailo device (if used). The model can't be used afterwards."""
        if self._hailo_session is not None:
            self._hailo_session.close()
            self._hailo_session = None

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize, crop and convert the image to the network input.

        Returns:
            The input [1, 3, H, W] (or [1, H, W, 3] for channels-last Hailo
            networks), uint8 RGB or float32 normalized RGB.
        """
        cfg = self.config
        image = cv2.resize(
            image, (cfg.train_width, int(cfg.train_height / cfg.crop_ratio))
        )
        if image.ndim == 2:  # noqa: PLR2004
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif self._assume_bgr:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # The network sees the bottom train_height rows
        image = image[-cfg.train_height :]

        if not self._uint8_input:
            image = (image.astype(np.float32) / 255.0 - IMAGE_MEAN) / IMAGE_STD
        if not self._channels_last:
            image = image.transpose(2, 0, 1)
        return np.ascontiguousarray(image[None])

    def _infer(self, model_input: np.ndarray) -> dict[str, np.ndarray]:
        """Run the network.

        Returns:
            The outputs "loc_row", "exist_row", "loc_col" and "exist_col".
        """
        if self._net is not None:
            with torch.inference_mode():
                pred = self._net(torch.from_numpy(model_input))
            return {key: pred[key].numpy() for key in OUTPUT_KEYS}

        if self._ort_session is not None:
            outputs = self._ort_session.run(
                self._ort_output_names, {self._ort_input_name: model_input}
            )
            return dict(zip(OUTPUT_KEYS, outputs, strict=True))

        if self._hailo_session is not None and self._hailo_outputs is not None:
            return self._hailo_outputs(self._hailo_session.infer(model_input))

        msg = "The model has been closed"
        raise RuntimeError(msg)

    def predict(self, image: np.ndarray) -> list[np.ndarray | None]:
        """Predict the lanes in the image.

        Args:
            image: Grayscale [H, W] or color [H, W, 3] image.

        Returns:
            Points [n, 2] (x, y in pixels of the given image) of every lane (left,
            center, right), None if a lane wasn't found.
        """
        height, width = image.shape[:2]

        with Timer(name="image_transform", filter_strength=40):
            model_input = self._preprocess(image)

        with Timer(name="inference", filter_strength=40):
            pred = self._infer(model_input)

        with Timer(name="pred2coords", filter_strength=40):
            row_lanes, col_lanes = decode_lanes(
                pred,
                self.config.row_anchor,
                self.config.col_anchor,
                width,
                height,
                refine=self._decode_mode == "softmax",
            )
            return merge_lanes(row_lanes, col_lanes)
