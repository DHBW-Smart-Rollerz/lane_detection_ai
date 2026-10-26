"""Inference on the Hailo-8 accelerator with HailoRT."""

from __future__ import annotations

import contextlib
import importlib
import sys
import time
from typing import TYPE_CHECKING, Self

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from types import ModuleType

OUTPUT_KEYS = ("loc_row", "exist_row", "loc_col", "exist_col")


def _import_hailo_platform() -> ModuleType:
    try:
        return importlib.import_module("hailo_platform")
    except ImportError as e:
        msg = (
            "The HailoRT Python bindings (hailo_platform) are not available for "
            f"{sys.executable} ({e!r}). Install the HailoRT wheel matching this "
            "Python version (see HAILO_INSTALL.md)."
        )
        raise RuntimeError(msg) from e


class HailoSession:
    """Blocking single-image inference of a compiled HEF network on a Hailo device.

    The input is fed in the host format given by `input_format` (HailoRT quantizes
    float32 input on the host). The outputs are always returned as dequantized
    float32, so they can be decoded like the outputs of the PyTorch/ONNX model.
    """

    def __init__(
        self,
        hef_path: str,
        *,
        input_format: str = "uint8",
        device_retries: int = 30,
        retry_delay_s: float = 0.1,
    ) -> None:
        """Load the HEF onto the Hailo device and create the inference pipeline.

        Args:
            hef_path: Path of the compiled network.
            input_format: Host format of the input, "uint8" (raw image) or
                "float32" (normalized image).
            device_retries: How often to retry if the device is still in use (e.g.
                by a node that is shutting down).
            retry_delay_s: Delay between the retries.
        """
        if input_format not in {"uint8", "float32"}:
            msg = f"Unsupported input_format {input_format!r}"
            raise ValueError(msg)

        hp = _import_hailo_platform()
        self._exit_stack = contextlib.ExitStack()
        self._vdevice = None

        hef = hp.HEF(hef_path)
        input_infos = hef.get_input_vstream_infos()
        if len(input_infos) != 1:
            msg = f"Expected a HEF with one input, got {len(input_infos)}"
            raise ValueError(msg)
        self.input_name: str = input_infos[0].name
        self.input_shape: tuple[int, ...] = tuple(input_infos[0].shape)
        self.output_names: list[str] = [o.name for o in hef.get_output_vstream_infos()]
        self.input_dtype = np.uint8 if input_format == "uint8" else np.float32

        self._vdevice = self._create_vdevice(hp, device_retries, retry_delay_s)
        try:
            configure_params = hp.ConfigureParams.create_from_hef(
                hef, interface=hp.HailoStreamInterface.PCIe
            )
            network_group = self._vdevice.configure(hef, configure_params)[0]
            input_params = hp.InputVStreamParams.make_from_network_group(
                network_group,
                format_type=getattr(hp.FormatType, input_format.upper()),
            )
            output_params = hp.OutputVStreamParams.make_from_network_group(
                network_group, format_type=hp.FormatType.FLOAT32
            )
            self._exit_stack.enter_context(network_group.activate())
            self._pipeline = self._exit_stack.enter_context(
                hp.InferVStreams(network_group, input_params, output_params)
            )
        except Exception:
            self.close()
            raise

    @staticmethod
    def _create_vdevice(hp: ModuleType, retries: int, delay_s: float) -> object:
        for attempt in range(retries):
            try:
                return hp.VDevice()
            except Exception as e:
                if attempt == retries - 1:
                    msg = (
                        "Failed to open the Hailo device, it is probably used by "
                        "another process. Find it with `sudo fuser -v /dev/hailo*`."
                    )
                    raise RuntimeError(msg) from e
                time.sleep(delay_s)
        msg = "device_retries must be at least 1"
        raise ValueError(msg)

    @property
    def channels_last(self) -> bool:
        """Return True if the network expects HWC input (the usual Hailo layout)."""
        return self.input_shape[-1] == 3  # noqa: PLR2004

    def infer(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Run the network on one image.

        Args:
            image: Image [1, ...] in the layout and dtype of the HEF input.

        Returns:
            The float32 outputs by output vstream name.
        """
        return self._pipeline.infer({self.input_name: image})

    def close(self) -> None:
        """Stop the pipeline and release the device (can be called repeatedly)."""
        try:
            self._exit_stack.close()
        finally:
            if self._vdevice is not None:
                self._vdevice.release()
                self._vdevice = None

    def __enter__(self) -> Self:
        """Use the session as context manager."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the session."""
        self.close()


class HailoOutputMapper:
    """Maps the output vstreams of a HEF to the network outputs.

    The Hailo compiler renames the outputs and may return them flattened or in
    HWC layout. The mapping is found once (by configured names, by names that
    contain the output key or by the element count) and reused for every frame.
    """

    def __init__(
        self,
        expected_shapes: Mapping[str, tuple[int, int, int]],
        output_names: Mapping[str, str] | None = None,
        log: Callable[[str], None] = print,
    ) -> None:
        """Initialize the mapper.

        Args:
            expected_shapes: Shape [grid/2, anchors, lanes] (without batch) of every
                network output in OUTPUT_KEYS.
            output_names: Optional explicit vstream name of every network output.
            log: Function to log the found mapping with.
        """
        self._expected_shapes = dict(expected_shapes)
        self._configured_names = dict(output_names) if output_names else None
        self._log = log
        self._mapping: dict[str, str] | None = None

    def _find_mapping(self, outputs: Mapping[str, np.ndarray]) -> dict[str, str]:
        if self._configured_names is not None:
            missing = set(OUTPUT_KEYS) - set(self._configured_names)
            unknown = set(self._configured_names.values()) - set(outputs)
            if missing or unknown:
                msg = (
                    f"Invalid hailo_output_names: missing keys {sorted(missing)}, "
                    f"unknown outputs {sorted(unknown)}; outputs: {sorted(outputs)}"
                )
                raise ValueError(msg)
            return dict(self._configured_names)

        mapping: dict[str, str] = {}
        remaining = list(outputs)
        for key in OUTPUT_KEYS:
            name = next(
                (
                    n
                    for n in remaining
                    if key in n.lower() or key.replace("_", "") in n.lower()
                ),
                None,
            )
            if name is not None:
                mapping[key] = name
                remaining.remove(name)

        ambiguous = False
        for key in OUTPUT_KEYS:
            if key in mapping:
                continue
            size = int(np.prod(self._expected_shapes[key]))
            candidates = [n for n in remaining if outputs[n].size == size]
            if not candidates:
                shapes = {n: outputs[n].shape for n in outputs}
                expected = {k: (1, *s) for k, s in self._expected_shapes.items()}
                msg = (
                    f"No HEF output matches {key}. Outputs: {shapes}, expected "
                    f"(any layout): {expected}. Set hailo_output_names in the model "
                    "config or check the anchor/grid sizes."
                )
                raise ValueError(msg)
            ambiguous |= len(candidates) > 1
            mapping[key] = candidates[0]
            remaining.remove(candidates[0])

        message = f"[LaneDetectionAiModel] HEF output mapping: {mapping}"
        if ambiguous:
            message += (
                " (outputs with equal size were assigned in HEF order, set "
                "hailo_output_names in the model config to make it explicit)"
            )
        self._log(message)
        return mapping

    def _reshape(self, key: str, array: np.ndarray) -> np.ndarray:
        grid, anchors, lanes = self._expected_shapes[key]
        array = np.asarray(array, dtype=np.float32)
        shape = array.shape[1:] if array.ndim == 4 else array.shape  # noqa: PLR2004
        if tuple(shape) == (anchors, lanes, grid) != (grid, anchors, lanes):
            # HWC layout of Hailo: [anchors, lanes, grid] -> [grid, anchors, lanes]
            array = np.reshape(array, (anchors, lanes, grid)).transpose(2, 0, 1)
        return np.reshape(array, (1, grid, anchors, lanes))

    def __call__(self, outputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Return the outputs as "loc_row", "exist_row", "loc_col", "exist_col".

        Args:
            outputs: The outputs of `HailoSession.infer`.

        Returns:
            The outputs with batch dimension [1, grid/2, anchors, lanes].
        """
        if self._mapping is None:
            self._mapping = self._find_mapping(outputs)
        return {
            key: self._reshape(key, outputs[self._mapping[key]]) for key in OUTPUT_KEYS
        }
