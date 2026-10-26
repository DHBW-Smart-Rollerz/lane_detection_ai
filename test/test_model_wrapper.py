import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from lane_detection_ai.model import model_wrapper
from lane_detection_ai.model.model_wrapper import LaneDetectionAiModel
from lane_detection_ai.model.utils.common import get_config, get_model

REPO = Path(__file__).resolve().parents[1]

MODEL_CONFIG = """
dataset = "Smartrollerz"
backbone = "18"
num_row = 8
num_col = 6
num_cell_row = 20
num_cell_col = 10
num_lanes = 3
use_aux = False
fc_norm = False
mlp_mid_dim = 64
train_width = 128
train_height = 64
crop_ratio = 0.8
"""


def write_config(base: Path, name: str, test_model: str, extra: str = "") -> str:
    (base / name).write_text(MODEL_CONFIG + f'test_model = "{test_model}"\n' + extra)
    return name


@pytest.fixture(scope="module")
def package(tmp_path_factory):
    """Package dir with a random model as .pth and as float32/uint8 ONNX."""
    base = tmp_path_factory.mktemp("package")
    write_config(base, "pth.py", "model.pth")
    cfg = get_config(str(base / "pth.py"))

    torch.manual_seed(0)
    net = get_model(cfg).eval()
    with torch.no_grad():  # larger existence logits, so that lanes are found
        net.cls[-1].weight.mul_(20)
    state = {f"module.{k}": v for k, v in net.state_dict().items()}
    torch.save({"model": state, "epoch": 3}, base / "model.pth")

    # float32 input, outputs in the order of the network dict
    torch.onnx.export(
        net,
        torch.zeros(1, 3, cfg.train_height, cfg.train_width),
        str(base / "model_float.onnx"),
        input_names=["image"],
        output_names=["loc_row", "loc_col", "exist_row", "exist_col"],
        dynamo=False,
    )
    # uint8 input with the normalization in the graph (as for the Hailo compiler)
    subprocess.run(  # noqa: S603 (runs the export script of this repo)
        [
            sys.executable,
            str(REPO / "scripts" / "convert_pth_onnx.py"),
            "--base-path",
            str(base),
            "--model-config",
            "pth.py",
            "--pth",
            "model.pth",
            "--onnx-out",
            "model_uint8.onnx",
        ],
        check=True,
    )
    return base


def load(base: Path, config: str, test_model: str, extra: str = ""):
    write_config(base, config, test_model, extra)
    return LaneDetectionAiModel(str(base), config, log=lambda _: None)


def outputs(model, image):
    return model._infer(model._preprocess(image))


@pytest.fixture
def image():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(150, 200), dtype=np.uint8)


def test_backends_give_same_outputs(package, image):
    pth = load(package, "pth.py", "model.pth")
    onnx_float = load(package, "onnx_float.py", "model_float.onnx")
    onnx_uint8 = load(package, "onnx_uint8.py", "model_uint8.onnx")
    assert (pth.backend, onnx_float.backend, onnx_uint8.backend) == (
        "pytorch",
        "onnxruntime",
        "onnxruntime",
    )
    assert onnx_uint8._uint8_input
    assert not onnx_float._uint8_input

    reference = outputs(pth, image)
    for model in (onnx_float, onnx_uint8):
        actual = outputs(model, image)
        for key, value in reference.items():
            np.testing.assert_allclose(actual[key], value, rtol=1e-3, atol=1e-3)


def test_predict(package, image):
    model = load(package, "pth.py", "model.pth")

    lanes = model.predict(image)

    assert len(lanes) == 3
    found = [lane for lane in lanes if lane is not None]
    assert found, "the test model should find lanes"
    for lane in found:
        assert lane.dtype == np.int32
        assert lane.shape[1] == 2
        assert (lane[:, 0] <= image.shape[1] + 1).all()
        assert (lane[:, 1] <= image.shape[0]).all()

    color = np.repeat(image[..., None], 3, axis=2)
    for actual, expected in zip(model.predict(color), lanes, strict=True):
        np.testing.assert_array_equal(actual, expected)


class FakeHailoSession:
    """Runs an ONNX model like the HEF would run on the Hailo device."""

    onnx_path: Path

    def __init__(self, hef_path, *, input_format):
        self.input_format = input_format
        self.channels_last = True
        self.input_shape = (64, 128, 3)
        self.closed = False
        self._session = ort.InferenceSession(
            str(self.onnx_path), providers=["CPUExecutionProvider"]
        )

    def infer(self, image):
        assert image.dtype == (np.uint8 if self.input_format == "uint8" else np.float32)
        assert image.shape == (1, *self.input_shape)
        nchw = np.ascontiguousarray(image.transpose(0, 3, 1, 2))
        input_name = self._session.get_inputs()[0].name
        names = [o.name for o in self._session.get_outputs()]
        result = dict(
            zip(names, self._session.run(None, {input_name: nchw}), strict=True)
        )
        # Compiled HEFs have generic output names and flat outputs
        order = ["exist_col", "loc_row", "exist_row", "loc_col"]
        return {f"net/fc{i}": result[key].reshape(1, -1) for i, key in enumerate(order)}

    def close(self):
        self.closed = True


@pytest.mark.parametrize(
    ("preprocess", "onnx_file", "reference_file"),
    [
        ("baked", "model_uint8.onnx", "model_uint8.onnx"),
        ("pytorch", "model_float.onnx", "model_float.onnx"),
    ],
)
def test_hailo_backend(
    package, image, monkeypatch, preprocess, onnx_file, reference_file
):
    monkeypatch.setattr(
        FakeHailoSession, "onnx_path", package / onnx_file, raising=False
    )
    monkeypatch.setattr(model_wrapper, "HailoSession", FakeHailoSession)
    (package / "model.hef").write_bytes(b"hef")

    hailo = load(package, "hef.py", "model.hef", f'hailo_preprocess = "{preprocess}"\n')
    reference = load(package, "ref.py", reference_file)

    for actual, expected in zip(
        hailo.predict(image), reference.predict(image), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)

    session = hailo._hailo_session
    hailo.close()
    assert session.closed
    with pytest.raises(RuntimeError, match="closed"):
        hailo.predict(image)


def test_invalid_configs(package):
    (package / "model.tflite").write_bytes(b"")
    with pytest.raises(ValueError, match="Unsupported model file format"):
        load(package, "bad.py", "model.tflite")
    with pytest.raises(ValueError, match="decode_mode"):
        load(package, "bad.py", "model.pth", 'decode_mode = "max"\n')
