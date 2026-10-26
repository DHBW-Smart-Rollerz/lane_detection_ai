import types
from enum import Enum

import numpy as np
import pytest

from lane_detection_ai.model.hailo import HailoOutputMapper, HailoSession

SHAPES = {
    "loc_row": (20, 8, 3),
    "exist_row": (2, 8, 3),
    "loc_col": (10, 6, 3),
    "exist_col": (2, 6, 3),
}


def network_outputs(seed=0):
    rng = np.random.default_rng(seed)
    return {k: rng.normal(size=(1, *s)).astype(np.float32) for k, s in SHAPES.items()}


def test_mapper_flat_outputs_by_size():
    expected = network_outputs()
    # Hailo names, different order, flattened (as the FC output of the network)
    outputs = {
        "net/fc4": expected["exist_col"].reshape(1, -1),
        "net/fc1": expected["loc_row"].reshape(1, 1, 1, -1),
        "net/fc3": expected["exist_row"].reshape(-1),
        "net/fc2": expected["loc_col"].reshape(1, -1),
    }
    logs = []
    mapper = HailoOutputMapper(SHAPES, log=logs.append)

    for _ in range(2):  # second call uses the cached mapping
        mapped = mapper(outputs)
        for key, value in expected.items():
            np.testing.assert_array_equal(mapped[key], value)
    assert len(logs) == 1
    assert "assigned in HEF order" not in logs[0]


def test_mapper_hwc_outputs_by_name():
    expected = network_outputs()
    # HWC layout [anchors, lanes, grid] with the names of the ONNX export
    outputs = {k: np.transpose(v[0], (1, 2, 0))[None] for k, v in expected.items()}

    mapped = HailoOutputMapper(SHAPES, log=lambda _: None)(outputs)

    for key, value in expected.items():
        np.testing.assert_array_equal(mapped[key], value)


def test_mapper_configured_names_and_ambiguity():
    shapes = {**SHAPES, "loc_col": SHAPES["loc_row"], "exist_col": SHAPES["exist_row"]}
    rng = np.random.default_rng(1)
    outputs = {
        f"out{i}": rng.normal(size=(1, 480)).astype(np.float32) for i in range(2)
    }
    outputs |= {
        f"out{i}": rng.normal(size=(1, 48)).astype(np.float32) for i in range(2, 4)
    }

    logs = []
    HailoOutputMapper(shapes, log=logs.append)(outputs)
    assert "assigned in HEF order" in logs[0]

    names = {
        "loc_row": "out1",
        "loc_col": "out0",
        "exist_row": "out3",
        "exist_col": "out2",
    }
    mapped = HailoOutputMapper(shapes, output_names=names)(outputs)
    np.testing.assert_array_equal(mapped["loc_row"].ravel(), outputs["out1"].ravel())
    np.testing.assert_array_equal(mapped["exist_col"].ravel(), outputs["out2"].ravel())


def test_mapper_errors():
    outputs = {"a": np.zeros((1, 5))}
    with pytest.raises(ValueError, match="No HEF output matches"):
        HailoOutputMapper(SHAPES, log=lambda _: None)(outputs)
    with pytest.raises(ValueError, match="Invalid hailo_output_names"):
        HailoOutputMapper(SHAPES, output_names={"loc_row": "a"})(outputs)


class FakeFormatType(Enum):
    AUTO = 0
    UINT8 = 1
    FLOAT32 = 2


class FakeHailo:
    """Mimics the parts of hailo_platform used by HailoSession."""

    def __init__(self, busy_attempts=0):
        self.busy_attempts = busy_attempts
        self.events = []
        self.quant_scale = 0.5
        fake = self

        class VStreamInfo:
            def __init__(self, name, shape):
                self.name, self.shape = name, shape

        class HEF:
            def __init__(self, path):
                fake.events.append(("hef", path))

            def get_input_vstream_infos(self):
                return [VStreamInfo("net/input_layer1", (4, 6, 3))]

            def get_output_vstream_infos(self):
                return [VStreamInfo("net/fc1", (1, 1, 10))]

        class VDevice:
            def __init__(self):
                if fake.busy_attempts > 0:
                    fake.busy_attempts -= 1
                    raise RuntimeError("device busy")
                fake.events.append("vdevice")

            def configure(self, hef, params):
                return [NetworkGroup()]

            def release(self):
                fake.events.append("release")

        class Activation:
            def __enter__(self):
                fake.events.append("activate")

            def __exit__(self, *args):
                fake.events.append("deactivate")

        class NetworkGroup:
            def activate(self):
                return Activation()

        def make_params(direction):
            def make(network_group, format_type=None):
                fake.events.append((direction, format_type))
                return {"format": format_type}

            return types.SimpleNamespace(make_from_network_group=make)

        class InferVStreams:
            def __init__(self, network_group, input_params, output_params):
                self.input_format = input_params["format"]
                self.output_format = output_params["format"]

            def __enter__(self):
                fake.events.append("pipeline")
                return self

            def __exit__(self, *args):
                fake.events.append("pipeline_closed")

            def infer(self, data):
                (image,) = data.values()
                expected = np.uint8 if self.input_format.name == "UINT8" else np.float32
                # HailoRT silently casts to the format of the input vstream
                fake.received = image.astype(expected)
                quantized = np.arange(10, dtype=np.uint8).reshape(1, 1, 1, 10)
                if self.output_format.name == "FLOAT32":
                    return {"net/fc1": quantized * np.float32(fake.quant_scale)}
                return {"net/fc1": quantized}

        self.module = types.SimpleNamespace(
            HEF=HEF,
            VDevice=VDevice,
            ConfigureParams=types.SimpleNamespace(
                create_from_hef=lambda hef, interface: {}
            ),
            HailoStreamInterface=types.SimpleNamespace(PCIe="pcie"),
            FormatType=FakeFormatType,
            InputVStreamParams=make_params("input"),
            OutputVStreamParams=make_params("output"),
            InferVStreams=InferVStreams,
        )


@pytest.fixture
def fake_hailo(monkeypatch):
    fake = FakeHailo(busy_attempts=2)
    monkeypatch.setitem(__import__("sys").modules, "hailo_platform", fake.module)
    return fake


@pytest.mark.parametrize("input_format", ["uint8", "float32"])
def test_session_formats_and_lifecycle(fake_hailo, input_format):
    session = HailoSession("model.hef", input_format=input_format, retry_delay_s=0)

    expected_input = getattr(FakeFormatType, input_format.upper())
    assert ("input", expected_input) in fake_hailo.events
    # Outputs are always dequantized on the host
    assert ("output", FakeFormatType.FLOAT32) in fake_hailo.events
    assert session.channels_last
    assert session.input_shape == (4, 6, 3)

    image = np.full((1, 4, 6, 3), 200, dtype=session.input_dtype)
    out = session.infer(image)
    np.testing.assert_array_equal(fake_hailo.received, image)
    assert out["net/fc1"].dtype == np.float32
    np.testing.assert_allclose(out["net/fc1"].ravel(), np.arange(10) * 0.5)

    session.close()
    session.close()  # idempotent
    assert fake_hailo.events[-3:] == ["pipeline_closed", "deactivate", "release"]
    assert fake_hailo.events.count("release") == 1


def test_session_device_busy(monkeypatch):
    fake = FakeHailo(busy_attempts=5)
    monkeypatch.setitem(__import__("sys").modules, "hailo_platform", fake.module)
    with pytest.raises(RuntimeError, match="used by another process"):
        HailoSession("model.hef", device_retries=3, retry_delay_s=0)


def test_session_without_hailort(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "hailo_platform", None)
    with pytest.raises(RuntimeError, match="HailoRT Python bindings"):
        HailoSession("model.hef")
