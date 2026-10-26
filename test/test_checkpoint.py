import pytest
import torch

from lane_detection_ai.model.checkpoint import (
    adapt_state_dict,
    extract_state_dict,
    load_checkpoint,
)


class Net(torch.nn.Module):
    def __init__(self, hidden=4):
        super().__init__()
        self.backbone = torch.nn.Linear(3, hidden)
        self.head = torch.nn.Linear(hidden, 2)


class Unpicklable:
    """Stands for the config object pickled into training checkpoints."""


def test_extract_state_dict_variants():
    weights = {"a": torch.zeros(1), "step": 3}
    assert list(extract_state_dict(weights)) == ["a"]
    assert list(extract_state_dict({"model": weights, "epoch": 1})) == ["a"]
    assert list(extract_state_dict({"state_dict": weights})) == ["a"]
    with pytest.raises(TypeError):
        extract_state_dict([1, 2])


def test_adapt_state_dict_strips_prefixes():
    target = set(Net().state_dict())
    weights = {f"module.model.{k}": v for k, v in Net().state_dict().items()}
    assert set(adapt_state_dict(weights, target)) == target

    lightning = {f"model.{k}": v for k, v in Net().state_dict().items()}
    assert set(adapt_state_dict(lightning, target)) == target


def test_load_checkpoint(tmp_path):
    source = Net()
    path = tmp_path / "model.pth"
    torch.save(
        {"model": {f"module.{k}": v for k, v in source.state_dict().items()}}, path
    )

    target = Net()
    load_checkpoint(target, str(path), log=lambda _: None)

    for key, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], value)


def test_load_checkpoint_with_pickled_objects(tmp_path):
    source = Net()
    path = tmp_path / "model.ckpt"
    torch.save({"model": source.state_dict(), "cfg": Unpicklable()}, path)

    logs = []
    load_checkpoint(Net(), str(path), log=logs.append)
    assert "weights_only=False" in logs[0]


def test_load_checkpoint_mismatch(tmp_path):
    path = tmp_path / "model.pth"
    torch.save(Net(hidden=8).state_dict(), path)
    with pytest.raises(RuntimeError, match="different layer sizes"):
        load_checkpoint(Net(hidden=4), str(path))

    torch.save({"other.weight": torch.zeros(1)}, path)
    with pytest.raises(RuntimeError, match="does not fit the model"):
        load_checkpoint(Net(), str(path))
