import pytest

from lane_detection_ai.model.utils.config import Config


def test_config_inheritance(tmp_path):
    (tmp_path / "base.py").write_text(
        "backbone = '18'\noptimizer = dict(lr=0.1, momentum=0.9)\nnum_lanes = 4\n"
    )
    (tmp_path / "model.py").write_text(
        "_base_ = 'base.py'\nnum_lanes = 3\noptimizer = dict(lr=0.01)\n"
    )

    cfg = Config.fromfile(str(tmp_path / "model.py"))

    assert cfg.backbone == "18"
    assert cfg.num_lanes == 3
    assert cfg.optimizer.lr == 0.01
    assert cfg.optimizer.momentum == 0.9
    assert getattr(cfg, "missing", "default") == "default"
    with pytest.raises(KeyError):
        cfg["missing"]

    cfg.test_model = "models/model.pth"
    cfg.extra = {"a": 1}
    assert cfg["test_model"] == "models/model.pth"
    assert cfg.extra.a == 1


def test_config_only_python(tmp_path):
    (tmp_path / "model.yaml").write_text("a: 1\n")
    with pytest.raises(OSError, match="Only python"):
        Config.fromfile(str(tmp_path / "model.yaml"))
