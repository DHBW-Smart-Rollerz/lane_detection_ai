"""Helpers to load the model config and to create the model."""

import importlib

import numpy as np
import torch

from lane_detection_ai.model.utils.config import Config

# ImageNet normalization of the RGB input the network was trained with
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_config(path: str) -> Config:
    """Load a model config and add the row and column anchors of its dataset.

    Args:
        path: Path to the python model config file.

    Returns:
        The model config.
    """
    cfg = Config.fromfile(path)

    if cfg.dataset == "CULane":
        cfg.row_anchor = np.linspace(0.42, 1, cfg.num_row)
        cfg.col_anchor = np.linspace(0, 1, cfg.num_col)
    elif cfg.dataset == "Tusimple":
        cfg.row_anchor = np.linspace(160, 710, cfg.num_row) / 720
        cfg.col_anchor = np.linspace(0, 1, cfg.num_col)
    elif cfg.dataset == "CurveLanes":
        cfg.row_anchor = np.linspace(0.4, 1, cfg.num_row)
        cfg.col_anchor = np.linspace(0, 1, cfg.num_col)
    elif cfg.dataset == "Smartrollerz":
        cfg.row_anchor = np.linspace(100, 1540, cfg.num_row) / 1550
        cfg.col_anchor = np.linspace(0, 1, cfg.num_col)

    return cfg


def get_model(cfg: Config) -> torch.nn.Module:
    """Create the (untrained) model for the dataset of the config.

    Args:
        cfg: The model config.

    Returns:
        The model on the CPU.
    """
    return importlib.import_module(
        "lane_detection_ai.model.model.model_" + cfg.dataset.lower()
    ).get_model(cfg)


def initialize_weights(*models: torch.nn.Module) -> None:
    """Initialize the weights of the given modules (recursively).

    Args:
        models: The modules to initialize.
    """
    for model in models:
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
            elif isinstance(module, torch.nn.Linear):
                module.weight.data.normal_(0.0, std=0.01)
            elif isinstance(module, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(module.weight, 1)
                torch.nn.init.constant_(module.bias, 0)
