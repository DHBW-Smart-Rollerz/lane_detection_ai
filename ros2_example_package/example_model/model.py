import os

import numpy as np
import yaml


class ExampleModel:
    """Example model class."""

    def __init__(self, root: str, model_config_path: str):
        """
        Initializes the ExampleModel.

        Arguments:
            root -- Root path of the package
            model_config_path -- Path to the model configuration
        """
        cfg = self._load_config(os.path.join(root, model_config_path))
        self._load_model(os.path.join(root, cfg["model_path"]))

    def predict(self, image: np.ndarray) -> int:
        """
        Calculates the average of an image and returns the result with a bias.

        Args:
            image (np.ndarray): input image

        Returns:
            int: average of image with bias
        """
        # Just some example calculation with the image
        return np.mean(image) + 42

    def _load_config(self, path: str) -> dict:
        """
        Loads the model configuration.

        Args:
            path (str): path to the configuration

        Returns:
            dict: configuration
        """
        with open(path, "r") as file:
            return yaml.safe_load(file)

    def _load_model(self, path: str):
        """
        Loads the model.

        Args:
            path (str): path to the model
        """
        pass
