#!/usr/bin/env python3
import os
from typing import List

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from camera_preprocessing.transformation.calibration import Calibration
from timing.timer import Timer

from lane_detection_ai.model.utils.common import get_config, get_model


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
        self.image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        self.net = self.load_model()

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

        net = get_model(self.config)

        state_dict = torch.load(
            self.config.test_model, map_location="cpu", weights_only=True
        )["model"]
        compatible_state_dict = {}
        for k, v in state_dict.items():
            if "lane_detection_ai.module." in k:
                compatible_state_dict[k[7:]] = v
            else:
                compatible_state_dict[k] = v

        net.load_state_dict(compatible_state_dict, strict=False)
        net.eval()

        return net

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
            image = self.image_transform(image)
            image = image[None, :, -self.config.train_height :, :]

        with Timer(name="inference", filter_strength=40):
            with torch.inference_mode():
                pred = self.net(image)

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
