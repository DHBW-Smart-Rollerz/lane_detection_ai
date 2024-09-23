#!/usr/bin/env python3
import os
from typing import List

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from timing.timer import Timer

from lane_detection_ai.model.utils.common import get_config, get_model


class LaneDetectionAiModel:
    """LaneDetectionAiModel class"""

    def __init__(self, base_path: str, model_config_path: str):
        torch.backends.cudnn.benchmark = True

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
        self.config.batch_size = 1

        assert self.config.backbone in [
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
            with torch.no_grad():
                pred = self.net(image)

        with Timer(name="pred2coords", filter_strength=40):
            coords = self.pred2coords(
                pred,
                self.config.row_anchor,
                self.config.col_anchor,
                original_image_width=2064,
                original_image_height=1544,
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
