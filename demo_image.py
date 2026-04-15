import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import tqdm

from data.dataset import LaneTestDataset
from utils.common import get_model, merge_config
from utils.dist_utils import dist_print


def pred2coords(
    pred,
    row_anchor,
    col_anchor,
    local_width=1,
    original_image_width=1640,
    original_image_height=590,
    num_lanes=None,
    decode_mode="row_col",
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

    max_row_lanes = num_lane_row if num_lanes is None else min(num_lane_row, int(num_lanes))
    max_col_lanes = num_lane_col if num_lanes is None else min(num_lane_col, int(num_lanes))
    row_lane_idx = list(range(max_row_lanes))
    col_lane_idx = list(range(max_col_lanes))

    decode_row = decode_mode in ("row", "row_col")
    decode_col = decode_mode in ("col", "row_col")

    if decode_row:
        for i in row_lane_idx:
            tmp = []
            # print(valid_row[0, :, i])
            for k in range(valid_row.shape[1]):
                if valid_row[0, k, i]:
                    all_ind = torch.tensor(
                        list(
                            range(
                                max(0, max_indices_row[0, k, i] - local_width),
                                min(
                                    num_grid_row - 1, max_indices_row[0, k, i] + local_width
                                )
                                + 1,
                            )
                        )
                    )

                    out_tmp = (
                        pred["loc_row"][0, all_ind, k, i].softmax(0) * all_ind.float()
                    ).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_row - 1) * original_image_width
                    tmp.append((int(out_tmp), int(row_anchor[k] * original_image_height)))
            if len(tmp) > 0:
                coords.append(tmp)

    if decode_col:
        for i in col_lane_idx:
            tmp = []
            for k in range(valid_col.shape[1]):
                if valid_col[0, k, i]:
                    all_ind = torch.tensor(
                        list(
                            range(
                                max(0, max_indices_col[0, k, i] - local_width),
                                min(
                                    num_grid_col - 1, max_indices_col[0, k, i] + local_width
                                )
                                + 1,
                            )
                        )
                    )

                    out_tmp = (
                        pred["loc_col"][0, all_ind, k, i].softmax(0) * all_ind.float()
                    ).sum() + 0.5

                    out_tmp = out_tmp / (num_grid_col - 1) * original_image_height
                    tmp.append((int(col_anchor[k] * original_image_width), int(out_tmp)))
            if len(tmp) > 0:
                coords.append(tmp)

    return coords


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()
    cfg.batch_size = 1
    print("setting batch_size to 1 for demo generation")

    dist_print("start testing...")
    assert cfg.backbone in [
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
        "mobilenet-v3-small",
        "yolov5n",
        "yolov5s",
    ]

    if cfg.dataset == "CULane":
        cls_num_per_lane = 18
    elif cfg.dataset == "Tusimple":
        cls_num_per_lane = 56
    elif cfg.dataset == "Smartrollerz":
        cls_num_per_lane = 64
    else:
        raise NotImplementedError

    net = get_model(cfg)

    state_dict = torch.load(cfg.test_model, map_location="cpu")["model"]
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if "module." in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    img_transforms = transforms.Compose(
        [
            transforms.Resize(
                (int(cfg.train_height / cfg.crop_ratio), cfg.train_width)
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    if cfg.dataset == "CULane":
        splits = [
            "test0_normal.txt",
            "test1_crowd.txt",
            "test2_hlight.txt",
            "test3_shadow.txt",
            "test4_noline.txt",
            "test5_arrow.txt",
            "test6_curve.txt",
            "test7_cross.txt",
            "test8_night.txt",
        ]
        datasets = [
            LaneTestDataset(
                cfg.data_root,
                os.path.join(cfg.data_root, "list/test_split/" + split),
                img_transform=img_transforms,
                crop_size=cfg.train_height,
            )
            for split in splits
        ]
        img_w, img_h = 1640, 590
    elif cfg.dataset == "Tusimple":
        splits = ["test.txt"]
        datasets = [
            LaneTestDataset(
                cfg.data_root,
                os.path.join(cfg.data_root, split),
                img_transform=img_transforms,
                crop_size=cfg.train_height,
            )
            for split in splits
        ]
        img_w, img_h = 1280, 720
    elif cfg.dataset == "Smartrollerz":
        # splits = ['eval_2025-03-06.txt', "eval_2024-07-04.txt"]
        splits = ["eval_2025-03-06.txt"]
        # splits = ["eval_vid.txt", "eval_vid2.txt"]
        # splits = ['eval.txt']
        datasets = [
            LaneTestDataset(
                cfg.data_root,
                os.path.join(cfg.data_root, "labels", split),
                img_transform=img_transforms,
                crop_size=cfg.train_height,
            )
            for split in splits
        ]
        img_w, img_h = 2064, 1544
        # img_w, img_h = 1364, 1364
        # img_w, img_h = 512, 384
    else:
        raise NotImplementedError
    for split, dataset in zip(splits, datasets):
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1
        )
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        print(split[:-3] + "avi")
        vout = cv2.VideoWriter(split[:-3] + "avi", fourcc, 30.0, (img_w, img_h))
        for i, data in enumerate(tqdm.tqdm(loader)):
            imgs, names = data
            imgs = imgs.cuda()
            with torch.no_grad():
                pred = net(imgs)

            vis = cv2.imread(os.path.join(cfg.data_root, names[0]))
            vis = cv2.resize(vis, (img_w, img_h))
            coords = pred2coords(
                pred,
                cfg.row_anchor,
                cfg.col_anchor,
                original_image_width=img_w,
                original_image_height=img_h,
                num_lanes=cfg.num_lanes,
                decode_mode="row" if cfg.dataset == "Smartrollerz" else "row_col",
            )

            colors = [
                (0, 0, 255),
                (0, 255, 0),
                (255, 0, 0),
                (255, 255, 0),
                (255, 0, 255),
                (0, 255, 255),
            ]
            for lane_idx, lane in enumerate(coords):
                color = colors[lane_idx % len(colors)]
                for coord in lane:
                    cv2.circle(vis, coord, 5, color, -1)
            # cv2.imwrite(f'./demo_images/img{i}.jpg', vis)
            vout.write(vis)
        vout.release()
