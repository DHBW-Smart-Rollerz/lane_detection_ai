import torch
import cv2
import numpy as np
from utils.common import get_model, merge_config
from utils.config import Config
import datetime
import argparse
import os


def pred2coords(pred, row_anchor, col_anchor, local_width=1, original_image_width=1640, original_image_height=590, num_grid_row=50, num_grid_col=50):
    batch_size, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
    batch_size, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape

    max_indices_row = pred['loc_row'].argmax(1).cpu()
    valid_row = pred['exist_row'].argmax(1).cpu()
    max_indices_col = pred['loc_col'].argmax(1).cpu()
    valid_col = pred['exist_col'].argmax(1).cpu()

    pred['loc_row'] = pred['loc_row'].cpu()
    pred['loc_col'] = pred['loc_col'].cpu()

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
                            min(num_grid_row - 1, max_indices_row[0, k, i] + local_width) + 1,
                        )
                    )
                )
                out_tmp = (pred['loc_row'][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                out_tmp = out_tmp / (num_grid_row - 1) * original_image_width
                tmp.append((int(out_tmp), int(row_anchor[k] * original_image_height)))
        coords.append(tmp)

    for i in col_lane_idx:
        tmp = []
        for k in range(valid_col.shape[1]):
            if valid_col[0, k, i]:
                all_ind = torch.tensor(
                    list(
                        range(
                            max(0, max_indices_col[0, k, i] - local_width),
                            min(num_grid_col - 1, max_indices_col[0, k, i] + local_width) + 1,
                        )
                    )
                )
                out_tmp = (pred['loc_col'][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                out_tmp = out_tmp / (num_grid_col - 1) * original_image_height
                tmp.append((int(col_anchor[k] * original_image_width), int(out_tmp)))
        coords.append(tmp)

    return coords


def main(args):
    # Load config and model
    cfg = Config.fromfile(args.config)
    net = get_model(cfg)

    # Set up anchors for Smartrollerz dataset
    cfg.row_anchor = np.linspace(100, 1540, cfg.num_row) / 1550
    cfg.col_anchor = np.linspace(0, 1, cfg.num_col)

    # Load trained weights
    state_dict = torch.load(args.weights, map_location="cuda", weights_only=True)["model"]
    net.load_state_dict(state_dict)
    net = net.to("cuda")  # or use "cuda" if you want GPU inference
    net.eval()

    # Load and preprocess image
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not read image from {args.image}")
    
    original_h, original_w = image.shape[:2]
    image_resized = cv2.resize(image, (cfg.train_width, cfg.train_height))
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to("cuda")  # Match the device

    # Inference
    with torch.no_grad():
        output = net(image_tensor)
        # output contains: loc_row, loc_col, exist_row, exist_col

    # Get lane coordinates
    coords = pred2coords(
        output,
        cfg.row_anchor,
        cfg.col_anchor,
        original_image_width=original_w,
        original_image_height=original_h,
        num_grid_row=cfg.num_cell_row,
        num_grid_col=cfg.num_cell_col
    )

    # Create visualization - draw lanes on original image
    vis = image.copy()
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    for lane_idx, lane in enumerate(coords):
        if len(lane) > 0:
            color = colors[lane_idx % len(colors)]
            # Draw points
            for coord in lane:
                cv2.circle(vis, coord, 5, color, -1)
            # Draw lines connecting points
            points = np.array(lane, dtype=np.int32)
            points = points[np.argsort(points[:, 1])]  # Sort by y coordinate for proper line drawing
            cv2.polylines(vis, [points], False, color, 2)

    # Save the visualization
    os.makedirs("result_inference", exist_ok=True)
    output_path = f"result_inference/visualization-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    cv2.imwrite(output_path, vis)
    print(f"Visualization saved to {output_path}")

    # Display image info
    print(f"Original image size: {original_w}x{original_h}")
    print(f"Number of detected lanes: {len(coords)}")
    for i, lane in enumerate(coords):
        print(f"Lane {i}: {len(lane)} points detected")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lane detection inference script")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/smartrollerz_res18_bev.py",
        help="Path to the config file (default: configs/smartrollerz_res18_bev.py)"
    )
    
    parser.add_argument(
        "--weights",
        type=str,
        default="results/20251028_112344_lr_6e-03_b_32/best_model.pth",
        help="Path to the trained weights file (default: results/20251028_112344_lr_6e-03_b_32/best_model.pth)"
    )
    
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image file (required)"
    )
    
    args = parser.parse_args()
    
    # Validate that required arguments exist
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights file not found: {args.weights}")
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image file not found: {args.image}")
    
    main(args)