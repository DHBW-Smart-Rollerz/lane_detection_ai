"""
Batch inference script for lane detection on a folder of images.
Can be used standalone or integrated with MLflow logging.
"""
import torch
import cv2
import numpy as np
from utils.common import get_model
from utils.config import Config, ConfigDict
from lightning.module import LaneDetectionLightningModule
import datetime
import argparse
import os
from pathlib import Path


def _ensure_dataset_anchors(cfg):
    """Populate row/col anchors only when they are missing or invalid."""
    row_anchor = getattr(cfg, "row_anchor", None)
    col_anchor = getattr(cfg, "col_anchor", None)

    row_ok = row_anchor is not None and len(row_anchor) == int(cfg.num_row)
    col_ok = col_anchor is not None and len(col_anchor) == int(cfg.num_col)

    if row_ok and col_ok:
        cfg.row_anchor = np.asarray(row_anchor, dtype=np.float32)
        cfg.col_anchor = np.asarray(col_anchor, dtype=np.float32)
        print("[BATCH_INFERENCE] Using anchors already present in config")
        return

    dataset = str(getattr(cfg, "dataset", "")).lower()

    if dataset == "culane":
        cfg.row_anchor = np.linspace(0.42, 1.0, cfg.num_row, dtype=np.float32)
        cfg.col_anchor = np.linspace(0.0, 1.0, cfg.num_col, dtype=np.float32)
    elif dataset == "tusimple":
        cfg.row_anchor = np.linspace(160, 710, cfg.num_row, dtype=np.float32) / 720.0
        cfg.col_anchor = np.linspace(0.0, 1.0, cfg.num_col, dtype=np.float32)
    elif dataset == "curvelanes":
        cfg.row_anchor = np.linspace(0.4, 1.0, cfg.num_row, dtype=np.float32)
        cfg.col_anchor = np.linspace(0.0, 1.0, cfg.num_col, dtype=np.float32)
    elif dataset == "smartrollerz":
        # Keep project default behavior for Smartrollerz unless config explicitly defines anchors.
        cfg.row_anchor = np.linspace(100, 1540, cfg.num_row, dtype=np.float32) / 1550.0
        cfg.col_anchor = np.linspace(0.0, 1.0, cfg.num_col, dtype=np.float32)
    else:
        raise NotImplementedError(f"Unsupported dataset for anchor setup: {cfg.dataset}")

    print("[BATCH_INFERENCE] Anchors were missing/invalid in config; using dataset defaults")


def _preprocess_image_for_model(image_bgr, cfg):
    """Match demo/training preprocessing: resize -> bottom crop -> RGB tensor -> normalize."""
    resized_h = int(round(float(cfg.train_height) / float(cfg.crop_ratio)))
    if resized_h < int(cfg.train_height):
        raise ValueError(
            f"Invalid crop_ratio={cfg.crop_ratio}: resized_h ({resized_h}) < train_height ({cfg.train_height})"
        )

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (int(cfg.train_width), resized_h), interpolation=cv2.INTER_LINEAR)
    image_cropped = image_resized[-int(cfg.train_height):, :, :]

    image_tensor = torch.from_numpy(image_cropped).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=image_tensor.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=image_tensor.dtype).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std

    return image_tensor.unsqueeze(0)


def _extract_state_dict_from_checkpoint(
    checkpoint_path,
    device="cuda",
    allow_unsafe_load=True,
    weights_only=True,
):
    """
    Extract the state_dict from a PyTorch Lightning checkpoint file (.ckpt).
    """
    try:
        # Temporarily increase recursion limit to handle deeply nested objects
        import sys
        import numpy as np
        import _codecs
        from utils.config import Config, ConfigDict
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(10000)

        # Allowlist safe globals for weights_only=True
        torch.serialization.add_safe_globals([
            Config,
            ConfigDict,
            np.ndarray,
            np.core.multiarray._reconstruct,
            _codecs.encode,
            np.dtype,
            np.dtypes.Float64DType,
        ])

        # Always load checkpoint tensors on CPU to avoid CUDA-side crashes
        map_location = "cpu"

        if weights_only:
            # Load the checkpoint with weights_only=True
            try:
                checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
            except Exception as e:
                if not allow_unsafe_load:
                    raise
                print(f"[WARNING] weights_only=True failed: {e}")
                print("[INFO] Retrying with weights_only=False on CPU...")
                checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except RecursionError as e:
        raise RuntimeError(f"RecursionError: The checkpoint contains deeply nested objects. {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to extract state_dict from checkpoint: {e}")
    finally:
        try:
            sys.setrecursionlimit(old_limit)
        except Exception:
            pass

    # Debug: Print the keys in the checkpoint
    print(f"[DEBUG] Checkpoint keys: {list(checkpoint.keys())}")

    # Extract the state_dict
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        raise ValueError(f"No 'state_dict' or 'model' key found in checkpoint: {checkpoint_path}")

    # Strip Lightning prefix if present
    if all(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}

    return state_dict


def run_batch_inference(config_path, weights_path, image_folder, output_folder=None, device="cuda", decode_mode="auto"):
    """
    Run inference on all images in a folder.

    Args:
        config_path: Path to config file
        weights_path: Path to checkpoint/weights (.ckpt for Lightning, .pth for PyTorch)
        image_folder: Folder containing images
        output_folder: Where to save visualizations (default: batch_inference_results)
        device: "cuda" or "cpu"
        decode_mode: "auto", "row", "col", or "row_col"

    Returns:
        results: List of dicts with inference results
    """
    print(f"\n[BATCH_INFERENCE] Starting batch inference...")
    print(f"[BATCH_INFERENCE] config_path: {config_path}")
    print(f"[BATCH_INFERENCE] weights_path: {weights_path}")
    print(f"[BATCH_INFERENCE] image_folder: {image_folder}")
    print(f"[BATCH_INFERENCE] output_folder: {output_folder}")
    print(f"[BATCH_INFERENCE] device: {device}")

    # Create output folder
    if output_folder is None:
        output_folder = "batch_inference_results"
    os.makedirs(output_folder, exist_ok=True)

    # Extract run name from weights_path or use timestamp
    run_name = os.path.basename(os.path.dirname(weights_path))  # Extract folder name from weights path
    if not run_name or run_name == "":
        run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # Fallback to timestamp

    model_output_folder = os.path.join(output_folder, run_name)
    os.makedirs(model_output_folder, exist_ok=True)
    print(f"[BATCH_INFERENCE] Created run-specific output folder: {model_output_folder}")

    # Update paths to save inference images in `model_output_folder`
    output_folder = model_output_folder

    # Load config
    print(f"[BATCH_INFERENCE] Loading config from {config_path}...")
    cfg = Config.fromfile(config_path)
    print(f"[BATCH_INFERENCE] Config loaded successfully")

    if decode_mode == "auto":
        decode_mode = "row" if cfg.dataset == "Smartrollerz" else "row_col"
    print(f"[BATCH_INFERENCE] Decode mode: {decode_mode}")

    # Set up dataset anchors (do not override if config already provides valid anchors)
    _ensure_dataset_anchors(cfg)
    print(f"[BATCH_INFERENCE] Anchors set up: num_row={cfg.num_row}, num_col={cfg.num_col}")

    # Load trained weights and model
    try:
        print(f"[INFO] Loading checkpoint: {weights_path}")
        if str(weights_path).endswith(".ckpt"):
            print("[INFO] Using Lightning module with checkpoint state_dict")
            module = LaneDetectionLightningModule(cfg)
            state_dict = _extract_state_dict_from_checkpoint(
                weights_path,
                device,
                allow_unsafe_load=True,
                weights_only=False,
            )
            print(f"[INFO] Loading {len(state_dict)} weights into model")
            module.model.load_state_dict(state_dict, strict=False)
            net = module.model
        else:
            net = get_model(cfg)
            state = torch.load(weights_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state_dict = state["state_dict"]
            elif isinstance(state, dict) and "model" in state:
                state_dict = state["model"]
            else:
                state_dict = state
            print(f"[INFO] Loading {len(state_dict)} weights into model")
            net.load_state_dict(state_dict)

        print("✓ Successfully loaded checkpoint")

    except Exception as e:
        print(f"✗ ERROR: Failed to load checkpoint: {e}")
        import traceback
        traceback.print_exc()
        raise

    net = net.to(device)
    net.eval()
    print(f"[INFO] Model moved to {device} and set to eval mode")

    # Debug: Check if the model is ready
    print("[DEBUG] Model is ready for inference.")

    # Get all image files
    print(f"[BATCH_INFERENCE] Searching for images in {image_folder}...")
    image_folder = Path(image_folder)
    print(f"[BATCH_INFERENCE] Resolved image folder path: {image_folder.absolute()}")
    print(f"[BATCH_INFERENCE] Image folder exists: {image_folder.exists()}")

    if image_folder.exists():
        print(f"[BATCH_INFERENCE] Contents of {image_folder}:")
        for item in list(image_folder.iterdir())[:10]:
            print(f"  - {item}")

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = sorted([
        f for f in image_folder.rglob('*') 
        if f.suffix.lower() in image_extensions
    ])

    print(f"[BATCH_INFERENCE] Found {len(image_files)} image files")
    if len(image_files) > 0:
        print(f"[BATCH_INFERENCE] First few images:")
        for img_file in image_files[:5]:
            print(f"  - {img_file}")

    if not image_files:
        print(f"[BATCH_INFERENCE] ✗ No images found in {image_folder}")
        return []

    print(f"[BATCH_INFERENCE] Starting inference on {len(image_files)} images...")

    results = []

    with torch.no_grad():
        for idx, image_path in enumerate(image_files):
            print(f"[BATCH_INFERENCE] [{idx+1}/{len(image_files)}] Processing {image_path.name}...")

            try:
                # Load and preprocess image
                image = cv2.imread(str(image_path))
                if image is None:
                    print(f"[BATCH_INFERENCE]   FAILED - Could not read image")
                    continue

                print(f"[BATCH_INFERENCE]   Image shape: {image.shape}")
                original_h, original_w = image.shape[:2]
                image_tensor = _preprocess_image_for_model(image, cfg).to(device)

                print(f"[BATCH_INFERENCE]   Tensor shape: {image_tensor.shape}")

                # Inference
                output = net(image_tensor)
                print(f"[BATCH_INFERENCE]   Inference complete, output keys: {output.keys() if isinstance(output, dict) else 'not a dict'}")

                # Get lane coordinates
                coords = pred2coords(
                    output,
                    cfg.row_anchor,
                    cfg.col_anchor,
                    original_image_width=original_w,
                    original_image_height=original_h,
                    num_grid_row=cfg.num_cell_row,
                    num_grid_col=cfg.num_cell_col,
                    num_lanes=cfg.num_lanes,
                    decode_mode=decode_mode,
                )

                print(f"[BATCH_INFERENCE]   Detected {len(coords)} lanes")

                # Visualize
                vis = visualize_lanes(image, coords)

                # Save visualization
                output_path = os.path.join(output_folder, f"{image_path.stem}_detected.png")
                cv2.imwrite(output_path, vis)
                print(f"[BATCH_INFERENCE]   Saved to: {output_path}")

                result = {
                    'image_path': str(image_path),
                    'output_path': output_path,
                    'num_lanes': len(coords),
                    'image_size': (original_w, original_h),
                    'success': True
                }
                results.append(result)
                print(f"[BATCH_INFERENCE]   ✓ OK - {len(coords)} lanes detected")

            except Exception as e:
                print(f"[BATCH_INFERENCE]   ✗ FAILED - {str(e)}")
                import traceback
                traceback.print_exc()
                results.append({
                    'image_path': str(image_path),
                    'success': False,
                    'error': str(e)
                })

    # Print summary
    successful = sum(1 for r in results if r['success'])
    print(f"\n{'='*60}")
    print(f"[BATCH_INFERENCE] Summary: {successful}/{len(results)} images processed successfully")
    print(f"[BATCH_INFERENCE] Visualizations saved to: {output_folder}")
    print(f"[BATCH_INFERENCE] Output folder contents: {os.listdir(output_folder)}")
    print(f"{'='*60}\n")

    return results


def pred2coords(
    pred,
    row_anchor,
    col_anchor,
    local_width=1,
    original_image_width=1640,
    original_image_height=590,
    num_grid_row=50,
    num_grid_col=50,
    num_lanes=None,
    decode_mode="row_col",
):
    """Convert model predictions to lane coordinates."""
    batch_size, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
    batch_size, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape

    max_indices_row = pred['loc_row'].argmax(1).cpu()
    valid_row = pred['exist_row'].argmax(1).cpu()
    max_indices_col = pred['loc_col'].argmax(1).cpu()
    valid_col = pred['exist_col'].argmax(1).cpu()

    pred['loc_row'] = pred['loc_row'].cpu()
    pred['loc_col'] = pred['loc_col'].cpu()

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
                                min(num_grid_col - 1, max_indices_col[0, k, i] + local_width) + 1,
                            )
                        )
                    )
                    out_tmp = (pred['loc_col'][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_col - 1) * original_image_height
                    tmp.append((int(col_anchor[k] * original_image_width), int(out_tmp)))
            if len(tmp) > 0:
                coords.append(tmp)

    return coords


def visualize_lanes(image, coords, colors=None):
    """Draw lanes on image and return visualization."""
    if colors is None:
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    """Draw lanes on image and return visualization."""
    vis = image.copy()
    for lane_idx, lane in enumerate(coords):
        if len(lane) > 0:
            color = colors[lane_idx % len(colors)]
            # Draw points
            for coord in lane:
                cv2.circle(vis, coord, 5, color, -1)
            # Draw lines connecting points
            points = np.array(lane, dtype=np.int32)
            points = points[np.argsort(points[:, 1])]  # Sort by y coordinate
            cv2.polylines(vis, [points], False, color, 2)
    
    return vis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch lane detection inference")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/smartrollerz_res18_bev.py",
        help="Path to the config file"
    )
    
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to the trained checkpoint/weights"
    )
    
    parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help="Path to folder containing images"
    )
    
    parser.add_argument(
        "--output_folder",
        type=str,
        default="batch_inference_results",
        help="Where to save visualizations (default: batch_inference_results)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for inference"
    )

    parser.add_argument(
        "--decode_mode",
        type=str,
        default="auto",
        choices=["auto", "row", "col", "row_col"],
        help="Lane decoding mode: auto=row for Smartrollerz, row_col otherwise"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights file not found: {args.weights}")
    if not os.path.exists(args.image_folder):
        raise FileNotFoundError(f"Image folder not found: {args.image_folder}")
    
    run_batch_inference(
        args.config,
        args.weights,
        args.image_folder,
        args.output_folder,
        args.device,
        args.decode_mode,
    )