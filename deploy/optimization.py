import argparse
import copy  # For deep copying model objects
import os
import sys
import time

import torch
import torch.quantization  # For quantization utilities

# Assuming utils are in the parent directory as per your original script
# Ensure this path is correct relative to where you run the script
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

# These are your utility functions, make sure they are available
from utils.common import get_model
from utils.config import Config


def get_args():
    parser = argparse.ArgumentParser(
        description="PyTorch CPU Optimization Script for ResNet-18"
    )
    parser.add_argument(
        "--config_path",
        default="configs/your_resnet18_config.py",  # IMPORTANT: Update this path
        help="Path to config file defining the ResNet-18 model structure.",
        type=str,
    )
    parser.add_argument(
        "--model_path",
        default="weights/your_resnet18_weights.pth",  # IMPORTANT: Update this path
        help="Path to trained ResNet-18 model weights file (.pth).",
        type=str,
    )
    parser.add_argument(
        "--output_model_path",
        default="weights/resnet18_quantized_jit_cpu.pt",
        help="Path to save the optimized (quantized and JIT-traced) model file.",
        type=str,
    )
    parser.add_argument(
        "--size",
        default=(224, 224),  # Common input size for ResNet-18 (width, height)
        help="Size of the input image (width, height). Example: 224 224",
        type=int,
        nargs=2,
    )
    parser.add_argument(
        "--calibration_batches",
        default=10,
        help="Number of dummy data batches to use for calibration.",
        type=int,
    )
    return parser.parse_args()


def print_size_of_model(model, label=""):
    """Prints the size of the model."""
    if isinstance(model, torch.jit.ScriptModule):
        # For JIT models, save the model to get its size
        temp_jit_path = "temp_jit_model_for_size_check.pt"
        torch.jit.save(model, temp_jit_path)
        size = os.path.getsize(temp_jit_path)
        os.remove(temp_jit_path)
    else:
        # For regular nn.Module, save state_dict
        temp_state_dict_path = "temp_state_dict_for_size_check.pth"
        torch.save(model.state_dict(), temp_state_dict_path)
        size = os.path.getsize(temp_state_dict_path)
        os.remove(temp_state_dict_path)
    print(f"Model Size ({label}): {size / 1e6:.2f} MB")


def benchmark_model(model, dummy_input, num_runs=100):
    """Benchmarks the inference speed of the model."""
    model.eval()
    # Ensure dummy_input is on CPU if model is not JIT (JIT model handles device)
    if not isinstance(model, torch.jit.ScriptModule):
        dummy_input = dummy_input.cpu()

    with torch.inference_mode():  # Preferred over torch.no_grad() for inference
        # Warm-up runs
        for _ in range(10):
            _ = model(dummy_input)

        start_time = time.time()
        for _ in range(num_runs):
            _ = model(dummy_input)
        end_time = time.time()

    avg_time_ms = (end_time - start_time) * 1000 / num_runs
    print(f"Average inference time over {num_runs} runs: {avg_time_ms:.3f} ms")
    return avg_time_ms


def fuse_resnet_modules(model_to_fuse):  # model_to_fuse is your main LaneATT model
    """
    Fuses Convolution, BatchNorm, and ReLU modules in a ResNet model
    that is an attribute of model_to_fuse.
    """
    assert isinstance(
        model_to_fuse, torch.nn.Module
    ), "Input model must be an nn.Module"
    fused_model = copy.deepcopy(model_to_fuse)
    fused_model.eval()

    # IMPORTANT: Specify the name of the attribute that holds the ResNet backbone
    # Based on your model's forward pass (self.model(x)), it's likely 'model'.
    # If your ResNet instance is stored under a different attribute name in LaneATT, change this.
    resnet_attr_name = "model"  # <--- VERIFY AND CHANGE IF NECESSARY

    module_paths_to_fuse = []

    # Check if the resnet_attr_name actually exists and is a module
    if not hasattr(fused_model, resnet_attr_name) or not isinstance(
        getattr(fused_model, resnet_attr_name), torch.nn.Module
    ):
        print(
            f"Error: Attribute '{resnet_attr_name}' not found or not an nn.Module in the provided model."
        )
        print("Please verify the attribute name that holds the ResNet backbone.")
        # You might want to print available attributes to help debug:
        # print(f"Available attributes: {[attr for attr in dir(fused_model) if not attr.startswith('_')]}")
        return fused_model  # Return unfused if attribute not found

    # Reference to the ResNet backbone module itself for easier hasattr checks
    resnet_backbone = getattr(fused_model, resnet_attr_name)

    # Initial Conv-BN-ReLU
    # Path will be like 'model.conv1', 'model.bn1', 'model.relu'
    if hasattr(resnet_backbone, "conv1") and hasattr(resnet_backbone, "bn1"):
        base_path = f"{resnet_attr_name}.conv1"
        bn_path = f"{resnet_attr_name}.bn1"
        if hasattr(resnet_backbone, "relu"):  # Assuming a ReLU exists after bn1
            relu_path = f"{resnet_attr_name}.relu"
            module_paths_to_fuse.append([base_path, bn_path, relu_path])
        else:  # If only Conv-BN
            module_paths_to_fuse.append([base_path, bn_path])

    # ResNet layers (layer1, layer2, layer3, layer4 for ResNet-18/34)
    for layer_name_short in ["layer1", "layer2", "layer3", "layer4"]:
        # Full path to the layer module, e.g., 'model.layer1'
        layer_module_full_path = f"{resnet_attr_name}.{layer_name_short}"
        if hasattr(resnet_backbone, layer_name_short):
            layer_module_on_backbone = getattr(resnet_backbone, layer_name_short)
            for block_idx, block in enumerate(layer_module_on_backbone):
                # Path prefix for elements within a block, e.g., 'model.layer1.0'
                block_path_prefix_full = f"{layer_module_full_path}.{block_idx}"

                # Conv1-BN1-ReLU in BasicBlock
                if (
                    hasattr(block, "conv1")
                    and hasattr(block, "bn1")
                    and hasattr(block, "relu")
                ):
                    module_paths_to_fuse.append(
                        [
                            f"{block_path_prefix_full}.conv1",
                            f"{block_path_prefix_full}.bn1",
                            f"{block_path_prefix_full}.relu",
                        ]
                    )
                # Conv2-BN2 in BasicBlock (ReLU typically follows the skip connection addition)
                if hasattr(block, "conv2") and hasattr(block, "bn2"):
                    module_paths_to_fuse.append(
                        [
                            f"{block_path_prefix_full}.conv2",
                            f"{block_path_prefix_full}.bn2",
                        ]
                    )
                # Downsample layer (Conv-BN), if present
                if hasattr(block, "downsample") and block.downsample is not None:
                    if (
                        len(block.downsample) >= 2
                        and isinstance(block.downsample[0], torch.nn.Conv2d)
                        and isinstance(block.downsample[1], torch.nn.BatchNorm2d)
                    ):
                        module_paths_to_fuse.append(
                            [
                                f"{block_path_prefix_full}.downsample.0",
                                f"{block_path_prefix_full}.downsample.1",
                            ]
                        )

    if module_paths_to_fuse:
        print(
            f"Identified {len(module_paths_to_fuse)} module paths for fusion (within '{resnet_attr_name}'):"
        )
        # for p in module_paths_to_fuse: print(f"  - {p}") # Uncomment to print all paths
        torch.quantization.fuse_modules(fused_model, module_paths_to_fuse, inplace=True)
        print("Module fusion applied successfully.")
    else:
        print(
            f"Warning: No specific ResNet module paths identified for fusion within '{resnet_attr_name}'."
        )
        print(
            "Ensure your model structure and the resnet_attr_name are correct for optimal fusion."
        )

    # Fusion for FC layers if they are direct attributes of `fused_model` or `resnet_backbone`
    # Example if FC is part of the resnet_backbone:
    # fc_parent = resnet_backbone
    # fc_prefix = resnet_attr_name
    # if hasattr(fc_parent, 'fc') and isinstance(getattr(fc_parent, 'fc'), torch.nn.Linear):
    #     # Check for an optional ReLU after FC
    #     if hasattr(fc_parent, 'fc_relu') and isinstance(getattr(fc_parent, 'fc_relu'), torch.nn.ReLU):
    #          torch.quantization.fuse_modules(fused_model, [[f'{fc_prefix}.fc', f'{fc_prefix}.fc_relu']], inplace=True)
    #          print(f"Fused {fc_prefix}.fc and {fc_prefix}.fc_relu")
    # (Your FC layer might be an attribute of the main LaneATT model, not the ResNet backbone)

    return fused_model


def optimize_for_cpu(
    original_fp32_model, dummy_input, num_calibration_batches, optimized_model_save_path
):
    """Applies static quantization and TorchScript tracing to a PyTorch model for CPU optimization."""
    print("Starting CPU optimization process...")

    # Ensure the original model is on CPU and in evaluation mode
    model_cpu_fp32 = original_fp32_model.cpu().eval()

    print("\n--- Original Model (FP32 CPU) ---")
    print_size_of_model(model_cpu_fp32, "Original FP32")
    benchmark_model(model_cpu_fp32, dummy_input.cpu())

    # Work on a deep copy for quantization to leave the original model untouched
    model_to_quantize = copy.deepcopy(model_cpu_fp32)
    model_to_quantize.eval()  # Ensure evaluation mode

    # 1. Fuse Modules: Combine Conv-BN-ReLU, etc.
    print("\n--- Step 1: Fusing Modules ---")
    # model_fused = fuse_resnet_modules(
    #     model_to_quantize
    # )  # Returns a new, fused model copy
    model_fused = model_to_quantize  # Skip fusion for simplicity

    # 2. Prepare for Static Quantization: Insert observers
    print("\n--- Step 2: Preparing for Static Quantization ---")
    # Set backend for quantized operations (fbgemm for x86, qnnpack for ARM)
    backend = "fbgemm"  # Default for x86 servers/desktops
    # You can add logic here to choose 'qnnpack' if targeting ARM CPUs
    # e.g., if platform.machine().startswith('aarch64') or platform.system() == 'Android'
    torch.backends.quantized.engine = backend
    print(f"Using quantized engine: {torch.backends.quantized.engine}")

    model_fused.qconfig = torch.quantization.get_default_qconfig(backend)
    print(f"Applied QConfig: {model_fused.qconfig}")

    # inplace=False returns a new model, inplace=True modifies model_fused
    model_prepared = torch.quantization.prepare(model_fused, inplace=False)

    # 3. Calibration: Gather activation statistics
    print(f"\n--- Step 3: Calibrating with {num_calibration_batches} batches ---")
    model_prepared.eval()  # Ensure model is in eval mode for calibration
    with torch.no_grad():  # Disable gradients during calibration
        for i in range(num_calibration_batches):
            # In a real scenario, use a representative calibration dataset here
            _ = model_prepared(dummy_input.cpu())
            if (i + 1) % 10 == 0 or i == num_calibration_batches - 1:  # Print progress
                print(f"  Calibration batch {i+1}/{num_calibration_batches} completed.")
    print("Calibration complete.")

    # 4. Convert to Quantized Model: Use statistics to convert weights and activations
    print("\n--- Step 4: Converting to Quantized Model (INT8) ---")
    # inplace=False returns a new model
    model_quantized_int8 = torch.quantization.convert(model_prepared, inplace=False)
    model_quantized_int8.eval()  # Ensure final quantized model is in eval mode
    print("Static quantization to INT8 complete.")

    print("\n--- Quantized Model (INT8 Python) ---")
    print_size_of_model(model_quantized_int8, "Quantized INT8 (Python)")
    benchmark_model(model_quantized_int8, dummy_input.cpu())

    # 5. TorchScript Tracing: Optimize further and serialize
    print("\n--- Step 5: Applying TorchScript Tracing ---")
    final_optimized_model = None
    try:
        traced_quantized_model = torch.jit.trace(
            model_quantized_int8, dummy_input.cpu()
        )
        print("TorchScript tracing successful.")

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(optimized_model_save_path), exist_ok=True)
        torch.jit.save(traced_quantized_model, optimized_model_save_path)
        print(f"Optimized TorchScript model saved to: {optimized_model_save_path}")

        print("\n--- Quantized + TorchScript Model (INT8 JIT) ---")
        # Load the saved JIT model for verification and benchmarking
        loaded_jit_model = torch.jit.load(optimized_model_save_path)
        loaded_jit_model.eval()
        print_size_of_model(loaded_jit_model, "Quantized JIT (INT8)")
        benchmark_model(loaded_jit_model, dummy_input.cpu())
        final_optimized_model = loaded_jit_model

    except Exception as e:
        print(f"Error during TorchScript tracing or saving: {e}")
        print("Saving the Python quantized model's state_dict as a fallback.")
        fallback_path = optimized_model_save_path.replace(
            ".pt", "_quantized_statedict.pth"
        )
        os.makedirs(os.path.dirname(fallback_path), exist_ok=True)
        torch.save(model_quantized_int8.state_dict(), fallback_path)
        print(
            f"Quantized model state_dict (Python nn.Module) saved to: {fallback_path}"
        )
        print(
            "To load this fallback: Recreate the model structure, apply fusion and quantization steps (prepare, convert without calibration), then load state_dict."
        )
        final_optimized_model = model_quantized_int8

    return final_optimized_model


if __name__ == "__main__":
    args = get_args()

    # Ensure paths from args are valid
    if not os.path.exists(args.config_path):
        print(f"Error: Config path not found: {args.config_path}")
        sys.exit(1)
    if not os.path.exists(args.model_path):
        print(f"Error: Model weights path not found: {args.model_path}")
        sys.exit(1)

    cfg = Config.fromfile(args.config_path)
    # It's common to use batch_size=1 for inference optimization
    cfg.batch_size = 1

    print("Loading original FP32 model...")
    # `get_model(cfg)` should return your ResNet-18 model structure
    # Ensure it's compatible with the weights you're loading.
    net_fp32 = get_model(cfg)

    print(f"Loading weights from: {args.model_path}")
    state_dict_data = torch.load(args.model_path, map_location="cpu")

    # Handle potential nested state_dict (e.g., if saved as part of a checkpoint)
    if "model" in state_dict_data:
        state_dict = state_dict_data["model"]
    elif "state_dict" in state_dict_data:
        state_dict = state_dict_data["state_dict"]
    else:
        state_dict = state_dict_data

    # Clean "module." prefix if weights were saved from DataParallel or DDP
    compatible_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            compatible_state_dict[key[len("module.") :]] = value
        else:
            compatible_state_dict[key] = value

    try:
        net_fp32.load_state_dict(compatible_state_dict, strict=True)
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}")
        print("Ensure the model definition from config matches the weights.")
        print("If using strict=False, be aware of potential missing/unexpected keys.")
        sys.exit(1)

    net_fp32.cpu()
    net_fp32.eval()
    print("Original FP32 model loaded successfully.")

    # Prepare dummy input: (N, C, H, W)
    # args.size is expected as (width, height) from argparse
    input_height, input_width = args.size[1], args.size[0]
    dummy_input_tensor = torch.randn(1, 3, input_height, input_width).cpu()
    print(f"Using dummy input of size: {dummy_input_tensor.shape} (N, C, H, W)")

    # Perform the optimization
    optimized_model = optimize_for_cpu(
        net_fp32, dummy_input_tensor, args.calibration_batches, args.output_model_path
    )

    print("\nOptimization process finished.")
    if isinstance(optimized_model, torch.jit.ScriptModule):
        print(
            f"The primary optimized model (JIT format) is saved at: {args.output_model_path}"
        )
    else:
        print(
            "JIT tracing failed. A Python quantized model's state_dict was saved instead."
        )

    # You can now use 'optimized_model' for inference, or load the saved .pt file:
    # Example:
    # loaded_model = torch.jit.load(args.output_model_path)
    # loaded_model.eval()
    # with torch.inference_mode():
    #     output = loaded_model(dummy_input_tensor)
    # print("Output from loaded optimized model (first element):", output[0] if isinstance(output, (list, tuple)) else output)
