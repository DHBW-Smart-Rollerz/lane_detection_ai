# import argparse
# import os

# # Assuming your project structure allows these imports
# # If running from deploy folder, adjust paths or ensure ufldv2 is in PYTHONPATH
# import sys

# import onnx
# import onnxruntime
# import torch
# from onnxruntime.quantization import QuantType, quantize_dynamic

# # Add the root directory of your project to sys.path
# # This is a common way to handle imports if the script is in a subdirectory
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from utils.common import get_model, merge_config  # merge_config loads the .py config
# from utils.config import Config  # For loading .py style configs


# class ModelWrapperForONNX(torch.nn.Module):
#     """
#     Wrapper for the parsingNet model to ensure its output is a tuple of tensors,
#     which is required by torch.onnx.export. The order of tensors in the tuple
#     must match the order of output_names provided to the exporter.
#     """

#     def __init__(self, model, output_names):
#         super().__init__()
#         self.model = model
#         # Store the expected output names in the desired order
#         self.output_names = output_names

#     def forward(self, x):
#         # Get the dictionary of outputs from the original model
#         pred_dict = self.model(x)
#         # Convert the dictionary to a tuple, ensuring the order matches self.output_names
#         # This is crucial for the ONNX exporter to correctly name the output nodes.
#         return tuple(pred_dict[name] for name in self.output_names)


# def main(args):
#     # --- 1. Load Configuration ---
#     print(f"Loading configuration from: {args.config}")
#     # Use the project's merge_config logic if it handles .py files directly
#     # For simplicity, if cfg is a .py file, we can use Config.fromfile
#     if args.config.endswith(".py"):
#         cfg = Config.fromfile(args.config)
#     else:
#         # Fallback or error if it's not a .py config as expected by UFLDv2
#         raise ValueError(
#             "Configuration file must be a .py file for this project structure."
#         )

#     # Override any config values if needed (e.g., for export)
#     # cfg.batch_size = 1 # Usually set to 1 for export

#     print("Configuration loaded successfully.")
#     # print(f"Effective Config: {cfg}")

#     # --- 2. Load PyTorch Model ---
#     print(f"Loading PyTorch model. Backbone: {cfg.backbone}")
#     pytorch_model = get_model(cfg)

#     if not os.path.exists(args.pth_model):
#         print(f"Error: PyTorch model file not found at {args.pth_model}")
#         return

#     print(f"Loading state dict from: {args.pth_model}")
#     try:
#         # Load the state dict, ensuring it's mapped to CPU if saved on GPU
#         state_dict = torch.load(args.pth_model, map_location="cpu")["model"]

#         # Handle potential 'module.' prefix if the model was saved from DataParallel/DistributedDataParallel
#         if any(key.startswith("module.") for key in state_dict.keys()):
#             state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

#         pytorch_model.load_state_dict(state_dict)
#         pytorch_model.eval()  # Set model to evaluation mode
#         print("PyTorch model loaded and set to eval mode.")
#     except Exception as e:
#         print(f"Error loading PyTorch model state_dict: {e}")
#         return

#     # --- 3. Prepare Dummy Input ---
#     # Batch size for export is typically 1
#     batch_size = 1
#     # Input channels (e.g., 3 for RGB)
#     input_channels = 3
#     # Input dimensions from config
#     dummy_input = torch.randn(
#         batch_size,
#         input_channels,
#         cfg.train_height,
#         cfg.train_width,
#         requires_grad=False,
#     )
#     print(f"Created dummy input with shape: {dummy_input.shape}")

#     # --- 4. Define Input and Output Names ---
#     input_names = ["input_image"]
#     # The output names must match the keys of the dictionary returned by parsingNet's forward method
#     # and also the order defined in ModelWrapperForONNX
#     # Common output names for this model structure:
#     output_names = ["loc_row", "loc_col", "exist_row", "exist_col"]

#     # --- 5. Wrap Model for ONNX Export ---
#     # This wrapper ensures the output is a tuple in the correct order
#     wrapped_model = ModelWrapperForONNX(pytorch_model, output_names)
#     print("Model wrapped for ONNX export.")

#     # --- 6. Export to ONNX (FP32) ---
#     if not os.path.exists(args.output_dir):
#         os.makedirs(args.output_dir)
#         print(f"Created output directory: {args.output_dir}")

#     onnx_fp32_path = os.path.join(args.output_dir, "model_fp32.onnx")
#     print(f"Exporting model to ONNX (FP32): {onnx_fp32_path}")

#     try:
#         torch.onnx.export(
#             wrapped_model,
#             dummy_input,
#             onnx_fp32_path,
#             input_names=input_names,
#             output_names=output_names,
#             opset_version=args.opset,
#             do_constant_folding=True,  # Optimization
#             export_params=True,  # Store the trained parameters in the model file
#             dynamic_axes={  # Optional: for variable batch size or input dimensions
#                 input_names[0]: {0: "batch_size", 2: "height", 3: "width"},
#                 **{
#                     name: {0: "batch_size"} for name in output_names
#                 },  # Assuming batch is dynamic for outputs
#             },
#         )
#         print("ONNX FP32 model exported successfully.")

#         # Verify the ONNX model
#         onnx_model_fp32 = onnx.load(onnx_fp32_path)
#         onnx.checker.check_model(onnx_model_fp32)
#         print("ONNX FP32 model checked successfully.")

#     except Exception as e:
#         print(f"Error during ONNX FP32 export: {e}")
#         return

#     # --- 7. Quantize ONNX Model to INT8 (Dynamic Quantization for CPU) ---
#     onnx_int8_path = os.path.join(args.output_dir, "model_int8.onnx")
#     print(f"Applying dynamic quantization (INT8) for CPU: {onnx_int8_path}")

#     try:
#         quantize_dynamic(
#             model_input=onnx_fp32_path,
#             model_output=onnx_int8_path,
#             weight_type=QuantType.QInt8,  # Quantize weights to INT8
#             # optimize_model=True, # Deprecated, optimization is on by default
#             # extra_options={'ActivationSymmetric': True} # Example extra option
#         )
#         print("ONNX INT8 model quantized and saved successfully.")

#         # Verify the quantized ONNX model
#         onnx_model_int8 = onnx.load(onnx_int8_path)
#         onnx.checker.check_model(onnx_model_int8)
#         print("ONNX INT8 model checked successfully.")

#     except Exception as e:
#         print(f"Error during ONNX INT8 quantization: {e}")
#         return

#     print("\n--- Process Complete ---")
#     print(f"FP32 ONNX model saved at: {onnx_fp32_path}")
#     print(f"INT8 Quantized ONNX model saved at: {onnx_int8_path}")
#     print("\nTo run inference with the quantized model using ONNX Runtime (Python):")
#     print("```python")
#     print("import onnxruntime")
#     print("import numpy as np")
#     print("# Create a dummy input matching your model's expected input shape and type")
#     print(
#         f"# dummy_input_np = np.random.randn(1, {input_channels}, {cfg.train_height}, {cfg.train_width}).astype(np.float32)"
#     )
#     print(f"ort_session = onnxruntime.InferenceSession('{onnx_int8_path}')")
#     print(f"input_name = ort_session.get_inputs()[0].name")
#     print("# outputs = ort_session.run(None, {input_name: dummy_input_np})")
#     print("# print(f'Number of outputs: {len(outputs)}')")
#     print("# for i, output in enumerate(outputs):")
#     print("#     print(f'Output {i} shape: {output.shape}')")
#     print("```")
#     print(
#         "\nConsider further optimizations with ONNX Runtime Execution Providers (e.g., OpenVINO for Intel CPUs) for potentially more speedup."
#     )


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Convert PyTorch model to ONNX, quantize for CPU, and optimize."
#     )
#     parser.add_argument(
#         "--config",
#         type=str,
#         required=True,
#         help="Path to the model configuration file (.py). Example: configs/smartrollerz_res18.py",
#     )
#     parser.add_argument(
#         "--pth_model",
#         type=str,
#         required=True,
#         help="Path to the input PyTorch model (.pth) file.",
#     )
#     parser.add_argument(
#         "--output_dir",
#         type=str,
#         default="./onnx_models",
#         help="Directory to save the exported ONNX models.",
#     )
#     parser.add_argument(
#         "--opset", type=int, default=12, help="ONNX opset version for export."
#     )

#     # Ensure the script is run from a context where 'utils' and 'model' can be imported
#     # e.g., from the root of the ufldv2 project or by setting PYTHONPATH appropriately.
#     # Example: python deploy/optimize_for_cpu.py --config configs/your_config.py --pth_model path/to/your/model.pth

#     parsed_args = parser.parse_args()
#     main(parsed_args)
