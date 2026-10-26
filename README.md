# Lane Detection AI

[![Build](https://github.com/DHBW-Smart-Rollerz/lane_detection_ai/actions/workflows/build.yaml/badge.svg)](https://github.com/DHBW-Smart-Rollerz/lane_detection_ai/actions/workflows/build.yaml)
[![Lint](https://github.com/DHBW-Smart-Rollerz/lane_detection_ai/actions/workflows/lint.yaml/badge.svg)](https://github.com/DHBW-Smart-Rollerz/lane_detection_ai/actions/workflows/lint.yaml)
[![Test](https://github.com/DHBW-Smart-Rollerz/lane_detection_ai/actions/workflows/test.yaml/badge.svg)](https://github.com/DHBW-Smart-Rollerz/lane_detection_ai/actions/workflows/test.yaml)

This repository contains the ros2 jazzy package for the ai lane detection based on the [ultra fast lane detection v2](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2).


## Installation

1. Clone the package into the `smarty_workspace/src` directory:
   ```bash
   git clone <repository-url> smarty_workspace/src
   ```
2. Install the required Python dependencies using the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the trained model and its corresponding configuration file. (See the [Usage](#usage) section for more details.)
4. Please set the following ENV variable in your .bashrc/.zshrc if not already done:
   ```bash
   PYTHON_EXECUTABLE="/home/$USER/.pyenv/versions/default/bin/python3" # Change this to the python3 executable path of your pyenv
   ```

### HailoRT (optional, required for Hailo backend)

For a detailed installation guide, see [HAILO_INSTALL.md](HAILO_INSTALL.md).

If the model config uses a `.hef` model (Hailo backend), this node imports `hailo_platform` (HailoRT Python bindings). Those bindings **must be installed for the exact Python interpreter that executes the ROS2 entrypoint script**.

Install the HailoRT wheel **matching that Python version** (for Jazzy this is typically CPython 3.12, so you need a `cp312` wheel). Example:

```bash
/home/$USER/.pyenv/versions/3.12.11/bin/python3 -m pip install ~/path/to/hailort-<VERSION>-cp312-cp312-linux_x86_64.whl
```

Notes:

- The `hailort` package is typically not available on PyPI; you usually install it from the `.whl` provided by Hailo.
- If you only have a `cp310` or `cp313` wheel, it will **not** work with Python 3.12. Download the matching `cp312` wheel from your HailoRT distribution.
- You can find .whl files in the Hailo Developer Portal: https://developer.hailo.ai/portal/en/downloads/hailort/ or in the `/resource` directory

## Usage

To run this package, you will need the pre-trained model weights. You can download them [here](https://it-nas.dhbw-stuttgart.de:5001/?launchApp=SYNO.SDS.Drive.Application#file_id=842460996588058121). Place the downloaded model in the `models/` folder, and the corresponding configuration file in the `config/` folder.

**Important:**
Ensure that you update the `model_config_path` in the `config/ros_params.yaml` file, and set the correct path for the model in the configuration file's `test_model` field (relative to the package).

### Backends

The backend is selected by the file extension of `test_model`:

| Extension | Backend | Notes |
|---|---|---|
| `.pth` | PyTorch (CPU) | Convert training checkpoints (`.ckpt`) with `scripts/convert_ckpt_to_pth.py` first |
| `.onnx` | ONNX Runtime (CPU) | Models with float32 input (normalized) and uint8 input (exported with `scripts/convert_pth_onnx.py`) are supported |
| `.hef` | Hailo-8 (HailoRT) | Compiled with `scripts/compile_onnx_hef.sh`, requires HailoRT (see below) |

Optional keys of the model config:

| Key | Default | Description |
|---|---|---|
| `mlp_mid_dim` | `2048` | Hidden size of the classification MLP. Set `512` for older models (`.pth` only) |
| `decode_mode` | `"softmax"` | `"softmax"` (sub-cell precision) or `"argmax"` (faster). The old key `hailo_decode_mode` is also read |
| `assume_bgr` | `False` | Convert color images from BGR to RGB (if `image_encoding` is `bgr8`). The old key `hailo_assume_bgr` is also read |
| `hailo_preprocess` | `"baked"` | `"baked"`: the HEF contains the normalization and gets the uint8 RGB image (what `convert_pth_onnx.py` exports). `"pytorch"`: the image is normalized on the host. The old key `hailo_preprocessor` is also read |
| `hailo_output_names` | - | Dict from `loc_row`, `exist_row`, `loc_col`, `exist_col` to the HEF output names. Only needed if the outputs can't be matched automatically (a warning is logged at startup) |

### ROS parameters

| Parameter | Default | Description |
|---|---|---|
| `model_config_path` | `config/model_sparse_config.py` | Model config, relative to the package |
| `image_encoding` | `8UC1` | Encoding of the subscribed images for CvBridge, e.g. `8UC1` (grayscale) or `bgr8` |
| `image_subscriber` | `/camera/undistorted` | Input image topic |
| `result_publisher` | `/lane_detection/result` | Detected lanes |
| `debug` | `False` | Publish a debug image on `debug_image_publisher` |

### Running the Node

To launch the node, use the following command:
```bash
ros2 launch lane_detection_ai lane_detection_ai.launch.py
```

For running with the debug image enabled, use:
```bash
ros2 launch lane_detection_ai lane_detection_ai.launch.py debug:=true
```


## Scripts

### Model conversion and benchmark

| Script | Purpose |
|---|---|
| `convert_ckpt_to_pth.py` | Extract the weights of a training checkpoint: `python3 scripts/convert_ckpt_to_pth.py models/model.ckpt models/model.pth` |
| `convert_pth_onnx.py` | Export a `.pth` model to ONNX with uint8 RGB input (normalization in the graph): `python3 scripts/convert_pth_onnx.py --base-path . --model-config config/model.py --pth models/model.pth --onnx-out build/model.onnx` |
| `compile_onnx_hef.sh` | Compile the ONNX model to a HEF with the Hailo Dataflow Compiler (DFC), see `./scripts/compile_onnx_hef.sh --help`. Set `CROP_RATIO` to the `crop_ratio` of the model config, so the calibration images are preprocessed like in the node |
| `benchmark_models.py` | Launch the node with several model configs and compare the `total` and `inference` times: `python3 scripts/benchmark_models.py config/a.py config/b.py --output results.json`. Needs images on the input topic (camera or rosbag) |

### SLURM jobs

The `*.slurm` scripts prepare calibration data and compile the model for Hailo hardware on a SLURM-managed HPC cluster. They use [Pyxis](https://github.com/NVIDIA/pyxis) (`--container-image`) to run inside a pre-built Enroot container.

Submit them from the repository root and create the log directory first (`mkdir -p slurm-outs`). All paths default to the layout below and can be overridden with `sbatch --export=ALL,KEY=VALUE,... scripts/<job>.slurm`:

```
<parent>/
├── containers/lane_detection_ai_run.sqsh                       # CONTAINER_IMAGE
├── lane_detection_ai/                                           # PROJECT_ROOT (this repo)
│   └── hailo/opt_samples/                                       # calibration images
└── lane_detection_ai_training/dataset/smartrollerz_bev/data/rosbags   # SRC_ROOT
```

---

### `import_calib_images.slurm`

**Purpose:** Copies BEV (bird's-eye-view) images from a ROS bag dataset directory into the calibration samples folder and resizes them.

**When to use:** Run this once before compiling a new HEF to populate `hailo/opt_samples/` with calibration images.

| Variable | Default | Description |
|---|---|---|
| `SRC_ROOT` | `<parent>/lane_detection_ai_training/dataset/smartrollerz_bev/data/rosbags` | Source directory containing the raw dataset |
| `DST_ROOT` | `${PROJECT_ROOT}/hailo/opt_samples` | Destination for calibration images |
| `TARGET_WIDTH` | `512` | Target image width in pixels |
| `TARGET_HEIGHT` | `384` | Target image height in pixels |

```bash
sbatch scripts/import_calib_images.slurm
sbatch --export=ALL,TARGET_WIDTH=640,TARGET_HEIGHT=480 scripts/import_calib_images.slurm
```

The script selects only files with `_bev` in their name and resizes them in-place using OpenCV.

---

### `flatten_opt_samples.slurm` (optional)

**Purpose:** Moves all images from subdirectories of `hailo/opt_samples/` into its root and deletes the empty subdirectories. `compile_onnx_hef.sh` searches the calibration directory recursively, so this is only needed for other tools that expect a flat directory.

| Variable | Default | Description |
|---|---|---|
| `OPT_SAMPLES_ROOT` | `${PROJECT_ROOT}/hailo/opt_samples` | Directory to flatten |
| `DRY_RUN` | `0` | Set to `1` to preview moves without touching any files |

```bash
sbatch --export=ALL,DRY_RUN=1 scripts/flatten_opt_samples.slurm
```

Name collisions are resolved automatically by appending a numeric suffix (e.g. `image_0001.jpg`). Non-image files are left in place and reported as skipped.

---

### `compile_pth_to_hef.slurm`

**Purpose:** Converts a PyTorch `.pth` model into a Hailo HEF file:

1. **PTH → ONNX** with `scripts/convert_pth_onnx.py`.
2. **ONNX → HEF** with `scripts/compile_onnx_hef.sh` (parse, optimize with the calibration images, compile).

The script installs Miniconda and the DFC dependencies inside the job if they are not already present.

| Variable | Default | Description |
|---|---|---|
| `MODEL_CONFIG` | required | Model config, relative to the repository |
| `PTH_PATH` | required | Trained checkpoint, relative to the repository |
| `CALIB_DIR` | `${PROJECT_ROOT}/hailo/opt_samples` | Directory of calibration images |
| `OUT_DIR` | `${PROJECT_ROOT}/hailo/build/<pth name>` | Output directory for ONNX and HEF files |
| `HAILO_WHEELS_DIR` | `${PROJECT_ROOT}/hailo/dfc_wheels` | Directory containing the Hailo DFC `.whl` files |
| `CROP_RATIO` | `1.0` | `crop_ratio` of the model config |
| `OPT_LEVEL` | `default` | Compiler optimization level (`default` or `max`) |
| `HW_ARCH` | `hailo8` | Target Hailo hardware architecture |
| `CALIB_SAMPLES` | `1024` | Number of calibration samples used during optimization |

**Requirements:** The Hailo Dataflow Compiler wheels (`hailo_dataflow_compiler-*.whl` or `hailo_sdk_client-*.whl`) must be present in `HAILO_WHEELS_DIR`. These are not on PyPI; obtain them from the [Hailo Developer Portal](https://developer.hailo.ai/portal/en/downloads/hailort/).

```bash
sbatch --export=ALL,MODEL_CONFIG=config/my_config.py,PTH_PATH=models/my_model.pth,CROP_RATIO=0.8 \
  scripts/compile_pth_to_hef.slurm
```

Copy the HEF from `OUT_DIR` to `models/` and set it as `test_model` in the model config.

---

### Typical workflow

```
import_calib_images  →  compile_pth_to_hef
```


## Structure

- `config/`: All configurations for ROS and the model
- `launch/`: Contains the launch files
- `models/`: Contains the models
- `resource/`: Contains the package name (required to build with colcon)
- `scripts/`: Contains model conversion/benchmark scripts and SLURM jobs for the Hailo compilation
- `lane_detection_ai/`: Contains all nodes and sources for the ros package
- `lane_detection_ai/model/`: Contains all sources of the ufldv2 model
- `test/`: Contains the tests
- `package.xml`: Contains metadata about the package
- `setup.py`: Used for Python package configuration
- `setup.cfg`: Additional configuration for the package
- `requirements.txt`: Python dependencies

<!--
The contributing section is currently not required as we do not plan to have public contribution.

## Contributing

Thank you for considering contributing to this repository! Here are a few guidelines to get you started:

1. Fork the repository and clone it locally.
2. Create a new branch for your contribution.
3. Make your changes and ensure they are properly tested.
4. Commit your changes and push them to your forked repository.
5. Submit a pull request with a clear description of your changes.

We appreciate your contributions and look forward to reviewing them! -->

## License

This repository is licensed under the MIT license. See [LICENSE](LICENSE) for details.
