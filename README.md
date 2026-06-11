# Lane Detection AI

[![Build Test](https://github.com/DHBW-Smart-Rollerz/ros2_exaple_package/actions/workflows/build-test.yaml/badge.svg)](https://github.com/DHBW-Smart-Rollerz/ros2_exaple_package/actions/workflows/build-test.yaml)

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

If your model config enables the Hailo backend, this node imports `hailo_platform` (HailoRT Python bindings). Those bindings **must be installed for the exact Python interpreter that executes the ROS2 entrypoint script**.

Install the HailoRT wheel **matching that Python version** (for Jazzy this is typically CPython 3.12, so you need a `cp312` wheel). Example:

```bash
/home/$USER/.pyenv/versions/3.12.11/bin/python3 -m pip install ~/path/to/hailort-<VERSION>-cp312-cp312-linux_x86_64.whl
```

Notes:

- The `hailort` package is typically not available on PyPI; you usually install it from the `.whl` provided by Hailo.
- If you only have a `cp310` or `cp313` wheel, it will **not** work with Python 3.12. Download the matching `cp312` wheel from your HailoRT distribution.
- You can find .whl files in the Hailo Developer Portal: https://developer.hailo.ai/portal/en/downloads/hailort/ or in the `/resource` directory

## Usage

To run this package, you will need the pre-trained model weights. As of September 2024, two versions are available: a dense model and a sparse model. You can download them [here](https://it-nas.dhbw-stuttgart.de:5001/?launchApp=SYNO.SDS.Drive.Application#file_id=842460996588058121). Place the downloaded model in the `models/` folder, and the corresponding configuration file in the `config/` folder.

**Important:**
Ensure that you update the `model_config_path` in the `config/ros_params.yaml` file, and set the correct path for the model in the configuration file's `test_model` field.

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

The `scripts/` directory contains SLURM job scripts used to prepare calibration data and compile the model for Hailo hardware on a SLURM-managed HPC cluster. All three scripts use [Pyxis](https://github.com/NVIDIA/pyxis) (`--container-image`) to run inside a pre-built Enroot container.

---

### `import_calib_images.slurm`

**Purpose:** Copies BEV (bird's-eye-view) images from a ROS bag dataset directory into the calibration samples folder and resizes them to the model's expected input resolution.

**When to use:** Run this once before compiling a new HEF to populate `hailo/opt_samples/` with correctly sized calibration images.

**Key parameters** (set via `sbatch --export=KEY=VALUE` or by editing the script):

| Variable | Default | Description |
|---|---|---|
| `SRC_ROOT` | `.../dataset/smartrollerz_bev/data/rosbags` | Source directory containing the raw dataset |
| `DST_ROOT` | `.../hailo/opt_samples` | Destination for calibration images |
| `TARGET_WIDTH` | `512` | Target image width in pixels |
| `TARGET_HEIGHT` | `384` | Target image height in pixels |

**Usage:**
```bash
# Default settings
sbatch scripts/import_calib_images.slurm

# Override resolution
sbatch --export=TARGET_WIDTH=640,TARGET_HEIGHT=480 scripts/import_calib_images.slurm
```

The script selects only files with `_bev` in their name and resizes them in-place using OpenCV.

---

### `flatten_opt_samples.slurm`

**Purpose:** Flattens a nested directory structure inside `hailo/opt_samples/` by moving all image files from subdirectories into the root of that folder and deleting the now-empty subdirectories. This is required because the Hailo DFC calibration tool expects all images in a single flat directory.

**When to use:** Run this after `import_calib_images.slurm` if the copied images ended up in subdirectories (e.g. organised by rosbag session).

**Key parameters:**

| Variable | Default | Description |
|---|---|---|
| `OPT_SAMPLES_ROOT` | `.../hailo/opt_samples` | Directory to flatten |
| `DRY_RUN` | `0` | Set to `1` to preview moves without touching any files |

**Usage:**
```bash
# Preview what would be moved (dry run)
sbatch --export=DRY_RUN=1 scripts/flatten_opt_samples.slurm

# Actually flatten
sbatch scripts/flatten_opt_samples.slurm
```

Name collisions are resolved automatically by appending a numeric suffix (e.g. `image_0001.jpg`). Non-image files are left in place and reported as skipped.

---

### `compile_pth_to_hef.slurm`

**Purpose:** End-to-end pipeline that converts a PyTorch `.pth` model checkpoint into a Hailo HEF file ready for deployment on Hailo-8 hardware. It performs two steps in sequence:

1. **PTH → ONNX** – calls `scripts/convert_pth_onnx.py` to export the model to ONNX format.
2. **ONNX → HEF** – calls `scripts/compile_onnx_hef.sh`, which uses the Hailo Dataflow Compiler (DFC) to parse, optimise (using the calibration images), and compile the ONNX model into a `.hef` file.

The script automatically installs Miniconda and all required DFC dependencies inside the job if they are not already present, so no manual environment setup on the cluster is needed.

**When to use:** Run this after training a new model and after calibration images are ready in `hailo/opt_samples/`.

**Key parameters:**

| Variable | Default | Description |
|---|---|---|
| `MODEL_CONFIG` | `config/21.04.2026_config.py` | Path to the model config file |
| `PTH_PATH` | `models/21.04.2026.pth` | Path to the trained PyTorch checkpoint |
| `CALIB_DIR` | `.../hailo/opt_samples` | Directory of calibration images |
| `OUT_DIR` | `.../hailo/build/21.04.2026/` | Output directory for ONNX and HEF files |
| `ONNX_OUT` | `${OUT_DIR}/lane_detection_ai.onnx` | Path for the intermediate ONNX export |
| `HAILO_WHEELS_DIR` | `${PROJECT_ROOT}/hailo/dfc_wheels` | Directory containing Hailo DFC `.whl` files |
| `HW_ARCH` | `hailo8` | Target Hailo hardware architecture |
| `CALIB_SAMPLES` | `1024` | Number of calibration samples used during optimisation |

**Requirements:** The Hailo Dataflow Compiler wheels (`hailo_dataflow_compiler-*.whl` or `hailo_sdk_client-*.whl`) must be present in `HAILO_WHEELS_DIR` before submission. These are not on PyPI; obtain them from the [Hailo Developer Portal](https://developer.hailo.ai/portal/en/downloads/hailort/) or the `/resource` directory.

**Usage:**
```bash
# Default (uses paths hard-coded in the script)
sbatch scripts/compile_pth_to_hef.slurm

# Custom model and output directory
sbatch --export=MODEL_CONFIG=config/my_config.py,PTH_PATH=models/my_model.pth,OUT_DIR=/raid/.../hailo/build/my_model/ \
  scripts/compile_pth_to_hef.slurm
```

The final HEF file is written to `OUT_DIR`. Copy it to `models/` and update `config/ros_params.yaml` to use it with the ROS node.

---

### Typical workflow

```
import_calib_images  →  flatten_opt_samples  →  compile_pth_to_hef
```

1. `import_calib_images.slurm` – populate calibration images from the dataset.
2. `flatten_opt_samples.slurm` – flatten into a single directory if needed.
3. `compile_pth_to_hef.slurm` – compile the trained model to a `.hef` for Hailo deployment.


## Structure

- `config/`: All configurations for ROS and the model
- `launch/`: Contains the launch files
- `models/`: Contains the models
- `resource/`: Contains the package name (required to build with colcon)
- `scripts/`: Contains SLURM job scripts for model compilation and data preparation
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