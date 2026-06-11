# DHBW Smart-Rollerz Lane Detection AI

This project is a custom implementation based on [Ultra-Fast-Lane-Detection-V2](https://github.com/cfzd/Ultra-Fast-Lane-Detection-V2) specifically tailored for the DHBW Smart-Rollerz team. It integrates PyTorch Lightning, Optuna for hyperparameter optimization, MLflow for experiment tracking, and a full suite of SLURM scripts for containerized execution on a DGX cluster using Enroot/Pyxis.

## Project Structure & Features
- **PyTorch Lightning**: Modular training loops in `train_lightning.py`.
- **Optuna Integration**: Automated hyperparameter search in `train_optuna_lightning.py`.
- **MLflow Tracking**: Integrated experiment logging and a UI hosted via SLURM (`mlflow_ui.slurm`).
- **SLURM Workflows**: End-to-end containerized pipelines for building, training, evaluating, and tracking.
- **Docker/Pyxis**: Docker-based build process (`docker_build_train.slurm`) which exports the image to `.sqsh` for efficient cluster execution using `srun`.

## Getting Started

### 1. Build the Container Environment
Before running any scripts, build the Docker image and convert it to a Pyxis-compatible `.sqsh` file:
```bash
sbatch docker_build_train.slurm
```
This uses `Dockerfile` to create the image and exports it to `~/containers/lane_detection_ai_training.sqsh`.
**Note:** Make sure to update the `USER` and `SQSH_PATH` variables in the `.slurm` file to match your account.

### 2. Configuration & Data Preparation
- **Config Files**: Model hyper-parameters and dataset paths are set in the `configs/` directory (e.g., `configs/smartrollerz_res18_bev.py`).
- **Data Conversion**: You can convert CVAT annotations or format your datasets using our provided SLURM scripts:
  - `convert_cvat.slurm` & `convert_cvat_bev.slurm`
  - `draw_labels_on_images.slurm`
  - `resize_labels.slurm`

### 3. Training the Model
You have two options for training: standard training and hyperparameter optimization.

#### Standard PyTorch Lightning Training
To train the model using a predefined configuration:
```bash
sbatch train_lightning.slurm
```
It logs metrics with MLflow and saves checkpoints to the shared results directory. Ensure `OWNER_RESULTS_DIR` and `LOCAL_PROJECT_DIR` in the `.slurm` script are correctly pointing to your directories.

#### Optuna Hyperparameter Optimization
To run an automated hyperparameter sweep with Optuna:
```bash
sbatch train_optuna_lightning.slurm
```
This script launches `train_optuna_lightning.py`. It uses a shared SQLite database for Optuna. You can adjust the number of trials (`--optuna_n_trials`), sampler, and other arguments directly inside the SLURM script.

### 4. Checkpoint Conversion & Inference

#### Convert Lightning Checkpoint to PyTorch `.pth`
Training produces `.ckpt` files containing optimizer states. To convert a `.ckpt` model into a standard `.pth` state dictionary for inference or TensorRT deployment:
```bash
sbatch convert_ckpt_to_pth.slurm
```

#### Single Image Inference
To test the model on a single image:
```bash
sbatch inference.slurm
```
Update the `--image` and `--weights` parameters inside the script beforehand.

#### Batch Inference
To run inference on an entire folder of images and generate output overlays:
```bash
sbatch batch_inference.slurm
```
Remember to update the `TRAINING_RUN_FOLDER` and set `USE_BEV=true` (or `false`) based on your configuration. It automatically discovers the best checkpoint in your run directory.

### 5. Tracking Results with MLflow
To view training metrics, logs, and Optuna results, start the MLflow UI on the cluster:
```bash
sbatch mlflow_ui.slurm
```
Check the output log (`slurm-outs/slurm-out-mlflow-*.out`). It will display an SSH port-forwarding command (e.g., `ssh -N -L 5050:<node_hostname>:5050 <your_username>@<dgx_server>`). Run that on your local machine and open `http://localhost:5050` in your browser.

To stop the UI:
```bash
scancel <JOB_ID>
```
Or run `sbatch stop_mlflow_ui.slurm` if available.

## Overview of SLURM Scripts

| Script | Purpose |
| ------ | ------- |
| `docker_build_train.slurm` | Builds the Docker image and creates a Pyxis `.sqsh` file. |
| `train_lightning.slurm` | Runs standard model training using PyTorch Lightning. |
| `train_optuna_lightning.slurm` | Runs hyperparameter search with Optuna. |
| `convert_ckpt_to_pth.slurm` | Extracts the raw model weights `.pth` from a Lightning `.ckpt`. |
| `inference.slurm` | Runs inference on a single test image. |
| `batch_inference.slurm` | Runs inference over a folder of images, picking the best checkpoint. |
| `mlflow_ui.slurm` | Starts the MLflow UI server for tracking metrics and trials. |
| `convert_cvat*.slurm` | Converts CVAT annotations into the required dataset format. |
| `add_user_folder.slurm` | Manages file/folder permissions across different cluster users. |

## Customizing User Variables
Many `.slurm` scripts include a `USER="<name>"` or hardcoded directory paths at the top. **Always update these variables** to match your DGX account setup before submitting a job. For collaborative runs, ensure `OWNER_PROJECT_DIR` and `OWNER_RESULTS_DIR` point to the shared team location.

## Acknowledgment
Original Ultra-Fast-Lane-Detection-V2 implementation by Zequn Qin, et al. For details on the original methodology, refer to their [paper](https://arxiv.org/abs/2206.07389).
