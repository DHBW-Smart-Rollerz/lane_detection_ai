## Preface

This is a quick guide to setting up Hailo 8 Accelerator and needed software packages. This guide has been tested on Python version `3.10`. Although higher versions might work in the future, currently version `3.10` is the highest available version that ensures correct installation of all software packages. Consider this when downloading wheel files on the Download platform of Hailo.
## Prerequisites

Before you start setting up Hailo tools for using AI Accelerator power, you of course need the accelerator itself. It will most likely be in a m.2 NVMe format.

> Connectivity to system
> In order to use the Hailo 8 Accelerator, it must be connected to PCIe. This can only either be achieved by plugging in the Accelerator to a m.2 NVMe m-type slot or by using a m.2 NVMe to Thunderbolt adapter (only suitable if system supports thunderbolt).

If your system only has one m.2 NVMe slot, you might consider using a m.2 NVMe to USB adapter for mass storage and using internal slot for the Accelerator.
## Step 1: Install `dkms`, `build-essential` and `python3.<version>-dev`

For the driver to compile for your specific Linux kernel, install `dkms` and `build-essential`  with:

```shell
sudo apt install dkms build-essential
```

To install `python3.<version>-dev`, execute:

```shell
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
```

Then install the development package:
```shell
sudo apt install python3.<version>-dev
```
## Step 2: Install Dataflow Compiler

The Dataflow Compiler is needed for model conversion and compilation to Hailo binary format. In order to install it, follow <a href="https://hailo.ai/developer-zone/documentation/dataflow-compiler-v3-33-0/?sp_referrer=install/install.html">this installation guide</a>. 

> Virtual Environments
> To install the dataflow compiler python wheel, it is highly recommended to create a virtual environment with the supported python version from the downloads <a href="https://hailo.ai/developer-zone/software-downloads/?product=ai_accelerators&device=hailo_8_8l">here</a>. From here on, always make sure that you have the created environment activated when installing wheels and packages in further steps.
## Step 3: Install HailoRT and Hailo PCIe Driver

Before you install the packages, ensure that `Secure Boot` is disabled. In order to achieve this, reboot your system and enter `UEFI` Setup, then under `Boot Configuration` (on Dell UEFI) uncheck the `Enable Secure Boot` checkbox.

Then simply download the packages `hailort_<version>_amd64.deb` and `hailort-pcie-driver_<version>_all.deb` from <a href="https://hailo.ai/developer-zone/software-downloads/?product=ai_accelerators&device=hailo_8_8l">here</a>, make sure that you select the right platform. After downloading the `.deb` files, execute this:

```bash
sudo dpkg --install hailort_4.23.0_$(dpkg --printarchitecture).deb hailort-pcie-driver_4.23.0_all.deb
```

Then reboot your systems in order for the installation to take effect.

## Step 4: Installation of pyHailoRT

PyHailoRT is a python package which wraps the C/C++ API that is called from your applications. It allows you to load models, send and receive data from the model.

To run your models on Hailo device, you need to install the modules from the `hailort-<version>-<python_tag>-<abi_tag>-<platform_tag>` file (download from <a href="https://hailo.ai/developer-zone/software-downloads/?product=ai_accelerators&device=hailo_8_8l">here</a>). Then execute this:

```bash
pip install ./hailort-<version>-<python_tag>-<abi_tag>-<platform_tag>.whl
```

## Test the Accelerator

To ensure the hardware runs correctly, you might download this `HEF` file (model weights in Hailo format):

```shell
wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.11.0/hailo8/yolov8s.hef
```

Then, you can run a benchmark on your Hailo 8 with:

```shell
hailortcli benchmark yolov8s.hef
```
