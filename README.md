# Lane Detection AI

[![Build Test](https://github.com/DHBW-Smart-Rollerz/ros2_exaple_package/actions/workflows/build-test.yaml/badge.svg)](https://github.com/DHBW-Smart-Rollerz/ros2_exaple_package/actions/workflows/build-test.yaml)

This repository contains the ros2 jazzy package for the ai lane detection based on the [ultra fast lane detection v2](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2).


# Lane Detection AI

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


## Structure

- `config/`: All configurations for ROS and the model
- `launch/`: Contains the launch files
- `models/`: Contains the models
- `resource/`: Contains the package name (required to build with colcon)
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
