import os

from ament_index_python import get_package_share_directory
from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """
    Generate the launch description.

    Returns:
        LaunchDescription -- The launch description.
    """
    python_executable = os.getenv("PYTHON_EXECUTABLE", "/usr/bin/python3")
    debug = LaunchConfiguration("debug")
    params_file = LaunchConfiguration("params_file")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "debug", default_value="False", description="Enable debug mode"
            ),
            DeclareLaunchArgument(
                "params_file",
                default_value=os.path.join(
                    get_package_share_directory("lane_detection_ai"),
                    "config",
                    "ros_params.yaml",
                ),
                description="Path to the ROS parameters file",
            ),
            Node(
                package="lane_detection_ai",
                namespace="",
                executable="lane_detection_ai_node",
                name="lane_detection_ai_node",
                parameters=[
                    {"debug": debug},
                    params_file,
                ],
                prefix=[python_executable],
            ),
        ]
    )
