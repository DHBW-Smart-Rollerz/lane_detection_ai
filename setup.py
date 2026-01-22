import os

from setuptools import find_packages, setup

package_name = "lane_detection_ai"


def _recursive_data_files(install_path: str, source_path: str):
    """Collect data_files entries recursively.

    This avoids depending on external helpers and ensures nested folders
    (e.g. models/hailo/*) are installed into the ROS2 share directory.
    """
    data_files = []
    for root, _, files in os.walk(source_path):
        if not files:
            continue
        rel_root = os.path.relpath(root, source_path)
        dest = install_path if rel_root == "." else os.path.join(install_path, rel_root)
        srcs = [os.path.join(root, f) for f in files]
        data_files.append((dest, srcs))
    return data_files

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        *_recursive_data_files(
            install_path=os.path.join("share", package_name, "config"),
            source_path="config",
        ),
        *_recursive_data_files(
            install_path=os.path.join("share", package_name, "launch"),
            source_path="launch",
        ),
        *_recursive_data_files(
            install_path=os.path.join("share", package_name, "models"),
            source_path="models",
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Smart Rollerz",
    maintainer_email="info@dhbw-smartrollerz.org",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            f"lane_detection_ai_node = {package_name}.lane_detection_ai_node:main",
        ],
    },
)
