import glob
import os

from setuptools import find_packages, setup

package_name = "lane_detection_ai"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob.glob(os.path.join("launch", "*launch.py")),
        ),
        (
            os.path.join("share", package_name, "config"),
            glob.glob(os.path.join("config", "*.*")),
        ),
        (
            os.path.join("share", package_name, "models"),
            glob.glob(os.path.join("models", "*.*")),
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
