#!/usr/bin/env python3
"""Publish the images of a folder as a ROS 2 image topic (offline testing without camera)."""
import argparse
import glob
import os

import cv2
import cv_bridge
import rclpy
import sensor_msgs.msg
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


class ImageFolderPublisher(Node):
    """Publishes all images of a folder in sorted order with a fixed rate."""

    def __init__(self, folder: str, topic: str, rate: float, loop: bool):
        """
        Initialize the ImageFolderPublisher.

        Arguments:
            folder -- Folder containing the images.
            topic -- Topic to publish the images on.
            rate -- Publish rate in Hz.
            loop -- Restart at the first image after the last one.
        """
        super().__init__("image_folder_publisher")
        self.files = sorted(
            f
            for ext in ("*.jpg", "*.png")
            for f in glob.glob(os.path.join(folder, ext))
        )
        if not self.files:
            raise FileNotFoundError(f"No images found in {folder}")

        self.loop = loop
        self.index = 0
        self.bridge = cv_bridge.CvBridge()
        self.publisher = self.create_publisher(sensor_msgs.msg.Image, topic, 1)
        self.timer = self.create_timer(1.0 / rate, self.publish_next)
        self.get_logger().info(
            f"Publishing {len(self.files)} images from {folder} on {topic} at {rate} Hz"
        )

    def publish_next(self):
        """Publish the next image."""
        if self.index >= len(self.files):
            if not self.loop:
                self.get_logger().info("All images published.")
                raise SystemExit
            self.index = 0

        path = self.files[self.index]
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        msg = self.bridge.cv2_to_imgmsg(image, encoding="mono8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = os.path.basename(path)
        self.publisher.publish(msg)

        if self.index % 50 == 0:
            self.get_logger().info(f"[{self.index}/{len(self.files)}] {path}")
        self.index += 1


def main():
    """Parse the arguments and spin the publisher."""
    default_folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "models", "eval_2025-03-06"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", default=default_folder)
    parser.add_argument("--topic", default="/camera/image/bev")
    parser.add_argument("--rate", type=float, default=10.0, help="Hz")
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()

    rclpy.init()
    node = ImageFolderPublisher(args.folder, args.topic, args.rate, args.loop)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
