import os

import cv_bridge
import rclpy
import rclpy.node
import rclpy.wait_for_message
import sensor_msgs.msg
import std_msgs.msg
from ament_index_python.packages import get_package_share_directory
from timing import timer

from ros2_example_package.example_model import model


class ROS2ExampleNode(rclpy.node.Node):
    """ROS2 Example Node."""

    def __init__(self):
        """Initialize the ROS2ExampleNode."""
        super().__init__("ros2_example_node")

        # Get the parameters from the yaml file
        self.package_share_path = get_package_share_directory("ros2_example_package")
        self.get_logger().info(f"Package Path: {self.package_share_path}")

        # Load the parameters from the ROS parameter server and initialize
        # the publishers and subscribers
        self.load_ros_params()
        self.init_publisher_and_subscriber()

        # Create required objects
        self.cv_bridge = cv_bridge.CvBridge()

        self.model = model.ExampleModel(
            root=self.package_share_path, model_config_path=self.model_config_path
        )

        self.get_logger().info("ROS2ExampleNode initialized")

    def load_ros_params(self):
        """Gets the parameters from the ROS parameter server."""
        # All command line arguments from the launch file and parameters from the
        # yaml config file must be declared here with a default value.
        self.declare_parameters(
            namespace="",
            parameters=[
                ("debug", False),
                ("image_path", "resources/img/example.png"),
                ("model_config_path", "config/model.yaml"),
                ("image_topic", "/camera/undistorted"),
                ("result_topic", "/example/result"),
                ("debug_image_topic", "/example/debug_image"),
                ("example_value", 500),
            ],
        )

        # Get parameters from the ROS parameter server into a local variable
        self.debug = self.get_parameter("debug").get_parameter_value().bool_value
        self.image_path = os.path.join(
            self.package_share_path,
            self.get_parameter("image_path").get_parameter_value().string_value,
        )  # full path to the image
        self.model_config_path = (
            self.get_parameter("model_config_path").get_parameter_value().string_value
        )

        self.image_topic = (
            self.get_parameter("image_topic").get_parameter_value().string_value
        )

        self.result_topic = (
            self.get_parameter("result_topic").get_parameter_value().string_value
        )

        self.debug_image_topic = (
            self.get_parameter("debug_image_topic").get_parameter_value().string_value
        )

        self.example_value = (
            self.get_parameter("example_value").get_parameter_value().integer_value
        )

    def init_publisher_and_subscriber(self):
        """Initializes the subscribers and publishers."""
        self.image_subscriber = self.create_subscription(
            sensor_msgs.msg.Image, self.image_topic, self.image_callback, 1
        )
        self.result_publisher = self.create_publisher(
            std_msgs.msg.Float32, self.result_topic, 1
        )

        if self.debug:
            self.debug_image_publisher = self.create_publisher(
                sensor_msgs.msg.Image, self.debug_image_topic, 1
            )

    def image_callback(self, msg: sensor_msgs.msg.Image):
        """Executed by the ROS2 system whenever a new image is received."""
        # Execute the prediction
        self.execute_prediction(msg)

    def wait_for_message_and_execute(self):
        """Waits for a new message on the image topic and then executes the prediction."""
        # Wait for a new message on the image topic
        _, msg = rclpy.wait_for_message.wait_for_message(
            sensor_msgs.msg.Image, self, self.image_topic
        )

        # Execute the prediction
        self.execute_prediction(msg)

    @timer.Timer(name="total", filter_strength=40)
    def execute_prediction(self, msg: sensor_msgs.msg.Image):
        """
        Execute the prediction.

        Arguments:
            msg -- The image message.
        """
        with timer.Timer(name="msg_transport", filter_strength=40):
            # The image has to be retrieved from the message
            image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="8UC1")

        with timer.Timer(name="prediction", filter_strength=40):
            # Get the result from the model
            result = self.model.predict(image)

        with timer.Timer(name="publish", filter_strength=40):
            # Publish the result as the custom ROS2 message defined in this package
            self.result_publisher.publish(
                std_msgs.msg.Float32(
                    data=result,
                )
            )

        if self.debug:
            with timer.Timer(name="debug", filter_strength=40):
                # Create the debug image. Here it is just the image filled with
                # the example value.
                debug_image = image.copy()
                debug_image.fill(self.example_value)

                self.debug_image_publisher.publish(
                    self.cv_bridge.cv2_to_imgmsg(image, encoding="8UC1")
                )

        timer.Timer(logger=self.get_logger().info).print()


def main(args=None):
    """
    Main function to start the ROS2ExampleNode.

    Keyword Arguments:
        args -- Launch arguments (default: {None})
    """
    rclpy.init(args=args)
    node = ROS2ExampleNode()

    # We have 2 options on how to run the node:
    # 1. Let the node idle in the background with 'rclpy.spin(node)' if we want to let
    #   subscriber callback function handle the execution of our code.
    #   TODO: is it possible in this way, that our callback gets executed multiple times
    #       in parallel?
    # 2. Run the node in a while loop that waits for incoming messages and then executes
    #   our code. This makes sure that always the latest message is processed and never
    #   multiple messages in parallel. It should be used if the processing of the
    #   message/execution of our code takes longer than the time between incoming #
    #   messages.

    try:
        use_wait_for_message = True
        if use_wait_for_message:
            while rclpy.ok():
                node.wait_for_message_and_execute()
                rclpy.spin_once(node)
        else:
            rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        # Shutdown if not already done by the ROS2 launch system
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
