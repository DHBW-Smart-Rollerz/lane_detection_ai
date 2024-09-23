import cv2
import cv_bridge
import geometry_msgs.msg
import lane_msgs.msg
import numpy as np
import rclpy
import rclpy.node
import rclpy.wait_for_message
import sensor_msgs.msg
from ament_index_python.packages import get_package_share_directory
from camera_preprocessing.transformation import (
    birds_eyed_view,
    coordinate_transform,
    distortion,
)
from timing import timer

from lane_detection_ai.model import model_wrapper as model


class LaneDetectionNode(rclpy.node.Node):
    """AI Lane Detection Node."""

    def __init__(self):
        """Initialize the LaneDetectionAiNode."""
        super().__init__("lane_detection_ai_node")

        # Get the parameters from the yaml file
        self.package_share_path = get_package_share_directory("lane_detection_ai")
        self.get_logger().info(f"Package Path: {self.package_share_path}")

        # Load the parameters from the ROS parameter server and initialize
        # the publishers and subscribers
        self.load_ros_params()
        self.init_publisher_and_subscriber()

        # Create required objects
        self.cv_bridge = cv_bridge.CvBridge()
        self.coord_transform = coordinate_transform.CoordinateTransform()
        self.distortion = distortion.Distortion(self.coord_transform._calib)
        self.bev = birds_eyed_view.Birdseye(
            self.coord_transform._calib, self.distortion
        )

        self.model = model.LaneDetectionAiModel(
            base_path=self.package_share_path, model_config_path=self.model_config_path
        )

        self.get_logger().info("LaneDetectionNode initialized")

    def load_ros_params(self):
        """Gets the parameters from the ROS parameter server."""
        # All command line arguments from the launch file and parameters from the
        # yaml config file must be declared here with a default value.
        self.declare_parameters(
            namespace="",
            parameters=[
                ("debug", False),
                ("model_config_path", "config/model_sparse_config.py"),
                ("image_topic", "/camera/undistorted"),
                ("result_topic", "/lane_detection/result"),
                ("debug_image_topic", "/lane_detection/debug_image"),
            ],
        )

        # Get parameters from the ROS parameter server into a local variable
        self.debug = self.get_parameter("debug").get_parameter_value().bool_value
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

    def init_publisher_and_subscriber(self):
        """Initializes the subscribers and publishers."""
        self.image_subscriber = self.create_subscription(
            sensor_msgs.msg.Image, self.image_topic, self.image_callback, 1
        )
        self.result_publisher = self.create_publisher(
            lane_msgs.msg.LaneDetectionResult, self.result_topic, 1
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

        if msg is None:
            return

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

        with timer.Timer(name="transformation", filter_strength=40):
            left_lane = (
                self.coord_transform.camera_to_world(result[0])
                if result[0] is not None
                else np.asarray([])
            )
            center_lane = (
                self.coord_transform.camera_to_world(result[1])
                if result[1] is not None
                else np.asarray([])
            )
            right_lane = (
                self.coord_transform.camera_to_world(result[2])
                if result[2] is not None
                else np.asarray([])
            )

        with timer.Timer(name="polyfit", filter_strength=40):
            # Fit lane to a polynomial function of 2nd degree. In our representation of
            # the polynomial function, the x and y are swapped: x = a * y² + b * y + c
            left_lane_coeff = None
            center_lane_coeff = None
            right_lane_coeff = None

            if len(left_lane) > 0:
                left_lane_coeff = np.asarray(
                    np.polyfit(left_lane.T[1], left_lane.T[0], 2)
                )
            if len(center_lane) > 0:
                center_lane_coeff = np.asarray(
                    np.polyfit(center_lane.T[1], center_lane.T[0], 2)
                )
            if len(right_lane) > 0:
                right_lane_coeff = np.asarray(
                    np.polyfit(right_lane.T[1], right_lane.T[0], 2)
                )

            # Build the two driving lanes (left and right) from the center lane
            # and the left/right lane
            driving_lane_left_coeff = [0, 0, 0]
            driving_lane_right_coeff = [0, 0, 0]

            if right_lane_coeff is not None and center_lane_coeff is not None:
                driving_lane_right_coeff = np.average(
                    [right_lane_coeff, center_lane_coeff], axis=0
                )
            if center_lane_coeff is not None and left_lane_coeff is not None:
                driving_lane_left_coeff = np.average(
                    [center_lane_coeff, left_lane_coeff], axis=0
                )

        with timer.Timer(name="publish", filter_strength=40):
            # Generate vec3 arrays for the output message
            left_lane_vec3s = [
                geometry_msgs.msg.Vector3(
                    x=float(coord[0]), y=float(coord[1]), z=float(coord[2])
                )
                for coord in left_lane
            ]
            center_lane_vec3s = [
                geometry_msgs.msg.Vector3(
                    x=float(coord[0]), y=float(coord[1]), z=float(coord[2])
                )
                for coord in center_lane
            ]
            right_lane_vec3s = [
                geometry_msgs.msg.Vector3(
                    x=float(coord[0]), y=float(coord[1]), z=float(coord[2])
                )
                for coord in right_lane
            ]
            self.result_publisher.publish(
                lane_msgs.msg.LaneDetectionResult(
                    left=lane_msgs.msg.Lane(
                        points=left_lane_vec3s,
                        detected=len(left_lane_vec3s) > 0,
                    ),
                    center=lane_msgs.msg.Lane(
                        points=center_lane_vec3s,
                        detected=len(center_lane_vec3s) > 0,
                    ),
                    right=lane_msgs.msg.Lane(
                        points=right_lane_vec3s,
                        detected=len(right_lane_vec3s) > 0,
                    ),
                    trajectory_left=geometry_msgs.msg.Vector3(
                        x=float(driving_lane_left_coeff[0]),
                        y=float(driving_lane_left_coeff[1]),
                        z=float(driving_lane_left_coeff[2]),
                    ),
                    trajectory_right=geometry_msgs.msg.Vector3(
                        x=float(driving_lane_right_coeff[0]),
                        y=float(driving_lane_right_coeff[1]),
                        z=float(driving_lane_right_coeff[2]),
                    ),
                )
            )

        # self.get_logger().info(left_lane)

        if self.debug:
            with timer.Timer(name="debug_image", filter_strength=40):
                debug_image = image.copy()
                debug_image = cv2.cvtColor(debug_image, cv2.COLOR_GRAY2RGB)

                if len(left_lane) > 0:
                    for coord in self.coord_transform.world_to_camera(left_lane).astype(
                        int
                    ):
                        cv2.circle(debug_image, coord, 5, (255, 0, 0), -1)
                if len(center_lane) > 0:
                    for coord in self.coord_transform.world_to_camera(
                        center_lane
                    ).astype(int):
                        cv2.circle(debug_image, coord, 5, (0, 255, 0), -1)
                if len(right_lane) > 0:
                    for coord in self.coord_transform.world_to_camera(
                        right_lane
                    ).astype(int):
                        cv2.circle(debug_image, coord, 5, (0, 0, 255), -1)

                # Draw the trajectories
                self._draw_trajectory(debug_image, driving_lane_left_coeff)
                self._draw_trajectory(debug_image, driving_lane_right_coeff)

                self.debug_image_publisher.publish(
                    self.cv_bridge.cv2_to_imgmsg(debug_image, encoding="rgb8")
                )

        timer.Timer(logger=self.get_logger().info).print()

    def _draw_trajectory(
        self, img: np.ndarray, driving_lane_coeff: np.ndarray
    ) -> np.ndarray:
        """Draws the trajectory on the given image.

        Args:
            img (np.ndarray): Image to draw the trajectory on.
            driving_lane_coeff (np.ndarray): Coefficients of the polynomial function
            of the driving lane.

        Returns:
            np.ndarray: Image with the trajectory drawn on it.
        """
        # Create lane points for fitted parabola
        row_points = np.asarray(range(-500, 2500, 10))
        col_points = (
            driving_lane_coeff[0] * row_points**2
            + driving_lane_coeff[1] * row_points
            + driving_lane_coeff[2]
        )
        fitted_lane_points_world = np.vstack([col_points, row_points, np.zeros(300)]).T

        # Transform lane points to bird's eye view
        fitted_lane_points_bev = self.coord_transform.world_to_bird(
            fitted_lane_points_world
        )
        fitted_lane_points_bev = fitted_lane_points_bev[
            (fitted_lane_points_bev >= 750).all(axis=1)
        ]

        for coord in fitted_lane_points_bev.astype(int):
            cv2.circle(img, coord, 2, (255, 255, 0), -1)

        return img


def main(args=None):
    """
    Main function to start the LaneDetectionNode.

    Keyword Arguments:
        args -- Launch arguments (default: {None})
    """
    rclpy.init(args=args)
    node = LaneDetectionNode()

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
