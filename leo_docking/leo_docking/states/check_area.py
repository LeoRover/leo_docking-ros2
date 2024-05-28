import math
from threading import Event

import rclpy
import smach

from aruco_opencv_msgs.msg import ArucoDetection

import PyKDL

from leo_docking.utils import (
    calculate_threshold_distances,
    get_location_points_from_board,
    visualize_position,
)
from typing import List, Optional

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy


class CheckArea(smach.State):
    """State responsible for checking the rover position regarding docking area
    (area where the docking is possible) threshold, and providing the target pose,
    when rover is outside the area."""

    def __init__(
        self,
        node: rclpy.node.Node,
        outcomes: Optional[List[str]] = None,
        input_keys: Optional[List[str]] = None,
        output_keys: Optional[List[str]] = None,
        threshold_angle: float = 0.26,  # 15 degrees
        docking_distance: float = 2.0,
        timeout: int = 3,
        name: str = "Check Area",
    ):
        if output_keys is None:
            output_keys = ["target_pose"]
        if input_keys is None:
            input_keys = ["action_goal", "action_feedback", "action_result"]
        if outcomes is None:
            outcomes = ["docking_area", "outside_docking_area", "board_lost", "preempted"]
        super().__init__(
            outcomes=outcomes, input_keys=input_keys, output_keys=output_keys
        )
        self.board_id = None
        self.board = None
        self.node = node

        self.threshold_angle = self.node.declare_parameter("threshold_angle", threshold_angle).value
        self.docking_distance = self.node.declare_parameter("docking_distance", docking_distance).value
        self.timeout = self.node.declare_parameter("timeout", timeout).value

        self.board_flag = Event()
        self.state_log_name = name

        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, durability=QoSDurabilityPolicy.VOLATILE)
        self.board_sub = self.node.create_subscription(
            ArucoDetection, "/aruco_detections", self.board_callback, qos_profile=qos
        )

        self.reset_state()

    def reset_state(self):
        self.board_id = None
        self.board_flag.clear()
        self.board = None

    def board_callback(self, data: ArucoDetection):
        """Function called every time there is new ArucoDetection message published on the topic.
        Saves the detected board's position for further calculations.
        """
        if len(data.boards) != 0 and not self.board_flag.is_set():
            for board in data.boards:
                if board.board_name == self.board_id:
                    self.board = board
                    if not self.board_flag.is_set():
                        self.board_flag.set()
                    break

    def check_threshold(self, dist_x: float, dist_y: float) -> bool:
        """Function checking if the rover is in the docking area threshold.

        Args:
            dist_x: distance of rover's projection on the board's x axis to the board
            dist_y: distance of rover's position to the boards' x axis
        Returns:
            True if rover is in the docking area, False otherwise
        """
        max_value = math.tan(self.threshold_angle) * dist_x

        return dist_y <= max_value

    def service_preempt(self):
        """Function called when the state catches preemption request.
        Removes all the publishers and subscribers of the state.
        """
        self.node.get_logger().warn(f"Preemption request handling for '{self.state_log_name}' state.")
        return super().service_preempt()

    def execute(self, ud):
        """Main state method invoked on state entered.
        Checks rover position and eventually calculates target pose of the rover.
        """
        self.reset_state()

        self.board_id = ud.action_goal.board_id
        self.node.get_logger().info(f"Waiting for board (id: {self.board_id}) detection.")

        rate = self.node.create_rate(10)
        time_start = self.node.get_clock().now()
        while not self.board_flag.is_set():
            if self.preempt_requested():
                self.service_preempt()
                ud.action_result.result = f"{self.state_log_name}: state preempted."
                return "preempted"
            secs = (self.node.get_clock().now() - time_start).nanoseconds//1e9
            if secs > self.timeout:
                self.node.get_logger().error(f"Board (id: {self.board_id}) lost. Docking failed.")
                ud.action_result.result = (
                    f"{self.state_log_name}: board lost. Docking failed."
                )
                return "board_lost"

            rate.sleep()

        # calculating the length of distances needed for threshold checking
        x_dist, y_dist = calculate_threshold_distances(self.board)

        if self.check_threshold(x_dist, y_dist):
            ud.action_feedback.current_state = (
                f"{self.state_log_name}: docking possible from current position. "
                f"Proceeding to 'Reaching Docking Point` sequence."
            )
            return "docking_area"

        # getting target pose
        point, orientation = get_location_points_from_board(
            self.board, self.docking_distance
        )

        target_pose = PyKDL.Frame(PyKDL.Rotation.Quaternion(*orientation), point)

        # passing calculated data to next states
        ud.target_pose = target_pose

        ud.action_feedback.current_state = (
            f"{self.state_log_name}: docking impossible from current position. "
            f"Proceeding to 'Reach Docking Area` sequence."
        )
        return "outside_docking_area"
