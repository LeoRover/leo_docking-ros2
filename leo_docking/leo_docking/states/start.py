from threading import Event

import rclpy
import smach

from aruco_opencv_msgs.msg import ArucoDetection
from typing import List, Optional


class StartState(smach.State):
    """State checking if the board is seen by the rover"""

    def __init__(
        self,
        node: rclpy.node.Node,
        outcomes: Optional[List[str]] = None,
        input_keys: Optional[List[str]] = None,
        timeout: int = 3,
        name: str = "Start",
    ):
        if outcomes is None:
            outcomes = ["board_not_found", "board_found", "preempted"]
        if input_keys is None:
            input_keys = ["action_goal", "action_feedback", "action_result"]
        super().__init__(outcomes, input_keys)
        self.node = node
        self.timeout = self.node.declare_parameter("start_state/timeout", timeout).value

        self.board_flag: Event = Event()

        self.state_log_name = name

        self.board_sub = self.node.create_subscription(
            ArucoDetection, "aruco_detections", self.board_callback, qos_profile=1
        )
        self.reset_state()

    def reset_state(self):
        self.board_id = None
        self.board_flag.clear()

    def board_callback(self, data: ArucoDetection):
        """Function called every time there is new ArucoDetection message published on the topic.
        Checks if the rover can see board with the desired id (passed as action goal).
        """
        # if board is not seen yet and there are any boards detected
        if len(data.boards) != 0 and not self.board_flag.is_set():
            # look for the desired board
            for board in data.boards:
                if board.board_name == self.board_id:
                    self.board_flag.set()
                    break

    def service_preempt(self):
        """Function called when the state catches preemption request.
        Removes all the publishers and subscribers of the state.
        """
        self.node.get_logger().warning(f"Preemption request handling for '{self.state_log_name}' state.")
        return super().service_preempt()

    def execute(self, ud):
        """Main state method, executed automatically on state entered"""
        self.reset_state()

        self.board_id = ud.action_goal.board_id
        self.node.get_logger().info(
            f"Waiting for board detection. Required board_id: {self.board_id}"
        )

        rate = self.node.create_rate(10)
        time_start = self.node.get_clock().now()
        while not self.board_flag.is_set():
            if self.preempt_requested():
                self.service_preempt()
                ud.action_result.result = f"{self.state_log_name}: state preempted."
                return "preempted"
            secs, _ = (self.node.get_clock().now() - time_start).seconds_nanoseconds()
            if secs > self.timeout:
                self.node.get_logger().error("Didn't find a board. Docking failed.")
                ud.action_result.result = (
                    f"{self.state_log_name}: didn't find a board. Docking failed."
                )
                return "board_not_found"

            rate.sleep()

        ud.action_feedback.current_state = (
            f"{self.state_log_name}: board with id: {self.board_id} found. "
            f"Proceeding to 'Check Area' state."
        )
        return "board_found"
