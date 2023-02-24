import math
from threading import Event

import rospy
import smach

from aruco_opencv_msgs.msg import ArucoDetection

import PyKDL

from leo_docking.utils import (
    calculate_threshold_distances,
    get_location_points_from_board,
    visualize_position,
)


class CheckArea(smach.State):
    """State responsible for checking the rover position regarding docking area
    (area where the docking is possible) threshold, and providing the target pose,
    when rover is outside the area."""

    def __init__(
        self,
        outcomes=["docking_area", "outside_docking_area", "board_lost", "preempted"],
        input_keys=["action_goal", "action_feedback", "action_result"],
        output_keys=["target_pose"],
        threshold_angle=0.26,  # 15 degrees
        docking_distance=2.0,
        timeout=3.0,
        name="Check Area",
    ):
        super().__init__(
            outcomes=outcomes, input_keys=input_keys, output_keys=output_keys
        )

        self.threshold_angle = rospy.get_param("~threshold_angle", threshold_angle)
        self.docking_distance = rospy.get_param("~docking_distance", docking_distance)

        if rospy.has_param("~check_area/timeout"):
            self.timeout = rospy.get_param("~check_area/timeout", timeout)
        else:
            self.timeout = rospy.get_param("~timeout", timeout)

        self.board_flag = Event()
        self.state_log_name = name

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
        rospy.logwarn(f"Preemption request handling for '{self.state_log_name}' state.")
        self.board_sub.unregister()
        return super().service_preempt()

    def execute(self, ud):
        """Main state method invoked on state entered.
        Checks rover position and eventually calculates target pose of the rover.
        """
        self.reset_state()

        self.board_id = ud.action_goal.board_id
        self.board_sub = rospy.Subscriber(
            "aruco_detections", ArucoDetection, self.board_callback, queue_size=1
        )
        rospy.loginfo(f"Waiting for board (id: {self.board_id}) detection.")

        rate = rospy.Rate(10)
        time_start = rospy.Time.now()
        while not self.board_flag.is_set():
            if self.preempt_requested():
                self.service_preempt()
                ud.action_result.result = f"{self.state_log_name}: state preempted."
                return "preempted"

            if (rospy.Time.now() - time_start).to_sec() > self.timeout:
                rospy.logerr(f"Board (id: {self.board_id}) lost. Docking failed.")
                ud.action_result.result = (
                    f"{self.state_log_name}: board lost. Docking failed."
                )
                self.board_sub.unregister()
                return "board_lost"

            rate.sleep()

        self.board_sub.unregister()

        # calculating the length of distances needed for threshold checking
        x_dist, y_dist = calculate_threshold_distances(self.board)

        if self.check_threshold(x_dist, y_dist):
            self.board_sub.unregister()
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
