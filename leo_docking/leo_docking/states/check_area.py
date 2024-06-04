# Copyright 2023 Fictionlab sp. z o.o.
# Copyright 2024 Karelics Oy
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import math
from threading import Event
from time import sleep, time

import smach

from aruco_opencv_msgs.msg import ArucoDetection

import PyKDL

from leo_docking.utils import (
    calculate_threshold_distances,
    get_location_points_from_board,
    LoggerProto,
)
from typing import List, Optional

from leo_docking.state_machine_params import CheckAreaParams


class CheckArea(smach.State):
    """State responsible for checking the rover position regarding docking area
    (area where the docking is possible) threshold, and providing the target pose,
    when rover is outside the area."""

    def __init__(
        self,
        check_area_params: CheckAreaParams,
        logger: LoggerProto,
        outcomes: Optional[List[str]] = None,
        input_keys: Optional[List[str]] = None,
        output_keys: Optional[List[str]] = None,
        name: str = "Check Area",
    ):
        output_keys = ["target_pose"] if output_keys is None else output_keys
        input_keys = ["action_goal", "action_feedback", "action_result"] if input_keys is None else input_keys
        outcomes = ["docking_area", "outside_docking_area", "board_lost", "preempted"] if outcomes is None else outcomes
        super().__init__(outcomes=outcomes, input_keys=input_keys, output_keys=output_keys)
        self.params = check_area_params
        self.board_id = None
        self.board = None

        self.board_flag = Event()
        self.state_log_name = name

        self.logger = logger
        self.reset_state()

    def reset_state(self):
        self.board_id = None
        self.board_flag.clear()
        self.board = None

    def aruco_detection_cb(self, data: ArucoDetection):
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
        max_value = math.tan(self.params.threshold_angle) * dist_x

        return dist_y <= max_value

    def service_preempt(self):
        """Function called when the state catches preemption request.
        Removes all the publishers and subscribers of the state.
        """
        self.logger.warning(f"Preemption request handling for '{self.state_log_name}' state.")
        return super().service_preempt()

    def execute(self, ud):
        """Main state method invoked on state entered.
        Checks rover position and eventually calculates target pose of the rover.
        """
        self.reset_state()

        self.board_id = ud.action_goal.board_id
        self.logger.info(f"Waiting for board (id: {self.board_id}) detection.")

        start_time = time()
        while not self.board_flag.is_set():
            if self.preempt_requested():
                self.service_preempt()
                ud.action_result.result = f"{self.state_log_name}: state preempted."
                return "preempted"
            if time() - start_time > self.params.timeout:
                self.logger.error(f"Board (id: {self.board_id}) lost. Docking failed.")
                ud.action_result.result = f"{self.state_log_name}: board lost. Docking failed."
                return "board_lost"

            sleep(0.1)

        # calculating the length of distances needed for threshold checking
        x_dist, y_dist = calculate_threshold_distances(self.board)

        if self.check_threshold(x_dist, y_dist):
            ud.action_feedback.current_state = (
                f"{self.state_log_name}: docking possible from current position. "
                f"Proceeding to 'Reaching Docking Point` sequence."
            )
            return "docking_area"

        # getting target pose
        point, orientation = get_location_points_from_board(self.board, self.params.docking_distance)

        target_pose = PyKDL.Frame(PyKDL.Rotation.Quaternion(*orientation), point)

        # passing calculated data to next states
        ud.target_pose = target_pose

        ud.action_feedback.current_state = (
            f"{self.state_log_name}: docking impossible from current position. "
            f"Proceeding to 'Reach Docking Area` sequence."
        )
        return "outside_docking_area"
