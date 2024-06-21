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

from threading import Event
from time import sleep, time

import smach

from aruco_opencv_msgs.msg import ArucoDetection
from typing import List, Optional

from leo_docking.state_machine_params import StartParams
from leo_docking.utils import LoggerProto


class StartState(smach.State):
    """State checking if the board is seen by the rover"""

    def __init__(
        self,
        start_params: StartParams,
        logger: LoggerProto,
        outcomes: Optional[List[str]] = None,
        input_keys: Optional[List[str]] = None,
        name: str = "Start",
    ):
        outcomes = (
            ["board_not_found", "board_found", "preempted"]
            if outcomes is None
            else outcomes
        )
        input_keys = (
            ["action_goal", "action_feedback", "action_result"]
            if input_keys is None
            else input_keys
        )
        super().__init__(outcomes, input_keys)
        self.params = start_params

        self.board_flag: Event = Event()
        self.board_id = None

        self.state_log_name = name
        self.logger = logger
        self.reset_state()

    def reset_state(self):
        self.board_id = None
        self.board_flag.clear()

    def aruco_detection_cb(self, data: ArucoDetection):
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
        self.logger.warning(
            f"Preemption request handling for '{self.state_log_name}' state."
        )
        return super().service_preempt()

    def execute(self, user_data):
        """Main state method, executed automatically on state entered"""
        self.reset_state()

        self.board_id = user_data.action_goal.board_id
        self.logger.info(
            f"Waiting for board detection. Required board_id: {self.board_id}"
        )

        start_time = time()
        while not self.board_flag.is_set():
            if self.preempt_requested():
                self.service_preempt()
                user_data.action_result.result = (
                    f"{self.state_log_name}: state preempted."
                )
                return "preempted"
            if time() - start_time > self.params.timeout:
                self.logger.error(
                    f"Couldn't find a board in {self.params.timeout} seconds. Docking failed."
                )
                user_data.action_result.result = (
                    f"{self.state_log_name}: couldn't find a board. Docking failed."
                )
                return "board_not_found"

            sleep(0.1)

        user_data.action_feedback.current_state = (
            f"{self.state_log_name}: board with id: {self.board_id} found. "
            f"Proceeding to 'Check Area' state."
        )
        return "board_found"
