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
from typing import Callable, Any

from smach import StateMachine, Sequence

from leo_docking.states.start import StartState
from leo_docking.states.check_area import CheckArea
from leo_docking.states.reach_docking_area import (
    RideToDockArea,
    RotateToDockArea,
    RotateToBoard,
)
from leo_docking.states.reach_docking_pose import (
    RotateToDockingPoint,
    ReachDockingPoint,
    ReachDockingOrientation,
)
from leo_docking.states.dock import Dock

from leo_docking.state_machine_params import StateMachineParams
from leo_docking.utils import LoggerProto


class DockingStateMachine:
    def __init__(
        self,
        state_machine_params: StateMachineParams,
        logger: LoggerProto,
        publish_cmd_vel_cb: Callable,
        debug_visualizations_cb: Callable,
    ):
        self.params = state_machine_params
        self.states: dict[str, dict[str, Any]] = {
            "Start": {
                "state": StartState(self.params.start_params, logger),
                "transitions": {
                    "board_not_found": "DOCKING FAILED",
                    "board_found": "Check Area",
                    "preempted": "DOCKING PREEMPTED",
                },
                "remapping": {
                    "action_goal": "action_goal",
                    "action_feedback": "action_feedback",
                    "action_result": "action_result",
                },
            },
            "Check Area": {
                "state": CheckArea(self.params.check_area_params, logger),
                "transitions": {
                    "board_lost": "DOCKING FAILED",
                    "docking_area": "Reach Docking Point",
                    "outside_docking_area": "Reach Docking Area",
                    "preempted": "DOCKING PREEMPTED",
                },
                "remapping": {
                    "target_pose": "docking_area_data",
                    "board_id": "action_goal",
                    "action_feedback": "action_feedback",
                    "action_result": "action_result",
                },
            },
            "Rotate To Dock Area": {
                "state": RotateToDockArea(
                    self.params.rotate_to_dock_area_params, publish_cmd_vel_cb, logger
                ),
                "remapping": {
                    "target_pose": "docking_area_data",
                    "action_feedback": "action_feedback",
                    "action_result": "action_result",
                },
            },
            "Ride To Dock Area": {
                "state": RideToDockArea(
                    self.params.ride_to_dock_area_params, publish_cmd_vel_cb, logger
                ),
                "remapping": {
                    "target_pose": "docking_area_data",
                    "action_feedback": "action_feedback",
                    "action_result": "action_result",
                },
            },
            "Rotate To Board": {
                "state": RotateToBoard(
                    self.params.rotate_to_board_params, publish_cmd_vel_cb, logger
                ),
                "remapping": {
                    "target_pose": "docking_area_data",
                    "action_feedback": "action_feedback",
                    "action_result": "action_result",
                },
            },
            "Rotate To Docking Point": {
                "state": RotateToDockingPoint(
                    self.params.global_params,
                    self.params.rotate_to_docking_point_params,
                    publish_cmd_vel_cb,
                    logger,
                    debug_visualizations_cb,
                ),
            },
            "Reach Docking Point": {
                "state": ReachDockingPoint(
                    self.params.global_params,
                    self.params.reach_docking_point_params,
                    publish_cmd_vel_cb,
                    logger,
                    debug_visualizations_cb,
                ),
            },
            "Reach Docking Point Orientation": {
                "state": ReachDockingOrientation(
                    self.params.global_params,
                    self.params.reach_docking_orientation_params,
                    publish_cmd_vel_cb,
                    logger,
                    debug_visualizations_cb,
                ),
            },
            "Dock": {
                "state": Dock(
                    self.params.global_params,
                    self.params.dock_params,
                    publish_cmd_vel_cb,
                    logger,
                    debug_visualizations_cb,
                ),
                "transitions": {
                    "succeeded": "ROVER DOCKED",
                    "odometry_not_working": "DOCKING FAILED",
                    "board_lost": "DOCKING FAILED",
                    "preempted": "DOCKING PREEMPTED",
                },
                "remapping": {
                    "action_goal": "action_goal",
                    "action_feedback": "action_feedback",
                    "action_result": "action_result",
                },
            },
        }
        self.sequences = {
            "Reach Docking Area": {
                "sequence": self.reach_docking_area_sequence(),
                "transitions": {
                    "succeeded": "Check Area",
                    "odometry_not_working": "DOCKING FAILED",
                    "preempted": "DOCKING PREEMPTED",
                },
                "remapping": {
                    "docking_area_data": "docking_area_data",
                    "action_feedback": "action_feedback",
                    "action_result": "action_result",
                },
            },
            "Reach Docking Point": {
                "sequence": self.reach_docking_pose_sequence(),
                "transitions": {
                    "succeeded": "Dock",
                    "odometry_not_working": "DOCKING FAILED",
                    "board_lost": "DOCKING FAILED",
                    "preempted": "DOCKING PREEMPTED",
                },
                "remapping": {
                    "action_goal": "action_goal",
                    "action_feedback": "action_feedback",
                    "action_result": "action_result",
                },
            },
        }

        self.state_machine = StateMachine(
            outcomes=["ROVER DOCKED", "DOCKING FAILED", "DOCKING PREEMPTED"],
            input_keys=["action_goal", "action_feedback", "action_result"],
        )

        with self.state_machine:
            StateMachine.add(
                "Start",
                self.states["Start"]["state"],
                self.states["Start"]["transitions"],
                self.states["Start"]["remapping"],
            )

            StateMachine.add(
                "Check Area",
                self.states["Check Area"]["state"],
                self.states["Check Area"]["transitions"],
                self.states["Check Area"]["remapping"],
            )

            StateMachine.add(
                "Reach Docking Area",
                self.sequences["Reach Docking Area"]["sequence"],
                self.sequences["Reach Docking Area"]["transitions"],
                self.sequences["Reach Docking Area"]["remapping"],
            )

            StateMachine.add(
                "Reach Docking Point",
                self.sequences["Reach Docking Point"]["sequence"],
                self.sequences["Reach Docking Point"]["transitions"],
                self.sequences["Reach Docking Point"]["remapping"],
            )

            StateMachine.add(
                "Dock",
                self.states["Dock"]["state"],
                self.states["Dock"]["transitions"],
                self.states["Dock"]["remapping"],
            )

    def reach_docking_area_sequence(self):
        reach_docking_area = Sequence(
            outcomes=["succeeded", "odometry_not_working", "preempted"],
            connector_outcome="succeeded",
            input_keys=["docking_area_data", "action_feedback", "action_result"],
        )

        with reach_docking_area:
            Sequence.add(
                label="Rotate To Dock Area",
                state=self.states["Rotate To Dock Area"]["state"],
                remapping=self.states["Rotate To Dock Area"]["remapping"],
            )
            Sequence.add(
                label="Ride To Dock Area",
                state=self.states["Ride To Dock Area"]["state"],
                remapping=self.states["Ride To Dock Area"]["remapping"],
            )
            Sequence.add(
                label="Rotate To Board",
                state=self.states["Rotate To Board"]["state"],
                remapping=self.states["Rotate To Board"]["remapping"],
            )
        return reach_docking_area

    def reach_docking_pose_sequence(self):
        reach_docking_pose = Sequence(
            outcomes=["succeeded", "odometry_not_working", "board_lost", "preempted"],
            connector_outcome="succeeded",
            input_keys=["action_goal", "action_feedback", "action_result"],
        )

        with reach_docking_pose:
            Sequence.add(
                "Rotate To Docking Point",
                self.states["Rotate To Docking Point"]["state"],
            )
            Sequence.add(
                "Reach Docking Point", self.states["Reach Docking Point"]["state"]
            )
            Sequence.add(
                "Reach Docking Orientation",
                self.states["Reach Docking Point Orientation"]["state"],
            ),
        return reach_docking_pose
