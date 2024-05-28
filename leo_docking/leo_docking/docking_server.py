#!/usr/bin/env python3
import rclpy
import smach
import smach_ros
from rclpy.executors import MultiThreadedExecutor
from smach_ros import ActionServerWrapper
from leo_docking_msgs.action import PerformDocking

from leo_docking.states.start import StartState
from leo_docking.states.check_area import CheckArea
from leo_docking.states.reach_docking_area import RideToDockArea, RotateToDockArea, RotateToBoard
from leo_docking.states.reach_docking_pose import (
    RotateToDockingPoint,
    ReachDockingPoint,
    ReachDockingOrientation,
)
from leo_docking.states.dock import Dock


def create_state_machine(node) -> smach.StateMachine:
    sm = smach.StateMachine(
        outcomes=["ROVER DOCKED", "DOCKING FAILED", "DOCKING PREEMPTED"],
        input_keys=["action_goal", "action_feedback", "action_result"],
    )

    with sm:
        smach.StateMachine.add(
            "Start",
            StartState(node=node, timeout=5.0),
            transitions={
                "board_not_found": "DOCKING FAILED",
                "board_found": "Check Area",
                "preempted": "DOCKING PREEMPTED",
            },
            remapping={
                "action_goal": "action_goal",
                "action_feedback": "action_feedback",
                "action_result": "action_result",
            },
        )

        smach.StateMachine.add(
            "Check Area",
            CheckArea(node=node, timeout=5, threshold_angle=0.17),
            transitions={
                "board_lost": "DOCKING FAILED",
                "docking_area": "Reach Docking Point",
                "outside_docking_area": "Reach Docking Area",
                "preempted": "DOCKING PREEMPTED",
            },
            remapping={
                "target_pose": "docking_area_data",
                "board_id": "action_goal",
                "action_feedback": "action_feedback",
                "action_result": "action_result",
            },
        )

        reach_docking_area = smach.Sequence(
            outcomes=["succeeded", "odometry_not_working", "preempted"],
            connector_outcome="succeeded",
            input_keys=["docking_area_data", "action_feedback", "action_result"],
        )

        with reach_docking_area:
            smach.Sequence.add(
                "Rotate To Dock Area",
                RotateToDockArea(node=node, timeout=2.0),
                remapping={
                    "target_pose": "docking_area_data",
                    "action_feedback": "action_feedback",
                    "action_result": "action_result",
                },
            )
            smach.Sequence.add(
                "Ride To Area",
                RideToDockArea(node=node, timeout=2.0),
                remapping={
                    "target_pose": "docking_area_data",
                    "action_feedback": "action_feedback",
                    "action_result": "action_result",
                },
            )
            smach.Sequence.add(
                "Rotate To Board",
                RotateToBoard(node=node, timeout=2.0),
                remapping={
                    "target_pose": "docking_area_data",
                    "action_feedback": "action_feedback",
                    "action_result": "action_result",
                },
            )

        smach.StateMachine.add(
            "Reach Docking Area",
            reach_docking_area,
            transitions={
                "succeeded": "Check Area",
                "odometry_not_working": "DOCKING FAILED",
                "preempted": "DOCKING PREEMPTED",
            },
            remapping={
                "docking_area_data": "docking_area_data",
                "action_feedback": "action_feedback",
                "action_result": "action_result",
            },
        )

        reach_docking_pose = smach.Sequence(
            outcomes=["succeeded", "odometry_not_working", "board_lost", "preempted"],
            connector_outcome="succeeded",
            input_keys=["action_goal", "action_feedback", "action_result"],
        )

        with reach_docking_pose:
            smach.Sequence.add(
                "Rotate To Docking Point",
                RotateToDockingPoint(node=node, timeout=2.0, docking_point_distance=0.8),
            )

            smach.Sequence.add(
                "Reach Docking Point",
                ReachDockingPoint(node=node, timeout=2.0, docking_point_distance=0.8),
            )

            smach.Sequence.add(
                "Reach Docking Orientation",
                ReachDockingOrientation(node=node, timeout=2.0, docking_point_distance=0.8),
            )

        smach.StateMachine.add(
            "Reach Docking Point",
            reach_docking_pose,
            transitions={
                "succeeded": "Dock",
                "odometry_not_working": "DOCKING FAILED",
                "board_lost": "DOCKING FAILED",
                "preempted": "DOCKING PREEMPTED",
            },
            remapping={
                "action_goal": "action_goal",
                "action_feedback": "action_feedback",
                "action_result": "action_result",
            },
        )

        smach.StateMachine.add(
            "Dock",
            Dock(node=node, epsilon=0.05),
            transitions={
                "succeeded": "ROVER DOCKED",
                "odometry_not_working": "DOCKING FAILED",
                "board_lost": "DOCKING FAILED",
                "preempted": "DOCKING PREEMPTED",
            },
            remapping={
                "action_goal": "action_goal",
                "action_feedback": "action_feedback",
                "action_result": "action_result",
            },
        )

    return sm


class DockingServer(rclpy.node.Node):
    def __init__(self):
        super().__init__("docking_server")
        self.declare_parameter("docking_point_distance", 0.6)
        self.declare_parameter("debug", True)
        self.declare_parameter("rotate_to_docking_point/timeout", 3)
        self.declare_parameter("rotate_to_docking_point/epsilon", 0.01)
        self.declare_parameter("rotate_to_docking_point/speed_min", 0.1)
        self.declare_parameter("rotate_to_docking_point/speed_max", 0.4)
        self.declare_parameter("rotate_to_docking_point/angle_min", 0.05)
        self.declare_parameter("rotate_to_docking_point/angle_max", 1.0)
        self._state_machine = create_state_machine(self)

        self._action_server_wrapper = ActionServerWrapper(
            node=self,
            server_name="leo_docking_action_server",
            action_spec=PerformDocking,
            wrapped_container=self._state_machine,
            succeeded_outcomes=["ROVER DOCKED"],
            aborted_outcomes=["DOCKING FAILED"],
            preempted_outcomes=["DOCKING PREEMPTED"],
            goal_key="action_goal",
            feedback_key="action_feedback",
            result_key="action_result",
        )

        # Create and start the introspection server
        self._smach_introspection_server = smach_ros.IntrospectionServer(
            "leo_docking", self._state_machine, "/LEO_DOCKING"
        )

    def start(self):
        self._smach_introspection_server.start()

    def stop(self):
        self._smach_introspection_server.stop()


if __name__ == "__main__":
    rclpy.init()
    node = DockingServer()
    node.start()

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    node.stop()
    executor.shutdown()
    node.destroy_node()
    rclpy.try_shutdown()

