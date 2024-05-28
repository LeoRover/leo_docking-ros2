#!/usr/bin/env python3
import rclpy
from rclpy.executors import SingleThreadedExecutor
from smach_ros import ActionServerWrapper, IntrospectionServer

from leo_docking.states.docking_state_machine import DockingStateMachine

from leo_docking.state_machine_params import StateMachineParams
from leo_docking_msgs.action import PerformDocking


class DockingServer(rclpy.node.Node):
    def __init__(self):
        super().__init__("docking_server")
        self._state_machine_params = StateMachineParams()
        self._state_machine = DockingStateMachine(self._state_machine_params)

        self._action_server_wrapper = ActionServerWrapper(
            node=self,
            server_name="/leo_rover/dock",
            action_spec=PerformDocking,
            wrapped_container=self._state_machine.state_machine,
            succeeded_outcomes=["ROVER DOCKED"],
            aborted_outcomes=["DOCKING FAILED"],
            preempted_outcomes=["DOCKING PREEMPTED"],
            goal_key="action_goal",
            feedback_key="action_feedback",
            result_key="action_result",
        )

        # Create and start the introspection server
        self._smach_introspection_server = IntrospectionServer(
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

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    node.stop()
    executor.shutdown()
    node.destroy_node()
    rclpy.try_shutdown()

