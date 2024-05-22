#!/usr/bin/env python3
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor

from leo_docking_msgs.action import PerformDockingAction
from rclpy.exceptions import ROSInterruptException


class DockingClient(rclpy.node.Node):

    def __init__(self):
        super().__init__('leo_docking_client')
        self.board_id = self.declare_parameter("board_id", "1").value
        self.timeout = self.declare_parameter("timeout", 3.0).value
        self.check_interval = self.declare_parameter("check_interval", 2.0).value
        self.rate = self.create_rate(10)
        self.goal_done = False

        self.action_client = ActionClient(
            self, PerformDockingAction, "leo_docking_action_server"
        )
        if not self.action_client.wait_for_server(self.timeout):
            self.get_logger().error(f"Server 'leo_docking_action_server' not active.")
            raise RuntimeError(f"Server 'leo_docking_action_server' not active.")

    def dock(self):
        goal = PerformDockingAction.Goal(board_id=self.board_id)
        future = self.action_client.send_goal_async(goal, feedback_callback=self.feedback_callback)
        future.add_done_callback(self.done_callback)
        self.goal_done = False
        while rclpy.ok() and not self.goal_done:
            try:
                self.rate.sleep()
            except ROSInterruptException:
                pass

    def done_callback(self, result):
        self.get_logger().info(f"Done callback: {result}")
        self.goal_done = True

    def feedback_callback(self, feedback):
        self.get_logger().info(f"Feedback: {feedback.progress}")


if __name__ == "__main__":
    rclpy.init()
    node = DockingClient()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    executor.shutdown()
    node.destroy_node()
    rclpy.try_shutdown()
