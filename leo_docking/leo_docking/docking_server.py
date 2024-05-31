#!/usr/bin/env python3
import rclpy
import tf2_ros
from aruco_opencv_msgs.msg import ArucoDetection
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import JointState
from smach_ros import ActionServerWrapper, IntrospectionServer

from leo_docking.states.docking_state_machine import DockingStateMachine

from leo_docking.state_machine_params import StateMachineParams
from std_msgs.msg import Float32

from leo_docking.utils import visualize_position
from leo_docking_msgs.action import PerformDocking


class DockingServer(rclpy.node.Node):
    def __init__(self):
        super().__init__("docking_server")
        self._state_machine_params = StateMachineParams(self)
        self._state_machine = DockingStateMachine(
            self._state_machine_params, self.get_logger(), self._publish_cmd_vel_cb, self._debug_visualizations_cb
        )
        self._init_ros()

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
            "leo_docking", self._state_machine.state_machine, "/LEO_DOCKING"
        )

    def _init_ros(self):
        sub_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT, durability=QoSDurabilityPolicy.VOLATILE, depth=1
        )
        self.create_subscription(ArucoDetection, "/aruco_detections", self._aruco_detection_cb, qos_profile=sub_qos)
        self.create_subscription(Odometry, "/odometry/filtered/local", self._wheel_odom_cb, qos_profile=sub_qos)
        self.create_subscription(Float32, "/firmware/battery", self._battery_cb, qos_profile=sub_qos)
        self.create_subscription(JointState, "/joint_states", self._effort_cb, qos_profile=sub_qos)

        pub_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.VOLATILE, depth=1
        )
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel/nav2", qos_profile=pub_qos)

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(node=self)

    def _aruco_detection_cb(self, msg: ArucoDetection):
        self._state_machine.states["Start"]["state"].aruco_detection_cb(msg)
        self._state_machine.states["Check Area"]["state"].aruco_detection_cb(msg)
        self._state_machine.states["Rotate To Docking Point"]["state"].aruco_detection_cb(msg)
        self._state_machine.states["Reach Docking Point"]["state"].aruco_detection_cb(msg)
        self._state_machine.states["Reach Docking Point Orientation"]["state"].aruco_detection_cb(msg)
        self._state_machine.states["Dock"]["state"].aruco_detection_cb(msg)

    def _wheel_odom_cb(self, msg: Odometry):
        self._state_machine.states["Rotate To Dock Area"]["state"].wheel_odom_cb(msg)
        self._state_machine.states["Ride To Dock Area"]["state"].wheel_odom_cb(msg)
        self._state_machine.states["Rotate To Board"]["state"].wheel_odom_cb(msg)
        self._state_machine.states["Rotate To Docking Point"]["state"].wheel_odom_cb(msg)
        self._state_machine.states["Reach Docking Point"]["state"].wheel_odom_cb(msg)
        self._state_machine.states["Reach Docking Point Orientation"]["state"].wheel_odom_cb(msg)
        self._state_machine.states["Dock"]["state"].wheel_odom_cb(msg)

    def _battery_cb(self, msg: Float32):
        self._state_machine.states["Dock"]["state"].battery_cb(msg)

    def _effort_cb(self, msg: JointState):
        self._state_machine.states["Dock"]["state"].effort_cb(msg)

    def _publish_cmd_vel_cb(self, msg: Twist):
        self.cmd_vel_pub.publish(msg)

    def _debug_visualizations_cb(self, docking_point, docking_orientation, board_normalized):
        visualize_position(
            docking_point,
            docking_orientation,
            "base_link",
            "docking_point",
            self.tf_broadcaster,
            self.get_clock().now().to_msg(),
        )
        visualize_position(
            board_normalized.p,
            board_normalized.M.GetQuaternion(),
            "base_link",
            "normalized_board",
            self.tf_broadcaster,
            self.get_clock().now().to_msg(),
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
