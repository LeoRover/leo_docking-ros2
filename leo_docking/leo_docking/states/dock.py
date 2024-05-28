from __future__ import annotations
from typing import Optional
from threading import Lock
from queue import Queue
import math
import numpy as np
import time
import rclpy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from rclpy.time import Time

from aruco_opencv_msgs.msg import BoardPose
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from sensor_msgs.msg import JointState


from leo_docking.utils import (
    angle_done_from_odom,
    distance_done_from_odom,
    translate,
    normalize_board,
)
from leo_docking.states.reach_docking_pose import BaseDockingState


class Dock(BaseDockingState):
    """State performing final phase of the docking - reaching the base.
    Drives the rover forward unitl one of three condition is satisfied:
    - rover is close enough to the board located on the docking base
    - the voltage on the topic with battery data is higher than the average collected before docking
    (if the average was low enough at the beggining)
    - the effort on the wheel motors is high enough - the rover is pushing against the base
    """

    def __init__(
        self,
        node: rclpy.node.Node,
        timeout: int = 3,
        speed_min: float = 0.05,
        speed_max: float = 0.2,
        route_min: float = 0.05,
        route_max: float = 0.8,
        bias_min: float = 0.01,
        bias_max: float = 0.10,
        bias_speed_min: float = 0.05,
        bias_speed_max: float = 0.3,
        epsilon: float = 0.25,
        debug: bool = False,
        battery_diff: float = 0.2,
        max_bat_average: float = 11.0,
        battery_averaging_time: float = 1.0,
        effort_summary_threshold: float = 2.0,
        effort_buffer_size: float = 10,
        motor_cd_time: float = 2.0,
        name:str = "Dock",
    ):
        self.node = node
        self.timeout = self.node.declare_parameter("dock/timeout", timeout).value
        epsilon = self.node.declare_parameter("dock/epsilon", epsilon).value
        debug = self.node.get_parameter("debug").value
        speed_min = self.node.declare_parameter("dock/speed_min", speed_min).value
        speed_max = self.node.declare_parameter("dock/speed_max", speed_max).value
        route_min = self.node.declare_parameter("dock/distance_min", route_min).value
        route_max = self.node.declare_parameter("dock/distance_max", route_max).value

        self.buff_size = self.node.declare_parameter("effort_buffer_size", effort_buffer_size).value

        super().__init__(
            node,
            timeout=timeout,
            speed_min=speed_min,
            speed_max=speed_max,
            route_min=route_min,
            route_max=route_max,
            epsilon=epsilon,
            debug=debug,
            name=name,
        )
        self.end_time = self.node.get_clock().now()
        self.battery_lock: Lock = Lock()
        self.battery_diff = self.node.declare_parameter("~battery_diff", battery_diff).value
        self.battery_threshold = self.node.declare_parameter("max_battery_average", max_bat_average).value
        self.collection_time = self.node.declare_parameter("battery_averaging_time", battery_averaging_time).value

        self.motor_cd_time = self.node.declare_parameter("motor_cd_time", motor_cd_time).value
        self.effort_lock: Lock = Lock()
        self.effort_threshold = self.node.declare_parameter("~effort_threshold", effort_summary_threshold).value
        self.bias_min = self.node.declare_parameter("~dock/bias_min", bias_min).value
        self.bias_max = self.node.declare_parameter("~dock/bias_max", bias_max).value
        self.bias_left = 0.0
        self.bias_done = 0.0
        self.bias_speed_min = self.node.declare_parameter("~dock/bias_speed_min", bias_speed_min).value
        self.bias_speed_max = self.node.declare_parameter("~dock/bias_speed_max", bias_speed_max).value
        self.bias_direction = 0.0

        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, durability=QoSDurabilityPolicy.VOLATILE, depth=1)
        self.battery_sub = self.node.create_subscription(
            Float32, "firmware/battery", self.battery_callback, qos_profile=qos
        )
        self.joint_state_sub = self.node.create_subscription(
            JointState, "joint_states", self.effort_callback, qos_profile=qos
        )

        self.reset_state()

    def reset_state(self):
        self.bias_direction = 0.0
        self.bias_left = 0.0
        self.bias_done = 0.0
        self.charging = False
        self.battery_reference = None
        self.acc_data = 0.0
        self.counter = 0
        self.effort_stop = False
        self.effort_buf = Queue(maxsize=self.buff_size)

        return super().reset_state()

    def battery_callback(self, data: Float32) -> None:
        """Function called every time, there is new message published on the battery topic.
        Calculates the battery average threshold and checks the battery stop condition.
        """
        with self.battery_lock:
            if self.node.get_clock().now() < self.end_time:
                self.acc_data += data.data
                self.counter += 1
            elif not self.battery_reference and self.counter != 0:
                self.battery_reference = self.acc_data / float(self.counter)
            else:
                # battery average level too high to notice difference
                if self.battery_reference > self.battery_threshold:
                    return

                if data.data > self.battery_reference + self.battery_diff:
                    self.charging = True

    def effort_callback(self, data: JointState) -> None:
        """Function called every time, there is new JointState message published on the topic.
        Calculates the sum of efforts on the wheel motors, and checks the wheel effort stop
        condition.
        """
        with self.effort_lock:
            effort_sum = 0.0
            for effort in data.effort:
                effort_sum += effort

            if self.effort_buf.full():
                buffer_to_np = np.array(list(self.effort_buf.queue))
                avr = np.mean(buffer_to_np)

                if avr >= self.effort_threshold:
                    self.effort_stop = True

                self.effort_buf.get_nowait()

            self.effort_buf.put_nowait(effort_sum)

    def movement_loop(self) -> Optional[str]:
        """Function performing rover movement; invoked in the "execute" method of the state."""

        self.node.get_logger().info("Waiting for motors effort and battery voltage to drop.")
        time.sleep(self.motor_cd_time)

        with self.battery_lock:
            self.end_time = self.node.get_clock().now() + Time(seconds=self.collection_time)

        # waiting for the end of colleting data
        self.node.get_logger().info("Measuring battery data...")
        time.sleep(self.collection_time)
        self.node.get_logger().info("Batery voltage average level calculated. Performing docking.")

        msg = Twist()

        while True:
            with self.battery_lock:
                if self.charging:
                    self.node.get_logger().info(
                        f"Docking stopped. Condition: battery charging detected."
                    )
                    break

            with self.effort_lock:
                if self.effort_stop:
                    self.node.get_logger().info(
                        f"Docking stopped. Condition: wheel motors effort rise detected."
                    )
                    break

            with self.route_lock:
                if self.route_done + self.epsilon >= self.route_left:
                    self.node.get_logger().info(
                        f"Docking stopped. Condition: distance to board reached."
                    )
                    break

                msg.linear.x = self.movement_direction * translate(
                    self.route_left - (self.route_done + self.epsilon),
                    self.route_min,
                    self.route_max,
                    self.speed_min,
                    self.speed_max,
                )

                msg.angular.z = self.bias_direction * translate(
                    self.bias_left - self.bias_done,
                    self.bias_min,
                    self.bias_max,
                    self.bias_speed_min,
                    self.bias_speed_max,
                )

                if self.preempt_requested():
                    self.service_preempt()
                    return "preempted"

                self.vel_pub.publish(msg)
            time.sleep(0.1)

        self.vel_pub.publish(Twist())
        return None

    def calculate_route_left(self, board: BoardPose) -> None:
        normalized_board = normalize_board(board)
        self.route_left = math.sqrt(
            normalized_board.p.x() ** 2 + normalized_board.p.y() ** 2
        )

        # calculating the correction for the docking point
        dock_bias = math.atan2(normalized_board.p.y(), normalized_board.p.x())
        self.movement_direction = 1.0 if normalized_board.p.x() >= 0.0 else -1.0
        self.bias_direction = 1.0 if dock_bias > 0.0 else -1.0
        self.bias_left = math.fabs(dock_bias)

    def calculate_route_done(self, odom_reference: Odometry, current_odom: Odometry):
        self.route_done = distance_done_from_odom(odom_reference, current_odom)
        self.bias_done = angle_done_from_odom(odom_reference, current_odom)

    def service_preempt(self):
        return super().service_preempt()
