from __future__ import annotations
from typing import Optional
from threading import Lock
from queue import Queue
import math
import numpy as np

import rospy

from aruco_opencv_msgs.msg import MarkerPose
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from sensor_msgs.msg import JointState


from leo_docking.utils import (
    angle_done_from_odom,
    distance_done_from_odom,
    translate,
    normalize_marker,
)
from .reach_docking_pose import BaseDockingState


class Dock(BaseDockingState):
    """State performing final phase of the docking - reaching the base.
    Drives the rover forward unitl one of three condition is satisfied:
    - rover is close enough to the marker located on the docking base
    - the voltage on the topic with battery data is higher than the average collected before docking
    (if the average was low enough at the beggining)
    - the effort on the wheel motors is high enough - the rover is pushing against the base
    """

    def __init__(
        self,
        timeout=3,
        speed_min=0.05,
        speed_max=0.2,
        route_min=0.05,
        route_max=0.8,
        bias_min=0.01,
        bias_max=0.10,
        bias_speed_min=0.05,
        bias_speed_max=0.3,
        epsilon=0.25,
        debug=False,
        battery_diff=0.2,
        max_bat_average=11.0,
        battery_averaging_time=1.0,
        effort_summary_threshold=2.0,
        effort_buffer_size=10,
        name="Dock",
    ):
        if rospy.has_param("~dock/timeout"):
            timeout = rospy.get_param("~dock/timeout", timeout)
        else:
            timeout = rospy.get_param("~timeout", timeout)

        if rospy.has_param("~dock/epsilon"):
            epsilon = rospy.get_param("~dock/epsilon", epsilon)
        else:
            epsilon = rospy.get_param("~epsilon", epsilon)

        debug = rospy.get_param("~debug", debug)

        speed_min = rospy.get_param("~dock/speed_min", speed_min)
        speed_max = rospy.get_param("~dock/speed_max", speed_max)
        route_min = rospy.get_param("~dock/distance_min", route_min)
        route_max = rospy.get_param("~dock/distance_max", route_max)

        self.buff_size = rospy.get_param("~effort_buffer_size", effort_buffer_size)

        super().__init__(
            timeout=timeout,
            speed_min=speed_min,
            speed_max=speed_max,
            route_min=route_min,
            route_max=route_max,
            epsilon=epsilon,
            debug=debug,
            name=name,
        )

        self.battery_lock: Lock = Lock()
        self.battery_diff = rospy.get_param("~battery_diff", battery_diff)
        self.battery_threshold = rospy.get_param(
            "~max_battery_average", max_bat_average
        )

        self.collection_time = rospy.get_param(
            "~battery_averaging_time", battery_averaging_time
        )

        self.effort_lock: Lock = Lock()
        self.effort_threshold = rospy.get_param(
            "~effort_threshold", effort_summary_threshold
        )

        self.bias_min = rospy.get_param("~dock/bias_min", bias_min)
        self.bias_max = rospy.get_param("~dock/bias_max", bias_max)
        self.bias_left = 0.0
        self.bias_done = 0.0
        self.bias_speed_min = rospy.get_param(
            "~dock/bias_speed_min", bias_speed_min
        )
        self.bias_speed_max = rospy.get_param(
            "~dock/bias_speed_max", bias_speed_max
        )
        self.bias_direction = 0.0

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
            if rospy.Time.now() < self.end_time:
                self.acc_data += data.data
                self.counter += 1
            elif not self.battery_reference:
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
        # self.charging = False
        # self.battery_reference = None
        # self.acc_data = 0.0
        # self.counter = 0
        # self.effort_stop = False
        # self.effort_buf = Queue(maxsize=self.buff_size)

        rospy.loginfo("Waiting for motors effort and battery voltage to drop.")
        rospy.sleep(rospy.Duration(secs=2.0))

        with self.battery_lock:
            self.end_time = rospy.Time.now() + rospy.Duration(secs=self.collection_time)

        self.battery_sub = rospy.Subscriber(
            "firmware/battery", Float32, self.battery_callback, queue_size=1
        )

        # waiting for the end of colleting data
        rospy.loginfo("Measuring battery data...")
        while rospy.Time.now() < self.end_time:
            rospy.sleep(rospy.Duration(secs=0.2))

        rospy.loginfo("Batery voltage average level calculated. Performing docking.")

        self.joint_state_sub = rospy.Subscriber(
            "joint_states", JointState, self.effort_callback, queue_size=1
        )

        msg = Twist()

        while True:
            with self.battery_lock:
                if self.charging:
                    rospy.loginfo(
                        f"Docking stopped. Condition: baterry charging detected."
                    )
                    break

            with self.effort_lock:
                if self.effort_stop:
                    rospy.loginfo(
                        f"Docking stopped. Condition: wheel motors effort rise detected."
                    )
                    break

            with self.route_lock:
                if self.route_done + self.epsilon >= self.route_left:
                    rospy.loginfo(
                        f"Docking stopped. Condition: distance to marker reached."
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
            rospy.sleep(rospy.Duration(secs=0.1))

        self.vel_pub.publish(Twist())
        self.battery_sub.unregister()
        self.joint_state_sub.unregister()
        return None

    def calculate_route_left(self, marker: MarkerPose) -> None:
        normalized_marker = normalize_marker(marker)
        self.route_left = math.sqrt(
            normalized_marker.p.x() ** 2 + normalized_marker.p.y() ** 2
        )

        # calculating the correction for the docking point
        dock_bias = math.atan2(normalized_marker.p.y(), normalized_marker.p.x())
        self.movement_direction = 1.0 if normalized_marker.p.x() >= 0.0 else -1.0
        self.bias_direction = 1.0 if dock_bias > 0.0 else -1.0
        self.bias_left = math.fabs(dock_bias)

    def calculate_route_done(self, odom_reference: Odometry, current_odom: Odometry):
        self.route_done = distance_done_from_odom(odom_reference, current_odom)
        self.bias_done = angle_done_from_odom(odom_reference, current_odom)

    def service_preempt(self):
        self.wheel_odom_sub.unregister()
        self.marker_sub.unregister()
        self.battery_sub.unregister()
        self.joint_state_sub.unregister()
        return super().service_preempt()
