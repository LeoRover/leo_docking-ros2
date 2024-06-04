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
from __future__ import annotations
from typing import Optional, Callable
from threading import Lock
from queue import Queue
import math
import numpy as np
from time import sleep, time

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
    LoggerProto,
)
from leo_docking.states.reach_docking_pose import BaseDockingState
from leo_docking.state_machine_params import GlobalParams, DockParams


class Dock(BaseDockingState):
    """State performing final phase of the docking - reaching the base.
    Drives the rover forward until one of three condition is satisfied:
    - rover is close enough to the board located on the docking base
    - the voltage on the topic with battery data is higher than the average collected before docking
    (if the average was low enough at the beginning)
    - the effort on the wheel motors is high enough - the rover is pushing against the base
    """

    def __init__(
        self,
        global_params: GlobalParams,
        params: DockParams,
        publish_cmd_vel_cb: Callable,
        logger: LoggerProto,
        debug_visualizations_cb,
        name: str = "Dock",
    ):
        super().__init__(
            global_params,
            params,
            publish_cmd_vel_cb,
            logger,
            debug_visualizations_cb,
            name=name,
        )
        self.battery_lock: Lock = Lock()
        self.effort_lock: Lock = Lock()
        self.bias_left = 0.0
        self.bias_done = 0.0
        self.bias_direction = 0.0
        self.end_time = 0.0

        self.charging = False
        self.battery_reference = None
        self.acc_data = 0.0
        self.counter = 0

        self.effort_stop = False
        self.effort_buf = Queue(maxsize=self.global_params.effort_buffer_size)
        super().reset_state()

    def reset_state(self):
        self.bias_direction = 0.0
        self.bias_left = 0.0
        self.bias_done = 0.0
        self.charging = False
        self.battery_reference = None
        self.acc_data = 0.0
        self.counter = 0
        self.effort_stop = False
        self.effort_buf = Queue(maxsize=self.global_params.effort_buffer_size)

        return super().reset_state()

    def battery_cb(self, data: Float32) -> None:
        """Function called every time, there is new message published on the battery topic.
        Calculates the battery average threshold and checks the battery stop condition.
        """
        with self.battery_lock:
            if not self.executing:
                return
            if time() < self.end_time:
                self.acc_data += data.data
                self.counter += 1
            elif not self.battery_reference and self.counter != 0:
                self.battery_reference = self.acc_data / float(self.counter)
            else:
                # battery average level too high to notice difference
                if self.battery_reference is None or self.battery_reference > self.global_params.max_battery_average:
                    return

                if data.data > self.battery_reference + self.global_params.battery_diff:
                    self.charging = True

    def effort_cb(self, data: JointState) -> None:
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

                if avr >= self.global_params.effort_threshold:
                    self.effort_stop = True

                self.effort_buf.get_nowait()

            self.effort_buf.put_nowait(effort_sum)

    def movement_loop(self) -> Optional[str]:
        """Function performing rover movement; invoked in the "execute" method of the state."""
        self.logger.info("Waiting for motors effort and battery voltage to drop.")
        sleep(self.global_params.motor_cd_time)

        with self.battery_lock:
            self.end_time = time() + self.global_params.battery_averaging_time

        # waiting for the end of colleting data
        self.logger.info("Measuring battery data...")
        sleep(self.global_params.battery_averaging_time)
        self.logger.info("Batery voltage average level calculated. Performing docking.")

        msg = Twist()

        while True:
            with self.battery_lock:
                if self.charging:
                    self.logger.info(f"Docking stopped. Condition: battery charging detected.")
                    break

            with self.effort_lock:
                if self.effort_stop:
                    self.logger.info(f"Docking stopped. Condition: wheel motors effort rise detected.")
                    break

            with self.route_lock:
                if self.route_done + self.params.epsilon >= self.route_left:
                    self.logger.info(f"Docking stopped. Condition: distance to board reached.")
                    break

                msg.linear.x = self.movement_direction * translate(
                    self.route_left - (self.route_done + self.params.epsilon),
                    self.params.dist_min,
                    self.params.dist_max,
                    self.params.speed_min,
                    self.params.speed_max,
                )

                msg.angular.z = self.bias_direction * translate(
                    self.bias_left - self.bias_done,
                    self.params.bias_min,
                    self.params.bias_max,
                    self.params.bias_speed_min,
                    self.params.bias_speed_max,
                )

                if self.preempt_requested():
                    self.service_preempt()
                    return "preempted"

                self.publish_cmd_vel_cb(msg)
            sleep(0.1)
        self.publish_cmd_vel_cb(Twist())
        return None

    def calculate_route_left(self, board: BoardPose) -> None:
        normalized_board = normalize_board(board)
        self.route_left = math.sqrt(normalized_board.p.x() ** 2 + normalized_board.p.y() ** 2)

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
