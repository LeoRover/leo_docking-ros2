import math
from threading import Event, Lock
from typing import Optional, List

import rclpy
import smach

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

import PyKDL

from leo_docking.utils import (
    translate,
    angle_done_from_odom,
    distance_done_from_odom,
)
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy


class BaseDockAreaState(smach.State):
    """Base class for the sequence states of the sub-state machine responsible
    for getting the rover in the area where the docking is possible."""

    def __init__(
        self,
        node: rclpy.node.Node,
        outcomes: Optional[List[str]] = None,
        input_keys: Optional[List[str]] = None,
        output_keys: Optional[List[str]] = None,
        timeout: int = 3,
        speed_min: float = 0.1,
        speed_max: float = 0.4,
        route_min: float = 0.0,
        route_max: float = 1.05,
        epsilon: float = 0.1,
        angle: bool = True,
        name: str = "",
    ):
        if outcomes is None:
            outcomes = ["succeeded", "odometry_not_working", "preempted"]
        if input_keys is None:
            input_keys = ["target_pose", "action_feedback", "action_result"]
        if output_keys is None:
            output_keys = ["target_pose"]
        super().__init__(outcomes, input_keys, output_keys)
        self.node = node
        self.timeout = timeout
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.route_min = route_min
        self.route_max = route_max
        self.epsilon = epsilon
        self.angle = angle

        self.output_len = len(output_keys)
        self.odom_flag: Event = Event()
        self.route_lock: Lock = Lock()

        self.state_log_name = name

        self.route_done = 0.0
        self.odom_reference = None

        self.publish_cmd_vel_cb = None

        self.reset_state()

    def reset_state(self):
        self.odom_reference = None
        self.odom_flag.clear()
        self.route_done = 0.0

    def calculate_route_done(
        self, odom_reference: Odometry, current_odom: Odometry, angle: bool = True
    ) -> None:
        """Function calculating route done (either angle, or distance)
        from the begining of the state (first received odometry message), to the current position.
        Saves the calculated route in a class variable "route_done".

        Args:
            odom_reference: first odometry message received by the state (start position)
            current_odom: the newest odometry message received by the state (current position)
            angle: flag specifying wheter the route is an angle or a distance
        """
        if angle:
            self.route_done = angle_done_from_odom(odom_reference, current_odom)
        else:
            self.route_done = distance_done_from_odom(odom_reference, current_odom)

    def calculate_route_left(self, target_pose: PyKDL.Frame) -> float:
        """Function calculating route left (either angle left to target or linear distance)
        from target pose.

        Args:
            target_pose: (PyKDL.Frame) target pose of the rover at the end of the docking phase
                         (sub-state machine)
        Returns:
            route_left: (float) calculated route that is left to traverse
        """
        raise NotImplementedError()

    def movement_loop(self, route_left: float, angle: bool = True) -> Optional[str]:
        """Function performing rover movement; invoked in the "execute" method of the state.

        Args:
            route_left: route (angle / distance) the rover has to ride
            angle: flag specifying wheter it will be movement in x axis, or rotation around z axis.
        """
        direction = 1.0 if route_left > 0 else -1.0
        route_left = math.fabs(route_left)
        msg = Twist()

        rate = self.node.create_rate(10)

        while True:
            with self.route_lock:
                if self.route_done + self.epsilon >= route_left:
                    break

                speed = direction * translate(
                    route_left - self.route_done,
                    self.route_min,
                    self.route_max,
                    self.speed_min,
                    self.speed_max,
                )

                if angle:
                    msg.angular.z = speed
                else:
                    msg.linear.x = speed

                if self.preempt_requested():
                    self.service_preempt()
                    return "preempted"

                self.publish_cmd_vel_cb(msg)
            rate.sleep()

        self.publish_cmd_vel_cb(Twist())

        return None

    def execute(self, ud):
        """Main state method, executed automatically on state entered"""
        self.reset_state()

        rate = self.node.create_rate(10)
        time_start = self.node.get_clock().now()
        while not self.odom_flag.is_set():
            if self.preempt_requested():
                self.service_preempt()
                ud.action_result.result = f"{self.state_log_name}: state preempted."
                return "preempted"
            secs = (self.node.get_clock().now() - time_start).nanoseconds//1e9
            if secs > self.timeout:
                self.node.get_logger().error("Didn't get wheel odometry message. Docking failed.")
                ud.action_result.result = (
                    f"{self.state_log_name}: No odom data. Docking failed."
                )
                return "odometry_not_working"

            rate.sleep()

        target_pose: PyKDL.Frame = ud.target_pose
        # calculating route left
        route_left = self.calculate_route_left(target_pose)
        # moving the rover
        outcome = self.movement_loop(route_left, self.angle)
        if outcome:
            ud.action_result.result = f"{self.state_log_name}: state preempted."
            return "preempted"

        # passing the data to next state
        if self.output_len > 0:
            ud.target_pose = target_pose

        ud.action_feedback.current_state = (
            f"'Reach Docking Area`: sequence completed. "
            f"Proceeding to 'Check Area' state."
        )
        return "succeeded"

    def wheel_odom_cb(self, data: Odometry) -> None:
        """Function called every time, there is new Odometry message published on the topic.
        Calculates the route done from the first message that it got, and the current one.
        """
        if not self.odom_flag.is_set():
            self.odom_flag.set()
            if not self.odom_reference:
                self.odom_reference = data

        with self.route_lock:
            self.calculate_route_done(self.odom_reference, data, self.angle)

    def service_preempt(self):
        """Function called when the state catches preemption request.
        Removes all the publishers and subscribers of the state.
        """
        self.node.get_logger().warn(f"Preemption request handling for {self.state_log_name} state")
        self.publish_cmd_vel_cb(Twist())
        return super().service_preempt()


class RotateToDockArea(BaseDockAreaState):
    """The first state of the sequence state machine getting rover to docking area;
    responsible for rotating the rover towards target point in the area (default: 2m in straight
    line from docking base)."""

    def __init__(
        self,
        node: rclpy.node.Node,
        timeout: int = 3,
        speed_min: float = 0.1,
        speed_max: float = 0.4,
        angle_min: float = 0.05,
        angle_max: float = 1.05,
        epsilon: float = 0.1,
        angle: bool = True,
        name: str = "Rotate Towards Area",
    ):
        timeout = node.declare_parameter("rotate_to_dock_area/timeout", timeout).value
        epsilon = node.declare_parameter("rotate_to_dock_area/epsilon", epsilon).value
        speed_min = node.declare_parameter("rotate_to_dock_area/speed_min", speed_min).value
        speed_max = node.declare_parameter("rotate_to_dock_area/speed_max", speed_max).value
        angle_min = node.declare_parameter("rotate_to_dock_area/angle_min", angle_min).value
        angle_max = node.declare_parameter("rotate_to_dock_area/angle_max", angle_max).value

        super().__init__(
            node,
            timeout=timeout,
            speed_min=speed_min,
            speed_max=speed_max,
            route_min=angle_min,
            route_max=angle_max,
            epsilon=epsilon,
            angle=angle,
            name=name,
        )

    def calculate_route_left(self, target_pose: PyKDL.Frame) -> float:
        position: PyKDL.Vector = target_pose.p
        route_left = math.atan2(position.y(), position.x())

        return route_left


class RideToDockArea(BaseDockAreaState):
    """The second state of the sequence state machine getting rover to docking area;
    responsible for driving the rover to the target point in the area (default: 2m in straight line
    from docking base)"""

    def __init__(
        self,
        node: rclpy.node.Node,
        timeout: int = 3,
        speed_min: float = 0.05,
        speed_max: float = 0.4,
        distance_min: float = 0.1,
        distance_max: float = 0.5,
        epsilon: float = 0.1,
        angle=False,
        name="Ride To Area",
    ):
        timeout = node.declare_parameter("ride_to_dock_area/timeout", timeout).value
        epsilon = node.declare_parameter("ride_to_dock_area/epsilon", epsilon).value
        speed_min = node.declare_parameter("ride_to_dock_area/speed_min", speed_min).value
        speed_max = node.declare_parameter("ride_to_dock_area/speed_max", speed_max).value
        distance_min = node.declare_parameter("ride_to_dock_area/distance_min", distance_min).value
        distance_max = node.declare_parameter("ride_to_dock_area/distance_max", distance_max).value

        super().__init__(
            node,
            timeout=timeout,
            speed_min=speed_min,
            speed_max=speed_max,
            route_min=distance_min,
            route_max=distance_max,
            epsilon=epsilon,
            angle=angle,
            name=name,
        )

    def calculate_route_left(self, target_pose: PyKDL.Frame) -> float:
        position: PyKDL.Vector = target_pose.p
        route_left = math.sqrt(position.x() ** 2 + position.y() ** 2)

        return route_left


class RotateToBoard(BaseDockAreaState):
    """The third state of the sequence state machine getting rover to docking area;
    responsible for rotating the rover toward board on the docking base"""

    def __init__(
        self,
        node: rclpy.node.Node,
        output_keys: List[str] = None,
        timeout: int = 3,
        speed_min: float = 0.1,
        speed_max: float = 0.4,
        angle_min: float = 0.05,
        angle_max: float = 1.05,
        epsilon: float = 0.1,
        angle: bool = True,
        name: str = "Rotate Towards Board",
    ):
        if output_keys is None:
            output_keys = []
        timeout = node.declare_parameter("rotate_to_board/timeout", timeout).value
        epsilon = node.declare_parameter("rotate_to_board/epsilon", epsilon).value
        speed_min = node.declare_parameter("rotate_to_board/speed_min", speed_min).value
        speed_max = node.declare_parameter("rotate_to_board/speed_max", speed_max).value
        angle_min = node.declare_parameter("rotate_to_board/angle_min", angle_min).value
        angle_max = node.declare_parameter("rotate_to_board/angle_max", angle_max).value

        super().__init__(
            node,
            output_keys=output_keys,
            timeout=timeout,
            speed_min=speed_min,
            speed_max=speed_max,
            route_min=angle_min,
            route_max=angle_max,
            epsilon=epsilon,
            angle=angle,
            name=name,
        )

    def calculate_route_left(self, target_pose: PyKDL.Frame) -> float:
        position: PyKDL.Vector = target_pose.p
        # calculating rotation done in the first state of sequence
        angle_done = math.atan2(position.y(), position.x())
        # rotating target pose by -angle, so the target orientation is looking at board again
        # (initial target pose is in the `base_link` frame)
        target_pose.M.DoRotZ(-angle_done)
        route_left = target_pose.M.GetRPY()[2]

        return route_left
