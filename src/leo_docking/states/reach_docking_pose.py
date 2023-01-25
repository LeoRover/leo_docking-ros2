from __future__ import annotations
from typing import Optional
from threading import Lock, Event
import math

import rospy
import tf2_ros
import smach

from aruco_opencv_msgs.msg import MarkerDetection, MarkerPose
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

import PyKDL

from leo_docking.utils import (
    get_location_points_from_marker,
    angle_done_from_odom,
    distance_done_from_odom,
    visualize_position,
    translate,
    normalize_marker,
)


class BaseDockingState(smach.State):
    """Base class for the sequence states of the sub-state machine responsible
    for getting the rover in the docking position."""

    def __init__(
        self,
        outcomes=["succeeded", "odometry_not_working", "marker_lost", "preempted"],
        input_keys=["action_goal", "action_feedback", "action_result"],
        timeout=3.0,
        speed_min=0.1,
        speed_max=0.4,
        route_min=0.05,
        route_max=1.0,
        epsilon=0.01,
        docking_point_distance=0.6,
        debug=False,
        angle=True,
        name="",
    ):
        super().__init__(outcomes, input_keys)
        self.timeout = timeout
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.route_min = route_min
        self.route_max = route_max
        self.epsilon = epsilon
        self.docking_point_distance = docking_point_distance

        self.marker_flag: Event = Event()
        self.odom_flag: Event = Event()
        self.route_lock: Lock = Lock()
        self.odom_lock: Lock = Lock()

        self.odom_reference: Odometry = None
        self.current_odom: Odometry = None

        self.route_left = 0.0
        self.route_done = -1.0
        self.movement_direction = 0.0
        self.angle = angle

        self.marker_id = None

        self.state_log_name = name

        # debug variables
        self.debug = debug
        self.seq = 0
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        self.reset_state()

    def reset_state(self):
        self.marker_flag.clear()
        self.odom_flag.clear()
        self.odom_reference = None
        self.current_odom = None

        self.route_left = 0.0
        self.route_done = -1.0
        self.movement_direction = 0.0

        self.marker_id = None

        self.seq = 0

    def calculate_route_left(self, marker: MarkerPose) -> None:
        """Function calculating route left (either angle left to target or linear distance)
        from the docking pose calculated from the current marker detection.
        Saves the calculation result in a class variable "route_left".

        Args:
            marker: (MarkerPose) the current detected pose of the marker
        """
        raise NotImplementedError()

    def movement_loop(self) -> Optional[str]:
        """Function performing rover movement; invoked in the "execute" method of the state."""
        msg = Twist()
        rate = rospy.Rate(10)

        while True:
            with self.route_lock:
                if self.route_done + self.epsilon >= self.route_left:
                    break

                speed = self.movement_direction * translate(
                    self.route_left - self.route_done,
                    self.route_min,
                    self.route_max,
                    self.speed_min,
                    self.speed_max,
                )

                if self.angle:
                    msg.angular.z = speed
                else:
                    msg.linear.x = speed

                if self.preempt_requested():
                    self.service_preempt()
                    return "preempted"

                self.vel_pub.publish(msg)
            rate.sleep()

        self.vel_pub.publish(Twist())
        return None

    def calculate_route_done(
        self, odom_reference: Odometry, current_odom: Odometry
    ) -> None:
        """Function calculating route done (either angle, or distance) from the odometry message
        saved in marker callback (odom_reference), to the current position.
        Saves the calculated route in a class variable "route_done".

        Args:
            odom_reference: the odometry message saved as reference position in marker callback
            current_odom: the newest odometry message received by the state (current position)
        """
        if self.angle:
            self.route_done = angle_done_from_odom(odom_reference, current_odom)
        else:
            self.route_done = distance_done_from_odom(odom_reference, current_odom)

    def marker_callback(self, data: MarkerDetection) -> None:
        """Function called every time, there is new MarkerDetection message published on the topic.
        Calculates the route left from the detected marker, set's odometry reference, and
        (if specified), sends debug tf's so you can visualize calculated target position in rviz.
        """
        if len(data.markers) == 0:
            return

        marker: MarkerPose
        for marker in data.markers:
            if marker.marker_id == self.marker_id:
                if not self.marker_flag.is_set():
                    self.marker_flag.set()

                if self.debug:
                    (
                        docking_point,
                        docking_orientation,
                    ) = get_location_points_from_marker(
                        marker, distance=self.docking_point_distance
                    )
                    visualize_position(
                        docking_point,
                        docking_orientation,
                        "base_link",
                        "docking_point",
                        self.seq,
                        self.tf_broadcaster,
                    )

                    marker_normalized = normalize_marker(marker)
                    visualize_position(
                        marker_normalized.p,
                        marker_normalized.M.GetQuaternion(),
                        "base_link",
                        "normalized_marker",
                        self.seq,
                        self.tf_broadcaster,
                    )

                    self.seq += 1

                with self.route_lock:
                    self.calculate_route_left(marker)

                with self.odom_lock:
                    self.odom_reference = self.current_odom

                break

    def wheel_odom_callback(self, data: Odometry) -> None:
        """Function called every time, there is new Odometry message published on the topic.
        Calculates the route done from the referance message saved in marker callback,
        and the current odometry pose.
        """
        if not self.odom_flag.is_set():
            self.odom_flag.set()

        self.current_odom = data

        with self.odom_lock:
            if self.odom_reference:
                self.calculate_route_done(self.odom_reference, self.current_odom)

    def execute(self, ud):
        """Main state method, executed automatically on state entered"""
        self.reset_state()

        self.marker_id = ud.action_goal.marker_id

        self.marker_sub = rospy.Subscriber(
            "marker_detections", MarkerDetection, self.marker_callback, queue_size=1
        )

        self.wheel_odom_sub = rospy.Subscriber(
            "wheel_odom_with_covariance", Odometry, self.wheel_odom_callback
        )

        if not self.marker_flag.wait(self.timeout):
            rospy.logerr(f"Marker (id: {self.marker_id}) lost. Docking failed.")
            self.marker_sub.unregister()
            self.wheel_odom_sub.unregister()
            ud.action_result = f"{self.state_log_name}: Marker lost. Docking failed."
            # if preempt request came during waiting for the marker detection
            # it won't be handled if the marker is not seen, but the request will stay and
            # will be handled in the next call to the state machine, so there is need to
            # call service_preempt method here
            super().service_preempt()
            return "marker_lost"

        if not self.odom_flag.wait(self.timeout):
            self.marker_sub.unregister()
            self.wheel_odom_sub.unregister()
            rospy.logerr("Didn't get wheel odometry message. Docking failed.")
            ud.action_result.result = (
                f"{self.state_log_name}: wheel odometry not working. Docking failed."
            )
            # if preempt request came during waiting for an odometry message
            # it won't be handled if the odometry doesn't work, but the request will stay and
            # will be handled in the next call to the state machine, so there is need to
            # call service_preempt method here
            super().service_preempt()
            return "odometry_not_working"

        self.vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)

        outcome = self.movement_loop()
        if outcome:
            ud.action_result.result = f"{self.state_log_name}: state preempted."
            return "preempted"

        self.wheel_odom_sub.unregister()
        self.marker_sub.unregister()
        self.vel_pub.unregister()

        ud.action_feedback.current_state = (
            f"'Reach Docking Point': sequence completed. "
            f"Proceeding to 'Dock' state."
        )
        if self.state_log_name == "Dock":
            ud.action_result.result = "docking succeeded. Rover docked."
        return "succeeded"

    def service_preempt(self):
        """Function called when the state catches preemption request.
        Removes all the publishers and subscribers of the state.
        """
        rospy.logwarn(f"Preemption request handling for {self.state_log_name} state")
        self.vel_pub.publish(Twist())
        self.vel_pub.unregister()
        self.wheel_odom_sub.unregister()
        self.marker_sub.unregister()
        return super().service_preempt()


class RotateToDockingPoint(BaseDockingState):
    """The first state of the sequence state machine getting rover to docking position;
    responsible for rotating the rover towards the target point.
    (default: 0.6m in straight line from docking base).
    """

    def __init__(
        self,
        timeout=3.0,
        speed_min=0.1,
        speed_max=0.4,
        angle_min=0.05,
        angle_max=1.0,
        epsilon=0.01,
        docking_point_distance=0.6,
        debug=True,
        angular=True,
        name="Rotate To Docking Point",
    ):
        if rospy.has_param("~rotate_to_docking_point/timeout"):
            timeout = rospy.get_param("~rotate_to_docking_point/timeout", timeout)
        else:
            timeout = rospy.get_param("~timeout", timeout)

        if rospy.has_param("~rotate_to_docking_point/epsilon"):
            epsilon = rospy.get_param("~rotate_to_docking_point/epsilon", epsilon)
        else:
            epsilon = rospy.get_param("~epsilon", epsilon)

        docking_point_distance = rospy.get_param(
            "~docking_point_distance", docking_point_distance
        )
        debug = rospy.get_param("~debug", debug)

        speed_min = rospy.get_param("~rotate_to_docking_point/speed_min", speed_min)
        speed_max = rospy.get_param("~rotate_to_docking_point/speed_max", speed_max)
        angle_min = rospy.get_param("~rotate_to_docking_point/angle_min", angle_min)
        angle_max = rospy.get_param("~rotate_to_docking_point/angle_max", angle_max)

        super().__init__(
            timeout=timeout,
            speed_min=speed_min,
            speed_max=speed_max,
            route_min=angle_min,
            route_max=angle_max,
            epsilon=epsilon,
            docking_point_distance=docking_point_distance,
            debug=debug,
            angle=angular,
            name=name,
        )

    def calculate_route_left(self, marker: MarkerPose):
        docking_point, _ = get_location_points_from_marker(
            marker, distance=self.docking_point_distance
        )

        if math.sqrt(docking_point.y() ** 2 + docking_point.x() ** 2) < 0.1:
            self.route_left = 0.0
        else:
            angle = math.atan2(docking_point.y(), docking_point.x())
            self.movement_direction = 1 if angle >= 0 else -1
            self.route_left = math.fabs(angle)


class ReachDockingPoint(BaseDockingState):
    """The second state of the sequence state machine getting rover to docking position;
    responsible for driving the rover to the target docking point
    (default: 0.6m in straight line from docking base).
    Performs linear and angular movement, as it fixes it's orientation
    to be always looking on the docking point.
    """

    def __init__(
        self,
        timeout=3,
        speed_min=0.1,
        speed_max=0.3,
        dist_min=0.05,
        dist_max=1.5,
        bias_min=0.01,
        bias_max=0.10,
        bias_speed_min=0.05,
        bias_speed_max=0.3,
        epsilon=0.01,
        docking_point_distance=0.6,
        debug=True,
        angular=False,
        name="Reach Docking Point",
    ):
        if rospy.has_param("~reach_docking_point/timeout"):
            timeout = rospy.get_param("~reach_docking_point/timeout", timeout)
        else:
            timeout = rospy.get_param("~timeout", timeout)

        if rospy.has_param("~reach_docking_point/epsilon"):
            epsilon = rospy.get_param("~reach_docking_point/epsilon", epsilon)
        else:
            epsilon = rospy.get_param("~epsilon", epsilon)

        docking_point_distance = rospy.get_param(
            "~docking_point_distance", docking_point_distance
        )
        debug = rospy.get_param("~debug", debug)

        speed_min = rospy.get_param("~reach_docking_point/speed_min", speed_min)
        speed_max = rospy.get_param("~reach_docking_point/speed_max", speed_max)
        dist_min = rospy.get_param("~reach_docking_point/distance_min", dist_min)
        dist_max = rospy.get_param("~reach_docking_point/distance_max", dist_max)

        super().__init__(
            timeout=timeout,
            speed_min=speed_min,
            speed_max=speed_max,
            route_min=dist_min,
            route_max=dist_max,
            epsilon=epsilon,
            docking_point_distance=docking_point_distance,
            debug=debug,
            angle=angular,
            name=name,
        )

        self.bias_min = rospy.get_param("~reach_docking_point/bias_min", bias_min)
        self.bias_max = rospy.get_param("~reach_docking_point/bias_max", bias_max)
        self.bias_left = 0.0
        self.bias_done = 0.0
        self.bias_speed_min = rospy.get_param(
            "~reach_docking_point/bias_speed_min", bias_speed_min
        )
        self.bias_speed_max = rospy.get_param(
            "~reach_docking_point/bias_speed_max", bias_speed_max
        )
        self.bias_direction = 0.0

        self.reset_state()

    def reset_state(self):
        self.bias_done = 0.0
        self.bias_left = 0.0
        self.bias_direction = 0.0
        return super().reset_state()

    def movement_loop(self) -> Optional[str]:
        msg = Twist()
        rate = rospy.Rate(10)

        while True:
            with self.route_lock:
                if self.route_done + self.epsilon >= self.route_left:
                    break

                msg.linear.x = self.movement_direction * translate(
                    self.route_left - self.route_done,
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
            rate.sleep()

        self.vel_pub.publish(Twist())
        return None

    def calculate_route_left(self, marker: MarkerPose):
        docking_point, _ = get_location_points_from_marker(
            marker, distance=self.docking_point_distance
        )

        self.route_left = math.sqrt(docking_point.x() ** 2 + docking_point.y() ** 2)
        self.movement_direction = 1.0 if docking_point.x() >= 0 else -1.0

        # calculating the correction for the docking point
        dock_bias = math.atan2(docking_point.y(), docking_point.x())
        self.bias_direction = 1.0 if dock_bias > 0.0 else -1.0
        self.bias_left = math.fabs(dock_bias)

    def calculate_route_done(self, odom_reference: Odometry, current_odom: Odometry):
        self.route_done = distance_done_from_odom(odom_reference, current_odom)
        self.bias_done = angle_done_from_odom(odom_reference, current_odom)


class ReachDockingOrientation(BaseDockingState):
    """The third state of the sequence state machine getting rover to docking position;
    responsible for rotating the rover toward marker on the docking base - a position where the
    rover needs just to drive forward to reach the base.
    """

    def __init__(
        self,
        timeout=3,
        speed_min=0.1,
        speed_max=0.4,
        angle_min=0.05,
        angle_max=1,
        epsilon=0.01,
        docking_point_distance=0.6,
        debug=True,
        angular=True,
        name="Reach Dockin Orientation",
    ):
        if rospy.has_param("~reach_docking_orientation/timeout"):
            timeout = rospy.get_param("~reach_docking_orientation/timeout", timeout)
        else:
            timeout = rospy.get_param("~timeout", timeout)

        if rospy.has_param("~reach_docking_orientation/epsilon"):
            epsilon = rospy.get_param("~reach_docking_orientation/epsilon", epsilon)
        else:
            epsilon = rospy.get_param("~epsilon", epsilon)

        docking_point_distance = rospy.get_param(
            "~docking_point_distance", docking_point_distance
        )
        debug = rospy.get_param("~debug", debug)

        speed_min = rospy.get_param(
            "~reach_docking_orientation/speed_min", speed_min
        )
        speed_max = rospy.get_param(
            "~reach_docking_orientation/speed_max", speed_max
        )
        angle_min = rospy.get_param(
            "~reach_docking_orientation/angle_min", angle_min
        )
        angle_max = rospy.get_param(
            "~reach_docking_orientation/angle_max", angle_max
        )

        super().__init__(
            timeout=timeout,
            speed_min=speed_min,
            speed_max=speed_max,
            route_min=angle_min,
            route_max=angle_max,
            epsilon=epsilon,
            docking_point_distance=docking_point_distance,
            debug=debug,
            angle=angular,
            name=name,
        )

    def calculate_route_left(self, marker: MarkerPose):
        _, docking_orientation = get_location_points_from_marker(
            marker, distance=self.docking_point_distance
        )

        rot = PyKDL.Rotation.Quaternion(*docking_orientation)
        angle = rot.GetRPY()[2]

        self.movement_direction = 1.0 if angle > 0 else -1.0
        self.route_left = math.fabs(angle)
