import math
from threading import Event, Lock

import rospy
import smach
import PyKDL

from aruco_opencv_msgs.msg import MarkerDetection, MarkerPose
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from leo_docking.utils import (
    translate,
    calculate_threshold_distances,
    get_location_points_from_marker,
    angle_done_from_odom,
    distance_done_from_odom,
)


class CheckArea(smach.State):
    """State responsible for checking the rover position regarding docking area
    (area where the docking is possible) threshold, and providing the target pose,
    when rover is outside the area."""

    def __init__(
        self,
        outcomes=["docking_area", "outside_docking_area", "marker_lost"],
        output_keys=["target_pose"],
        threshold_angle=0.26,  # 15 degrees
        docking_distance=2.0,
        timeout=3.0,
    ):
        super().__init__(outcomes=outcomes, output_keys=output_keys)

        self.threshold_angle = rospy.get_param("~threshold_angle", threshold_angle)
        self.docking_distance = rospy.get_param("~docking_distance", docking_distance)

        if rospy.has_param("~check_area/timeout"):
            self.timeout = rospy.get_param("~check_area/timeout", timeout)
        else:
            self.timeout = rospy.get_param("~timeout", timeout)

        self.marker_flag = Event()

    def marker_callback(self, data: MarkerDetection):
        """Function called everu time, there is new MarkerDetection message published on the topic.
        Saves the detected marker's position for further calculations.
        """
        if len(data.markers) != 0:
            self.marker: MarkerPose = data.markers[0]
            if not self.marker_flag.is_set():
                self.marker_flag.set()

    def check_threshold(self, dist_x: float, dist_y: float) -> bool:
        """Function checking if the rover is in the docking area threshold.

        Args:
            dist_x: distance of rover's projection on the marker's x axis to the marker
            dist_y: distance of rover's position to the markers' x axis
        Returns:
            True if rover is in the docking area, False otherwise
        """
        max_value = math.tan(self.threshold_angle) * dist_x

        return dist_y <= max_value

    def execute(self, user_data):
        """Main state method invoked on state entered.
        Checks rover position and eventually calculates target pose of the rover.
        """
        self.marker_flag.clear()
        marker_sub = rospy.Subscriber(
            "marker_detections", MarkerDetection, self.marker_callback, queue_size=1
        )
        self.marker = None
        rospy.loginfo("Waiting for marker detection.")

        if not self.marker_flag.wait(self.timeout):
            rospy.logerr("Marker lost. Docking failed.")
            return "marker_lost"

        # calculating the length of distances needed for threshold checking
        x_dist, y_dist = calculate_threshold_distances(self.marker)

        if self.check_threshold(x_dist, y_dist):
            marker_sub.unregister()
            return "docking_area"

        # getting target pose
        point, orientation = get_location_points_from_marker(
            self.marker, self.docking_distance
        )

        target_pose = PyKDL.Frame(
            PyKDL.Rotation.Quaternion(*orientation),
            point,
        )

        marker_sub.unregister()

        # passing calculated data to next states
        user_data.target_pose = target_pose

        return "outside_docking_area"


class BaseDockAreaState(smach.State):
    """Base class for the sequence states of the sub-state machine responsible
    for getting the rover in the area where the docking is possible."""

    def __init__(
        self,
        outcomes=["succeeded", "odometry_not_working"],
        input_keys=["target_pose"],
        output_keys=["target_pose"],
        timeout=3.0,
        speed_min=0.1,
        speed_max=0.4,
        route_min=0.0,
        route_max=1.05,
        epsilon=0.1,
        angle=True,
    ):
        super().__init__(outcomes, input_keys, output_keys)

        self.timeout = timeout
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.route_min = route_min
        self.route_max = route_max
        self.epsilon = epsilon
        self.angle = angle

        self.output_len = len(output_keys)
        self.route_done = 0.0
        self.odom_reference: Odometry = None
        self.odom_flag: Event = Event()
        self.route_lock: Lock = Lock()

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
            target_pose: (PyKDL.Frame) target pose of the rover at the end of the docking phase (sub-state machine)
        Returns:
            route_left: (float) calculated route that is left to traverse
        """
        raise NotImplementedError()

    def movement_loop(self, route_left: float, angle: bool = True) -> None:
        """Function performing rover movement; invoked in the "execute" method of the state.

        Args:
            route_left: route (angle / distance) the rover has to ride
            angle: flag specifying wheter it will be movement in x axis, or rotation around z axis.
        """
        direction = 1.0 if route_left > 0 else -1.0
        route_left = math.fabs(route_left)
        msg = Twist()

        r = rospy.Rate(10)

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

                self.cmd_vel_pub.publish(msg)
            r.sleep()

        self.cmd_vel_pub.publish(Twist())

    def execute(self, user_data):
        """Main state method, executed automatically on state entered"""
        self.odom_flag.clear()
        self.odom_reference = None
        self.route_done = 0.0

        self.wheel_odom_sub = rospy.Subscriber(
            "wheel_odom_with_covariance", Odometry, self.wheel_odom_callback
        )

        # waiting for odometry message
        if not self.odom_flag.wait(self.timeout):
            self.wheel_odom_sub.unregister()
            rospy.logerr("Didn't get wheel odometry message. Docking failed.")
            return "odometry_not_working"

        self.cmd_vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)

        target_pose: PyKDL.Frame = user_data.target_pose
        # calculating route left
        route_left = self.calculate_route_left(target_pose)
        # moving the rover
        self.movement_loop(route_left, self.angle)

        self.cmd_vel_pub.unregister()
        self.wheel_odom_sub.unregister()

        # passing the data to next state
        if self.output_len > 0:
            user_data.target_pose = target_pose

        return "succeeded"

    def wheel_odom_callback(self, data: Odometry) -> None:
        """Function called every time, there is new Odometry message published on the topic.
        Calculates the route done from the first message that it got, and the current one.
        """
        if not self.odom_flag.is_set():
            self.odom_flag.set()
            if not self.odom_reference:
                self.odom_reference = data

        with self.route_lock:
            self.calculate_route_done(self.odom_reference, data, self.angle)


class RotateToDockArea(BaseDockAreaState):
    """The first state of the sequence state machine getting rover to docking area;
    responsible for rotating the rover towards target point in the area (default: 2m in straight line from docking base)."""

    def __init__(
        self,
        timeout=3,
        speed_min=0.1,
        speed_max=0.4,
        angle_min=0.05,
        angle_max=1.05,
        epsilon=0.1,
        angle=True,
    ):
        if rospy.has_param("~rotate_to_dock_area/timeout"):
            timeout = rospy.get_param("~rotate_to_dock_area/timeout", timeout)
        else:
            timeout = rospy.get_param("~timeout", timeout)

        if rospy.has_param("~rotate_to_dock_area/epsilon"):
            epsilon = rospy.get_param("~rotate_to_dock_area/epsilon", epsilon)
        else:
            epsilon = rospy.get_param("~epsilon", epsilon)

        speed_min = rospy.get_param("~rotate_to_dock_area/speed_min", speed_min)
        speed_max = rospy.get_param("~rotate_to_dock_area/speed_max", speed_max)
        angle_min = rospy.get_param("~rotate_to_dock_area/angle_min", angle_min)
        angle_max = rospy.get_param("~rotate_to_dock_area/angle_max", angle_max)

        super().__init__(
            timeout=timeout,
            speed_min=speed_min,
            speed_max=speed_max,
            route_min=angle_min,
            route_max=angle_max,
            epsilon=epsilon,
            angle=angle,
        )

    def calculate_route_left(self, target_pose: PyKDL.Frame) -> float:
        position: PyKDL.Vector = target_pose.p
        route_left = math.atan2(position.y(), position.x())

        return route_left


class RideToDockArea(BaseDockAreaState):
    """The second state of the sequence state machine getting rover to docking area;
    responsible for driving the rover to the target point in the area (default: 2m in straight line from docking base)"""

    def __init__(
        self,
        timeout=3,
        speed_min=0.05,
        speed_max=0.4,
        distance_min=0.1,
        distance_max=0.5,
        epsilon=0.1,
        angle=False,
    ):

        if rospy.has_param("~ride_to_dock_area/timeout"):
            timeout = rospy.get_param("~ride_to_dock_area/timeout", timeout)
        else:
            timeout = rospy.get_param("~timeout", timeout)

        if rospy.has_param("~ride_to_dock_area/epsilon"):
            epsilon = rospy.get_param("~ride_to_dock_area/epsilon", epsilon)
        else:
            epsilon = rospy.get_param("~epsilon", epsilon)

        speed_min = rospy.get_param("~ride_to_dock_area/speed_min", speed_min)
        speed_max = rospy.get_param("~ride_to_dock_area/speed_max", speed_max)
        distance_min = rospy.get_param("~ride_to_dock_area/distance_min", distance_min)
        distance_max = rospy.get_param("~ride_to_dock_area/distance_max", distance_max)

        super().__init__(
            timeout=timeout,
            speed_min=speed_min,
            speed_max=speed_max,
            route_min=distance_min,
            route_max=distance_max,
            epsilon=epsilon,
            angle=angle,
        )

    def calculate_route_left(self, target_pose: PyKDL.Frame) -> float:
        position: PyKDL.Vector = target_pose.p
        route_left = math.sqrt(position.x() ** 2 + position.y() ** 2)

        return route_left


class RotateToMarker(BaseDockAreaState):
    """The third state of the sequence state machine getting rover to docking area;
    responsible for rotating the rover toward marker on the docking base"""

    def __init__(
        self,
        output_keys=[],
        timeout=3,
        speed_min=0.1,
        speed_max=0.4,
        angle_min=0.05,
        angle_max=1.05,
        epsilon=0.1,
        angle=True,
    ):
        if rospy.has_param("~rotate_to_marker/timeout"):
            timeout = rospy.get_param("~rotate_to_marker/timeout", timeout)
        else:
            timeout = rospy.get_param("~timeout", timeout)

        if rospy.has_param("~rotate_to_marker/epsilon"):
            epsilon = rospy.get_param("~rotate_to_marker/epsilon", epsilon)
        else:
            epsilon = rospy.get_param("~epsilon", epsilon)

        speed_min = rospy.get_param("~rotate_to_marker/speed_min", speed_min)
        speed_max = rospy.get_param("~rotate_to_marker/speed_max", speed_max)
        angle_min = rospy.get_param("~rotate_to_marker/angle_min", angle_min)
        angle_max = rospy.get_param("~rotate_to_marker/angle_max", angle_max)
        
        super().__init__(
            output_keys=output_keys,
            timeout=timeout,
            speed_min=speed_min,
            speed_max=speed_max,
            route_min=angle_min,
            route_max=angle_max,
            epsilon=epsilon,
            angle=angle,
        )

    def calculate_route_left(self, target_pose: PyKDL.Frame) -> float:
        position: PyKDL.Vector = target_pose.p
        # calculating rotation done in the first state of sequence
        angle_done = math.atan2(position.y(), position.x())
        # rotating target pose by -angle, so the target orientation is looking at marker again (initial target pose is in the `base_link` frame)
        target_pose.M.DoRotZ(-angle_done)
        route_left = target_pose.M.GetRPY()[2]

        return route_left
