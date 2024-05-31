from __future__ import annotations
from typing import Optional, List, Union, Callable
from threading import Lock, Event
import math
from time import sleep, time

import smach

from aruco_opencv_msgs.msg import ArucoDetection, BoardPose
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

import PyKDL

from leo_docking.state_machine_params import (
    GlobalParams,
    RotateToDockingPointParams,
    ReachDockingPointParams,
    ReachDockingOrientationParams,
    DockParams,
)
from leo_docking.utils import (
    get_location_points_from_board,
    angle_done_from_odom,
    distance_done_from_odom,
    translate,
    normalize_board,
    LoggerProto,
)


class BaseDockingState(smach.State):
    """Base class for the sequence states of the sub-state machine responsible
    for getting the rover in the docking position."""

    def __init__(
        self,
        global_params: GlobalParams,
        local_params: Union[
            RotateToDockingPointParams, ReachDockingPointParams, ReachDockingOrientationParams, DockParams
        ],
        publish_cmd_vel_cb: Callable,
        logger: LoggerProto,
        debug_visualizations_cb: Optional[Callable] = None,
        outcomes: Optional[List[str]] = None,
        input_keys: Optional[List[str]] = None,
        angle: bool = True,
        name: str = "",
    ):
        outcomes = ["succeeded", "odometry_not_working", "board_lost", "preempted"] if outcomes is None else outcomes
        input_keys = ["action_goal", "action_feedback", "action_result"] if input_keys is None else input_keys
        super().__init__(outcomes, input_keys)
        self.global_params = global_params
        self.params = local_params
        self.executing = False

        self.board_flag: Event = Event()
        self.odom_flag: Event = Event()
        self.route_lock: Lock = Lock()
        self.odom_lock: Lock = Lock()

        self.odom_reference: Optional[Odometry] = None
        self.current_odom: Optional[Odometry] = None

        self.route_left = 0.0
        self.route_done = -1.0
        self.movement_direction = 0.0
        self.angle = angle

        self.board_id = None

        self.state_log_name = name

        self.publish_cmd_vel_cb = publish_cmd_vel_cb
        self.debug_visualizations_cb = debug_visualizations_cb

        self.logger = logger
        self.reset_state()

    def reset_state(self):
        self.board_flag.clear()
        self.odom_flag.clear()
        self.odom_reference = None
        self.current_odom = None
        self.executing = False

        self.route_left = 0.0
        self.route_done = -1.0
        self.movement_direction = 0.0

        self.board_id = None

    def calculate_route_left(self, board: BoardPose) -> None:
        """Function calculating route left (either angle left to target or linear distance)
        from the docking pose calculated from the current board detection.
        Saves the calculation result in a class variable "route_left".

        Args:
            board: (BoardPose) the current detected pose of the board
        """
        raise NotImplementedError()

    def movement_loop(self) -> Optional[str]:
        """Function performing rover movement; invoked in the "execute" method of the state."""
        msg = Twist()

        self.logger.error("START OF MOVEMENT LOOP")
        while True:
            with self.route_lock:
                if self.route_done + self.params.epsilon >= self.route_left:
                    break

                speed = self._get_speed()

                if self.angle:
                    msg.angular.z = speed
                else:
                    msg.linear.x = speed

                if self.preempt_requested():
                    self.service_preempt()
                    return "preempted"

                self.publish_cmd_vel_cb(msg)
            sleep(0.1)

        self.publish_cmd_vel_cb(Twist())
        return None

    def _get_speed(self):
        if self.angle:
            return self.movement_direction * translate(
                self.route_left - self.route_done,
                self.params.angle_min,
                self.params.angle_max,
                self.params.speed_min,
                self.params.speed_max,
            )
        return self.movement_direction * translate(
            self.route_left - self.route_done,
            self.params.dist_min,
            self.params.dist_max,
            self.params.speed_min,
            self.params.speed_max,
        )

    def calculate_route_done(self, odom_reference: Odometry, current_odom: Odometry) -> None:
        """Function calculating route done (either angle, or distance) from the odometry message
        saved in board callback (odom_reference), to the current position.
        Saves the calculated route in a class variable "route_done".

        Args:
            odom_reference: the odometry message saved as reference position in board callback
            current_odom: the newest odometry message received by the state (current position)
        """
        if self.angle:
            self.route_done = angle_done_from_odom(odom_reference, current_odom)
        else:
            self.route_done = distance_done_from_odom(odom_reference, current_odom)

    def aruco_detection_cb(self, data: ArucoDetection) -> None:
        """Function called every time, there is new ArucoDetection message published on the topic.
        Calculates the route left from the detected board, set's odometry reference, and
        (if specified), sends debug tf's so you can visualize calculated target position in rviz.
        """
        if len(data.boards) == 0 or not self.executing:
            return

        for board in data.boards:
            if board.board_name == self.board_id:
                if not self.board_flag.is_set():
                    self.board_flag.set()

                if self.global_params.debug:
                    docking_point, docking_orientation = get_location_points_from_board(
                        board, distance=self.global_params.docking_point_dist
                    )
                    board_normalized = normalize_board(board)
                    self.debug_visualizations_cb(docking_point, docking_orientation, board_normalized)

                with self.route_lock:
                    self.calculate_route_left(board)

                with self.odom_lock:
                    self.odom_reference = self.current_odom

                break

    def wheel_odom_cb(self, data: Odometry) -> None:
        """Function called every time, there is new Odometry message published on the topic.
        Calculates the route done from the referance message saved in board callback,
        and the current odometry pose.
        """
        if not self.executing:
            return

        if not self.odom_flag.is_set():
            self.odom_flag.set()

        self.current_odom = data

        with self.odom_lock:
            if self.odom_reference:
                self.calculate_route_done(self.odom_reference, self.current_odom)

    def execute(self, ud):
        """Main state method, executed automatically on state entered"""
        self.reset_state()

        self.executing = True
        self.board_id = ud.action_goal.board_id

        start_time = time()
        while not self.board_flag.is_set() or not self.odom_flag.is_set():
            if self.preempt_requested():
                self.service_preempt()
                ud.action_result.result = f"{self.state_log_name}: state preempted."
                self.executing = False
                return "preempted"

            if time() - start_time > self.params.timeout:
                if not self.board_flag.is_set():
                    self.logger.error(f"Board (id: {self.board_id}) lost. Docking failed.")
                    ud.action_result.result = f"{self.state_log_name}: Board lost. Docking failed."
                    self.executing = False
                    return "board_lost"
                else:
                    self.logger.error("Didn't get wheel odometry message. Docking failed.")
                    ud.action_result.result = f"{self.state_log_name}: wheel odometry not working. Docking failed."
                    self.executing = False
                    return "odometry_not_working"

            sleep(0.1)

        outcome = self.movement_loop()
        if outcome:
            ud.action_result.result = f"{self.state_log_name}: state preempted."
            self.executing = False
            return "preempted"

        ud.action_feedback.current_state = f"'Reach Docking Point': sequence completed. " f"Proceeding to 'Dock' state."
        if self.state_log_name == "Dock":
            ud.action_result.result = "docking succeeded. Rover docked."

        self.executing = False
        return "succeeded"

    def service_preempt(self):
        """Function called when the state catches preemption request.
        Removes all the publishers and subscribers of the state.
        """
        self.logger.error(f"Preemption request handling for {self.state_log_name} state")
        self.publish_cmd_vel_cb(Twist())
        self.executing = False
        return super().service_preempt()


class RotateToDockingPoint(BaseDockingState):
    """The first state of the sequence state machine getting rover to docking position;
    responsible for rotating the rover towards the target point.
    (default: 0.6m in straight line from docking base).
    """

    def __init__(
        self,
        global_params: GlobalParams,
        local_params: RotateToDockingPointParams,
        publish_cmd_vel_cb: Callable,
        logger: LoggerProto,
        debug_visualizations_cb: Optional[Callable] = None,
        angular: bool = True,
        name: str = "Rotate To Docking Point",
    ):
        super().__init__(
            global_params,
            local_params,
            publish_cmd_vel_cb,
            logger,
            debug_visualizations_cb,
            angle=angular,
            name=name,
        )

    def calculate_route_left(self, board: BoardPose):
        docking_point, _ = get_location_points_from_board(board, distance=self.global_params.docking_point_dist)

        if math.sqrt(docking_point.y() ** 2 + docking_point.x() ** 2) < self.params.min_docking_point_distance:
            self.route_left = 0.0
            self.logger.info("Rover to close to the docking point in the beginning of docking process.")
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
        global_params: GlobalParams,
        local_params: ReachDockingPointParams,
        publish_cmd_vel_cb: Callable,
        logger: LoggerProto,
        debug_visualizations_cb: Optional[Callable] = None,
        angular: bool = False,
        name: str = "Reach Docking Point",
    ):
        super().__init__(
            global_params,
            local_params,
            publish_cmd_vel_cb,
            logger,
            debug_visualizations_cb,
            angle=angular,
            name=name,
        )
        self.bias_left = 0.0
        self.bias_done = 0.0
        self.bias_direction = 0.0

        self.reset_state()

    def reset_state(self):
        self.bias_done = 0.0
        self.bias_left = 0.0
        self.bias_direction = 0.0
        return super().reset_state()

    def movement_loop(self) -> Optional[str]:
        msg = Twist()

        while True:
            with self.route_lock:
                if self.route_done + self.params.epsilon >= self.route_left:
                    break

                msg.linear.x = self.movement_direction * translate(
                    self.route_left - self.route_done,
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

    def calculate_route_left(self, board: BoardPose):
        docking_point, _ = get_location_points_from_board(board, distance=self.global_params.docking_point_dist)

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
    responsible for rotating the rover toward board on the docking base - a position where the
    rover needs just to drive forward to reach the base.
    """

    def __init__(
        self,
        global_params: GlobalParams,
        local_params: ReachDockingOrientationParams,
        publish_cmd_vel_cb: Callable,
        logger: LoggerProto,
        debug_visualizations_cb: Optional[Callable] = None,
        angular: bool = True,
        name: str = "Reach Docking Orientation",
    ):
        super().__init__(
            global_params,
            local_params,
            publish_cmd_vel_cb,
            logger,
            debug_visualizations_cb,
            angle=angular,
            name=name,
        )

    def calculate_route_left(self, board: BoardPose):
        _, docking_orientation = get_location_points_from_board(board, distance=self.global_params.docking_point_dist)

        rot = PyKDL.Rotation.Quaternion(*docking_orientation)
        angle = rot.GetRPY()[2]

        self.movement_direction = 1.0 if angle > 0 else -1.0
        self.route_left = math.fabs(angle)
