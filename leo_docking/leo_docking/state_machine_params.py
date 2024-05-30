#  ------------------------------------------------------------------
#   Copyright (C) Karelics Oy - All Rights Reserved
#   Unauthorized copying of this file, via any medium is strictly
#   prohibited. All information contained herein is, and remains
#   the property of Karelics Oy.
#  ------------------------------------------------------------------

from dataclasses import dataclass
from rclpy.node import Node


@dataclass
class GlobalParams:
    """Global parameters"""
    timeout: int
    debug: bool
    epsilon: float
    docking_point_dist: float
    battery_diff: float
    max_battery_average: float
    battery_averaging_time: float
    effort_threshold: float
    effort_buffer_size: int
    motor_cd_time: float


@dataclass
class StartParams:
    """Parameters for the start state"""
    timeout: int


@dataclass
class CheckAreaParams:
    """Parameters for the CheckArea state"""
    timeout: int
    threshold_angle: float
    docking_distance: float


@dataclass
class RotateToDockAreaParams:
    """Parameters for the RotateToDockArea state"""
    timeout: int
    epsilon: float
    speed_min: float
    speed_max: float
    angle_min: float
    angle_max: float


@dataclass
class RideToDockAreaParams:
    """Parameters for the RideToDockArea state"""
    timeout: int
    epsilon: float
    speed_min: float
    speed_max: float
    dist_min: float
    dist_max: float


@dataclass
class RotateToBoardParams:
    """Parameters for the RotateToBoard state"""
    timeout: int
    epsilon: float
    speed_min: float
    speed_max: float
    angle_min: float
    angle_max: float


@dataclass
class RotateToDockingPointParams:
    """Parameters for the RotateToDockingPoint state"""
    timeout: int
    epsilon: float
    speed_min: float
    speed_max: float
    angle_min: float
    angle_max: float
    min_docking_point_distance: float


@dataclass
class ReachDockingPointParams:
    """Parameters for the ReachDockingPoint state"""
    timeout: int
    epsilon: float
    speed_min: float
    speed_max: float
    dist_min: float
    dist_max: float
    bias_speed_min: float
    bias_speed_max: float
    bias_min: float
    bias_max: float


@dataclass
class ReachDockingOrientationParams:
    """Parameters for the ReachDockingOrientation state"""
    timeout: int
    epsilon: float
    speed_min: float
    speed_max: float
    angle_min: float
    angle_max: float


@dataclass
class DockParams:
    """Parameters for the Dock state"""
    timeout: int
    epsilon: float
    speed_min: float
    speed_max: float
    dist_min: float
    dist_max: float
    bias_speed_min: float
    bias_speed_max: float
    bias_min: float
    bias_max: float


class StateMachineParams:
    def __init__(self, node: Node):
        self.global_params = GlobalParams(
            timeout=node.declare_parameter("timeout", 3).value,
            debug=node.declare_parameter("debug", True).value,
            epsilon=node.declare_parameter("epsilon", 0.1).value,
            docking_point_dist=node.declare_parameter("docking_point_distance", 0.8).value,
            battery_diff=node.declare_parameter("battery_diff", 0.2).value,
            max_battery_average=node.declare_parameter("max_battery_average", 11.0).value,
            battery_averaging_time=node.declare_parameter("battery_averaging_time", 1.0).value,
            effort_threshold=node.declare_parameter("effort_threshold", 2.0).value,
            effort_buffer_size=node.declare_parameter("effort_buffer_size", 10).value,
            motor_cd_time=node.declare_parameter("motor_cd_time", 2.0).value,
        )
        self.start_params = StartParams(
            timeout=node.declare_parameter("start_state/timeout", self.global_params.timeout).value,
        )
        self.check_area_params = CheckAreaParams(
            timeout=node.declare_parameter("check_area/timeout", self.global_params.timeout).value,
            threshold_angle=node.declare_parameter("check_area/threshold_angle", 0.17).value,
            docking_distance=node.declare_parameter("check_area/docking_distance", 2.0).value,
        )
        self.rotate_to_dock_area_params = RotateToDockAreaParams(
            timeout=node.declare_parameter("rotate_to_dock_area/timeout", self.global_params.timeout).value,
            epsilon=node.declare_parameter("rotate_to_dock_area/epsilon", self.global_params.epsilon).value,
            speed_min=node.declare_parameter("rotate_to_dock_area/speed_min", 0.1).value,
            speed_max=node.declare_parameter("rotate_to_dock_area/speed_max", 0.4).value,
            angle_min=node.declare_parameter("rotate_to_dock_area/angle_min", 0.05).value,
            angle_max=node.declare_parameter("rotate_to_dock_area/angle_max", 1.05).value,
        )
        self.ride_to_dock_area_params = RideToDockAreaParams(
            timeout=node.declare_parameter("ride_to_dock_area/timeout", self.global_params.timeout).value,
            epsilon=node.declare_parameter("ride_to_dock_area/epsilon", self.global_params.epsilon).value,
            speed_min=node.declare_parameter("ride_to_dock_area/speed_min", 0.05).value,
            speed_max=node.declare_parameter("ride_to_dock_area/speed_max", 0.4).value,
            dist_min=node.declare_parameter("ride_to_dock_area/distance_min", 0.1).value,
            dist_max=node.declare_parameter("ride_to_dock_area/distance_max", 0.5).value,
        )
        self.rotate_to_board_params = RotateToBoardParams(
            timeout=node.declare_parameter("rotate_to_board/timeout", self.global_params.timeout).value,
            epsilon=node.declare_parameter("rotate_to_board/epsilon", self.global_params.epsilon).value,
            speed_min=node.declare_parameter("rotate_to_board/speed_min", 0.1).value,
            speed_max=node.declare_parameter("rotate_to_board/speed_max", 0.4).value,
            angle_min=node.declare_parameter("rotate_to_board/angle_min", 0.05).value,
            angle_max=node.declare_parameter("rotate_to_board/angle_max", 1.05).value,
        )
        self.rotate_to_docking_point_params = RotateToDockingPointParams(
            timeout=node.declare_parameter("rotate_to_docking_point/timeout", self.global_params.timeout).value,
            epsilon=node.declare_parameter("rotate_to_docking_point/epsilon", self.global_params.epsilon).value,
            speed_min=node.declare_parameter("rotate_to_docking_point/speed_min", 0.1).value,
            speed_max=node.declare_parameter("rotate_to_docking_point/speed_max", 0.4).value,
            angle_min=node.declare_parameter("rotate_to_docking_point/angle_min", 0.05).value,
            angle_max=node.declare_parameter("rotate_to_docking_point/angle_max", 1.0).value,
            min_docking_point_distance=node.declare_parameter(
                "rotate_to_docking_point/min_docking_point_distance", 0.1
            ).value,
        )
        self.reach_docking_point_params = ReachDockingPointParams(
            timeout=node.declare_parameter("reach_docking_point/timeout", self.global_params.timeout).value,
            epsilon=node.declare_parameter("reach_docking_point/epsilon", self.global_params.epsilon).value,
            speed_min=node.declare_parameter("reach_docking_point/speed_min", 0.1).value,
            speed_max=node.declare_parameter("reach_docking_point/speed_max", 0.3).value,
            dist_min=node.declare_parameter("reach_docking_point/distance_min", 0.05).value,
            dist_max=node.declare_parameter("reach_docking_point/distance_max", 1.5).value,
            bias_speed_min=node.declare_parameter("reach_docking_point/bias_speed_min", 0.05).value,
            bias_speed_max=node.declare_parameter("reach_docking_point/bias_speed_max", 0.3).value,
            bias_min=node.declare_parameter("reach_docking_point/bias_min", 0.01).value,
            bias_max=node.declare_parameter("reach_docking_point/bias_max", 0.1).value,
        )
        self.reach_docking_orientation_params = ReachDockingOrientationParams(
            timeout=node.declare_parameter("reach_docking_orientation/timeout", self.global_params.timeout).value,
            epsilon=node.declare_parameter("reach_docking_orientation/epsilon", self.global_params.epsilon).value,
            speed_min=node.declare_parameter("reach_docking_orientation/speed_min", 0.1).value,
            speed_max=node.declare_parameter("reach_docking_orientation/speed_max", 0.4).value,
            angle_min=node.declare_parameter("reach_docking_orientation/angle_min", 0.05).value,
            angle_max=node.declare_parameter("reach_docking_orientation/angle_max", 1.05).value,
        )
        self.dock_params = DockParams(
            timeout=node.declare_parameter("dock/timeout", self.global_params.timeout).value,
            epsilon=node.declare_parameter("dock/epsilon", self.global_params.epsilon).value,
            speed_min=node.declare_parameter("dock/speed_min", 0.05).value,
            speed_max=node.declare_parameter("dock/speed_max", 0.2).value,
            dist_min=node.declare_parameter("dock/distance_min", 0.05).value,
            dist_max=node.declare_parameter("dock/distance_max", 0.8).value,
            bias_speed_min=node.declare_parameter("dock/bias_speed_min", 0.05).value,
            bias_speed_max=node.declare_parameter("dock/bias_speed_max", 0.3).value,
            bias_min=node.declare_parameter("dock/bias_min", 0.01).value,
            bias_max=node.declare_parameter("dock/bias_max", 0.1).value,
        )
