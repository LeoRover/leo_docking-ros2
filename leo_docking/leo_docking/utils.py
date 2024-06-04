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
from typing import Tuple, Protocol

import math

import tf2_ros
from rclpy.clock import ROSClock

from geometry_msgs.msg import TransformStamped, Pose
from aruco_opencv_msgs.msg import BoardPose
from nav_msgs.msg import Odometry

import PyKDL


class LoggerProto(Protocol):
    """Protocol class describing a typical logger object.

    Satisfied by rclpy.impl.rcutils_logger.RcutilsLogger, AKA standard rclpy logger.
    """

    def debug(self, msg: str) -> None:
        """Log debug message."""

    def info(self, msg: str) -> None:
        """Log info message."""

    def warning(self, msg: str) -> None:
        """Log warning message."""

    def error(self, msg: str) -> None:
        """Log error message."""


def frame_to_pose(frame: PyKDL.Frame) -> Pose:
    """Function converting PyKDL Frame object to rospy Pose object.

    Ars:
        frame: PyKDL Frame to be converted

    Returns:
        out_pose: rospy pose made from conversion of `frame`
    """

    out_pose: Pose = Pose()

    # converting orientation
    out_pose.orientation.x = frame.M.GetQuaternion()[0]
    out_pose.orientation.y = frame.M.GetQuaternion()[1]
    out_pose.orientation.z = frame.M.GetQuaternion()[2]
    out_pose.orientation.w = frame.M.GetQuaternion()[3]

    # converting position
    out_pose.position.x = frame.p.x()
    out_pose.position.y = frame.p.y()
    out_pose.position.z = frame.p.z()

    return out_pose


def pose_to_frame(pose: Pose) -> PyKDL.Frame:
    """Function converting rospy Pose object to PyKDL Frame object (hardcodes z coordinate equal
    to 0.0).

    Args:
        pose: rospy Pose to be converted

    Returns:
        out_frame: PyKDL Frame made from coversion of `pose`
    """

    # taking position and orientation from pose
    orientation = pose.orientation
    position = pose.position

    # creating frame
    out_frame: PyKDL.Frame = PyKDL.Frame(
        PyKDL.Rotation.Quaternion(orientation.x, orientation.y, orientation.z, orientation.w),
        PyKDL.Vector(position.x, position.y, 0.0),
    )

    return out_frame


def normalize_board(board: BoardPose) -> PyKDL.Frame:
    """Function normalizig pose of detected board in base_link frame:
    - position: puts board on level z = 0.0.
    - orientation: calculates board's yaw from unit vector z (vector looking in z axis) in respect
                   to board with arctangent, and normalizes the orientation - makes board have
                   x,y,z axis in the same scheme as rover has (x, y, z - forward, left, up).

    Args:
        board: pose of detected board used for localization of docking station

    Returns:
        normalized_board: normalized board position in base_link frame
    """

    # converting pose to frame, and placing board on z=0.0
    base_board = pose_to_frame(board.pose)

    # calculating board's yaw
    unit_z = base_board.M.UnitZ()
    angle = math.atan2(unit_z.y(), unit_z.x())

    # making normalized orientation - rotating base orientation in z axis by board's yaw
    rot = PyKDL.Rotation.RotZ(angle)

    # making final frame
    normalized_board = PyKDL.Frame(rot, base_board.p)

    return normalized_board


def get_location_points_from_board(
    board: BoardPose, distance: float = 0.0
) -> Tuple[PyKDL.Vector, Tuple[float, float, float, float]]:
    """Function calculating from detected board a pose and orientation that rover should have
    before riding on the docking station.

    Args:
        board: pose of the board used for calculating the final pose and orientation
        distance: distance (in meters) of the desired position from the board on the docking
                  station

    Returns:
        docking_point: point in front of the docking station away by `distance` - point where rover
                       can freely rotate without moving docking station
        docking_orientation: target orientation of the rover before reaching docking station; tuple
                             representing quaternion (x, y, z, w)
    """
    board_frame = normalize_board(board)

    # getting docking point - rotating vector (distance, 0, 0) by board orientation
    docking_point_base = PyKDL.Vector(distance, 0.0, 0.0)
    # no need of projection to z=0.0 because we use normalized board which is already on z=0.0
    docking_point = board_frame * docking_point_base

    # target orientation
    angle, *_ = board_frame.M.GetEulerZYX()
    docking_orientation = PyKDL.Rotation.RotZ(angle + math.pi).GetQuaternion()

    return docking_point, docking_orientation


def calculate_odom_diff_pose(start_odom_pose: Odometry, current_odom_pose: Odometry) -> PyKDL.Frame:
    """Function calculating difference between to odom poses returnig it as PyKDL Frame object.

    Args:
        start_odom_pose: odometry pose used as reference position
        current_odom_pose: current odometry pose of robot

    Returns:
        diff_pose: difference pose; represents distance and rotation made from `start_odom_pose` to
                   `current_odom_pose`
    """

    start_odom_frame = pose_to_frame(start_odom_pose.pose.pose).Inverse()
    current_odom_frame = pose_to_frame(current_odom_pose.pose.pose)

    # getting start_odom_pose -> current_odom_pose transform pose
    diff_pose = start_odom_frame * current_odom_frame

    return diff_pose


def angle_done_from_odom(start_odom_pose: Odometry, current_odom_pose: Odometry) -> float:
    """Function calculating angle done between two given odometry positions.

    Args:
        start_odom_pose: odometry pose used as reference position
        current_odom_pose: current odometry pose of robot

    Returns:
        angle: angle in radians done from `start_odom_pose` to `current_odom_pose`
    """

    odom_diff = calculate_odom_diff_pose(start_odom_pose, current_odom_pose)

    # getting yaw from the difference pose
    angle = math.fabs(odom_diff.M.GetRPY()[2])

    return angle


def distance_done_from_odom(start_odom_pose: Odometry, current_odom_pose: Odometry) -> float:
    """Function calculating distance done between two given odometry positions.

    Args:
        start_odom_pose: odometry pose used as reference position
        current_odom_pose: current odometry pose of robot

    Returns:
        distance: distance in meters done from `start_odom_pose` to `current_odom_pose`
    """

    odom_diff = calculate_odom_diff_pose(start_odom_pose, current_odom_pose)

    # calculating the distance using Pythagorean theorem
    distance = math.sqrt(odom_diff.p.x() ** 2 + odom_diff.p.y() ** 2)

    return distance


def visualize_position(
    position: PyKDL.Vector,
    orientation: PyKDL.Rotation.Quaternion,
    frame_id: str,
    child_frame_id: str,
    tf_broadcaster: tf2_ros.TransformBroadcaster,
    stamp,
) -> None:
    """Function used for visualizing poses in rviz as transforms. Used only for debug.

    Args:
        position: position of the visualized pose
        orientation: orientation of the visualized pose
        frame_id: name of the parent frame of the transformation
        child_frame_id: name of the child frame of the transformation
        seq: sequence number of the transform needed for header
        tf_broadcaster: transform boradcaster
    """

    msg = TransformStamped()

    # filling orientation
    msg.transform.rotation.x = orientation[0]
    msg.transform.rotation.y = orientation[1]
    msg.transform.rotation.z = orientation[2]
    msg.transform.rotation.w = orientation[3]

    # filing position
    msg.transform.translation.x = position.x()
    msg.transform.translation.y = position.y()
    msg.transform.translation.z = position.z()

    # filling header
    msg.child_frame_id = child_frame_id
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp

    # sending transform
    tf_broadcaster.sendTransform(msg)


def translate(value, left_min, left_max, right_min, right_max) -> float:
    """Function proportionaly translating value from one interval into second interval.

    Args:
        value: value to be translated
        left_min: minimal value of the first interval
        left_max: maximal value of the first interval
        right_min: minimal value of the second interval
        right_max: maximal value of the second interval

    Returns:
        ans: translated `value` in second interval
    """

    # placing the value in the first interval boundaries
    value = min(max(value, left_min), left_max)

    # Figure out how 'wide' each range is
    left_span = left_max - left_min
    right_span = right_max - right_min

    # Convert the left range into a 0-1 range (float)
    value_scaled = float(value - left_min) / float(left_span)

    # Convert the 0-1 range into a value in the right range.
    ans = right_min + (value_scaled * right_span)
    return ans


def calculate_threshold_distances(board: BoardPose) -> Tuple[float, float]:
    """Function calculating the distance from rover to board sight
    (x axis of the board frame) and how far on the board's x axis the rover is.
    which are needed for the area threshold checking.

    Args:
        board: board position in base_link frame
    returns:
        x_dist: distance from board to the rover's position projected on board's x axis
        y_dist: distance from rover to the board sight (x axis of the board)
    """
    normalized_board = normalize_board(board)

    unit_vec_x = normalized_board.M.UnitX()

    position = normalized_board.p

    # getting equation for the line going through board and unit_vec_x (from board's perspective)
    coeff_a = (position.y() - unit_vec_x[1]) / (position.x() - unit_vec_x[0])
    coeff_b = position.y() - coeff_a * position.x()

    # getting perpendicular line to the previous one going through rover's position (0,0)
    perpend_coeff_a = -1.0 / coeff_a
    perpend_coeff_b = 0.0

    # getting crssing point of the two lines
    x_cross = (perpend_coeff_b - coeff_b) / (coeff_a - perpend_coeff_a)
    y_cross = coeff_a * x_cross + coeff_b

    y_dist = math.sqrt(x_cross**2 + y_cross**2)

    x_dist = math.sqrt((x_cross - position.x()) ** 2 + (y_cross - position.y()) ** 2)

    return x_dist, y_dist
