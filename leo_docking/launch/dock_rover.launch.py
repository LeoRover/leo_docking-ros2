#!/usr/bin/env python3
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
import os

# ROS
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Insta360 camera driver launch."""
    launch_list = []
    params_dir = get_package_share_directory("leo_docking") + "/params"
    leo_docking = Node(
        package="leo_docking",
        executable="docking_server.py",
        output={"both": {"screen", "log", "own_log"}},
        emulate_tty=True,
        parameters=[
            os.path.join(params_dir, "docking_config.yaml"),
        ],
        arguments=["--ros-args", "--log-level", "info"],
    )

    aruco_tracker = Node(
        package="aruco_opencv",
        executable="aruco_tracker_autostart",
        output={"both": {"screen", "log", "own_log"}},
        emulate_tty=True,
        parameters=[
            os.path.join(params_dir, "aruco_tracker.yaml"),
            {
                "board_descriptions_path": os.path.join(params_dir, "board_descriptions.yaml"),
            },
        ],

    )

    launch_list += [
        leo_docking,
        aruco_tracker,
    ]
    return LaunchDescription(launch_list)
