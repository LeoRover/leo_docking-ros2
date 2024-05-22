#!/usr/bin/env python3
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
        executable="aruco_tracker",
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
