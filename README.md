# Leo Rover Docking

This is a ROS 2 port of the leo_docking repo.
Original repo: https://github.com/LeoRover/leo_docking

Run `ros2 launch leo_docking dock_rover.launch.py` to launch the state machine.

Run `ros2 action send_goal /leo_rover/dock leo_docking_msgs.action.PerformDocking '{board_id: "INSERT YOUR BOARD ID HERE"}'` to start docking.

For information on adding boards check the ros_aruco_opencv repo: https://github.com/fictionlab/ros_aruco_opencv