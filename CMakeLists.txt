cmake_minimum_required(VERSION 3.0.2)
project(leo_docking)

find_package(catkin REQUIRED)

catkin_python_setup()

catkin_package()

# include_directories(
#   ${catkin_INCLUDE_DIRS}
# )

catkin_install_python(
  PROGRAMS 
    scripts/test_angles
    scripts/docking_main
    scripts/test
    # scripts/start_state.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)