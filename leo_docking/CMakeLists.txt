cmake_minimum_required(VERSION 3.5)
project(leo_docking)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(action_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(aruco_opencv_msgs REQUIRED)
find_package(python3_pykdl)
find_package(aruco_opencv REQUIRED)

install(DIRECTORY
  params
  launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  leo_docking/docking_server.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
    #find_package(ament_cmake_pytest REQUIRED)
    #ament_add_pytest_test(${PROJECT_NAME}_test .)
endif()

ament_package()