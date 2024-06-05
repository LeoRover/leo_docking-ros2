cmake_minimum_required(VERSION 3.5)
project(leo_docking_msgs)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)


# For msg, src and action files
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)

set(msg_files

)

set(srv_files

)

set(action_files
  "action/PerformDocking.action"
)

set(dependencies
    builtin_interfaces
    std_msgs
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${action_files}
  ${msg_files}
  ${srv_files}
  DEPENDENCIES ${dependencies}
)

if(BUILD_TESTING)
endif()

ament_package()