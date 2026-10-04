cmake_minimum_required(VERSION 3.8)
project(swift_pico)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)   # ✅ Add this line
find_package(swift_msgs REQUIRED)
find_package(controller_msg REQUIRED)
find_package(error_msg REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclpy REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(swift_pico_description REQUIRED)

# Target installations

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

install(
  DIRECTORY worlds/
  DESTINATION share/${PROJECT_NAME}/worlds
)

# Install scripts
install(PROGRAMS
  scripts/task1a.cpp
  scripts/task1a.py
  src/pico_controller_PID.py
  swift_pico/pico_controller_LQR.py
  src/pico_controller_PID.cpp
  src/pico_controller_LQR.cpp
  swift_pico/pico_client.py
  swift_pico/pico_server.py
  swift_pico/waypoint_service.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_python_install_package(${PROJECT_NAME})

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()

