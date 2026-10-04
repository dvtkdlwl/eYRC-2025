cmake_minimum_required(VERSION 3.8)
project(waypoint_navigation)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate interfaces (messages, services, actions)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Waypoint.msg"
  "msg/MissionPlan.msg"
  "srv/GetWaypoints.srv"
  "action/NavToWaypoint.action"
  DEPENDENCIES builtin_interfaces
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()

