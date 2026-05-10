cmake_minimum_required(VERSION 3.0.2)
project(odom_bridge)

find_package(catkin REQUIRED COMPONENTS
  rospy
  nav_msgs
  geometry_msgs
  std_msgs
  tf
  tf2_ros
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/odom_bridge.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
