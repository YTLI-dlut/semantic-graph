cmake_minimum_required(VERSION 3.0.2)
project(habitat_bridge)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  cv_bridge
)

catkin_package()

# 这一步是为了让 rosrun 能找到你的 python 脚本
catkin_install_python(PROGRAMS
  src/run_habitat.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)