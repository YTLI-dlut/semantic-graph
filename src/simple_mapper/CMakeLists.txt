cmake_minimum_required(VERSION 3.0.2)
project(simple_mapper)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  nav_msgs
  cv_bridge
  std_msgs
  image_transport
)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_executable(mapper_node src/mapper_node.cpp)
target_link_libraries(mapper_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})