cmake_minimum_required(VERSION 3.0.2)
project(Experiments)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
  geometry_msgs
)

add_message_files(
  FILES
  Obstacle.msg
  ObstacleArray.msg
  AgentState.msg
  Tensor3D.msg
)


add_service_files(
  FILES
  ServerRequest.srv
)


generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs message_runtime
)