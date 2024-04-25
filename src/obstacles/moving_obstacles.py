#!/usr/bin/python3
import rospy
from geometry_msgs.msg import Pose, Twist
from Experiments.msg import Obstacle, ObstacleArray
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import yaml
import os

class MovingObstacle:
    def __init__(self, initial_position, velocity, moving_range):
        self.position = np.array(initial_position, dtype=np.float32)
        self.speed = float(velocity)
        self.velocity = np.array([self.speed, 0], dtype=np.float32)
        self.moving_range = np.array(moving_range, dtype=np.float32)
        self.target_position = moving_range[1]

    def update_position(self, timestep):
        # Calculate the direction vector
        direction = self.target_position - self.position

        # Normalize the direction vector
        direction /= np.linalg.norm(direction)

        # Calculate the displacement
        self.velocity = direction * self.speed * timestep

        # Add a small random noise to the displacement
        #noise = np.random.normal(scale=0.01, size=self.velocity.shape)
        #self.velocity += noise

        # Update the position
        self.position += self.velocity

        # If the obstacle has reached the target position, set the target position to the other end of the moving range
        if np.linalg.norm(self.target_position - self.position) < self.speed * timestep:
            self.target_position = self.moving_range[1] if np.array_equal(self.target_position, self.moving_range[0]) else self.moving_range[0]






class MovingObstaclesNode:
    def __init__(self, path):

        with open(path, 'r') as file:
            config = yaml.safe_load(file)

        self.obstacles = [MovingObstacle(**params) for params in config['obstacles']]
        self.pub = rospy.Publisher('obstacle_states', ObstacleArray, queue_size=10)
        self.marker_pub = rospy.Publisher('obstacle_markers', MarkerArray, queue_size=10)

    def update_and_publish(self, timestep):

        obstacle_array = ObstacleArray()
        for obstacle in self.obstacles:
            obstacle.update_position(timestep)
            obstacle_msg = Obstacle()

            obstacle_msg.position = Pose()
            obstacle_msg.position.position.x = obstacle.position[0]
            obstacle_msg.position.position.y = obstacle.position[1]
            obstacle_msg.position.position.z = 0.2

            obstacle_msg.velocity = Twist()
            obstacle_msg.velocity.linear.x = obstacle.velocity[0]
            obstacle_msg.velocity.linear.y = obstacle.velocity[1]
            obstacle_msg.velocity.linear.z = 0

            obstacle_array.obstacles.append(obstacle_msg)
        self.pub.publish(obstacle_array)

        # Publish Marker Array for RVIZ

        marker_array = MarkerArray()
        for i, obstacle in enumerate(self.obstacles):

            marker = Marker()
            marker.header.frame_id = "map"
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale.x = 1.0
            marker.scale.y = 1.0
            marker.scale.z = 1.0
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.9
            marker.color.b = 0.0
            marker.pose.position.x = obstacle.position[0]
            marker.pose.position.y = obstacle.position[1]
            marker.pose.position.z = 0.2
            marker.id = i
            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)




if __name__ == '__main__':
    
    rospy.init_node('moving_obstacles_node')

    # Load Path
    current_dir = os.path.dirname(os.path.realpath(__file__))
    config_dir = os.path.join(current_dir, '..', '..', 'conf', 'obstacles')
    path = os.path.join(config_dir, 'Test.yaml')

    
    node = MovingObstaclesNode(path)
    timestep = 0.1
    rate = rospy.Rate(10) # 10 Hz


    while not rospy.is_shutdown():
        node.update_and_publish(timestep)
        rate.sleep()