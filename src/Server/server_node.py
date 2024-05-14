#!/usr/bin/python3

import rospy
from Experiments.msg import AgentState, Tensor3D
from Experiments.srv import ServerRequest, ServerRequestResponse
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from std_msgs.msg import Float32MultiArray, Int32MultiArray
from collections import deque
import torch
import time
from geometry_msgs.msg import Quaternion
from tf.transformations import quaternion_from_euler

class ServerNode:
    def __init__(self):
        rospy.init_node('server_node')
        
        self.num_agents = rospy.get_param('~num_agents', 10)
        self.memory = rospy.get_param('~memory', 10)
        # Object to store state histories for each agent
        self.state_histories = {i: deque(maxlen=self.memory) for i in range(self.num_agents)}
        # Interactive: Publisher, subscriber anbd Service
        self.marker_pub = rospy.Publisher('multi_agent_viz', MarkerArray, queue_size=10)
        rospy.Subscriber('/agent_states', AgentState, self.state_callback, queue_size=10)
        
        rospy.Service('/get_state_histories', ServerRequest, self.handle_get_state_histories)

        self.rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            self.publish_markers()
            self.rate.sleep()
        

    def handle_get_state_histories(self, req):
        agent_id = int(req.id)
        # Index List
        agent_list = self.create_agent_list(self.num_agents, agent_id)

        # Create a 3D tensor of the last item of each state history 
        state_histories = torch.tensor([self.state_histories[i][-1] for i in agent_list], dtype=torch.float32)
        # Get shape of tensor and convert to integer list
        shape = list(state_histories.shape)
        
        # Flatten Tensor (Shape is 3D)
        state_histories = state_histories.view(-1)
        
        # Convert to list
        state_histories = state_histories.tolist()

       # Convert state_histories to a Float32MultiArray
        state_histories_array = Float32MultiArray(data=state_histories)

        # Convert shape to an Int32MultiArray
        shape_array = Int32MultiArray(data=shape)

        response = Tensor3D()
        response.data = state_histories_array
        response.shape = shape_array
        

        return ServerRequestResponse(state_histories=response)


    def state_callback(self, data):
        
        
        agent_id = data.id
        self.state_histories[agent_id].append(data.state)
        
        
        


    def publish_markers(self):
        marker_array = MarkerArray()
        
        for key, state_history in self.state_histories.items():
            
            if len(state_history) > 0:
                marker = Marker()
                marker.header.frame_id = "map"
                marker.id = key
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.scale.x = 1
                marker.scale.y = 1
                marker.scale.z = 1
                marker.color = ColorRGBA(key / float(self.num_agents), 1 - key / float(self.num_agents), 0.5, 1)
                marker.pose.position.x = state_history[-1][0]
                marker.pose.position.y = state_history[-1][1]
                marker.pose.position.z = 0.2
                # Set a valid orientation for the marker (e.g., no rotation)
                quat = quaternion_from_euler(0, 0, 0)  # No rotation
                marker.pose.orientation = Quaternion(*quat)
                marker_array.markers.append(marker)
        self.marker_pub.publish(marker_array)



    def spin(self):
        rospy.spin()

    def create_agent_list(self, num_agents, agent_id):
        return [i for i in range(num_agents) if i != agent_id]


if __name__ == "__main__":
    node = ServerNode()
    node.spin()