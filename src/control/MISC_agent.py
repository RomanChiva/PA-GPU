#!/usr/bin/python3

import sys
import os
# Get the parent directory of the current script
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# Add the parent directory to the Python path
sys.path.append(parent_dir)


import rospy
from tf.transformations import euler_from_quaternion
import numpy as np
import time
from planner.MPPI_wrapper import MPPI_Wrapper
from hydra.experimental import initialize, compose
import torch
from omegaconf import OmegaConf
from interface.AgentInterface import AgentInterfacePlus
from geometry_msgs.msg import PoseStamped
from costfn.KL_agent import ObjectiveLegibility
#from costfn.Entropy_Cost import ObjectiveLegibility
from utils.config_store import *
#from utils.plotter import plot_gmm
from PredicionModels.GoalOrientedPredictions import goal_oriented_predictions




class MISC_Agent:

    def __init__(self):

        config = self.load_config()
        self.cfg = OmegaConf.to_object(config)
        
        self.interface = AgentInterfacePlus(self.cfg)
        self.agent_id = self.interface.ID
        print('Hello from Agent: ', self.agent_id)
        
        rospy.sleep(1)
        
        self.objective = ObjectiveLegibility(self.cfg, None, self.interface, self.agent_id)
        self.mppi_planner = MPPI_Wrapper(self.cfg, self.objective)
        
        ## State Variables (X,Y, V, Phi) (Unicycle DYnmaics)
        self.state = torch.tensor(self.cfg.multi_agent.starts[self.agent_id], device=self.cfg.mppi.device).float()
        self.interface.state = torch.tensor(self.cfg.multi_agent.starts[self.agent_id], device=self.cfg.mppi.device).float()
        self.interface.publish_state_to_server()
        rospy.sleep(0.5)
        

    def load_config(self):
        # Load Hydra configurations
        with initialize(config_path="../../conf"):
            config = compose(config_name="KL_Cost")
        return config


    def actuate(self, next_state):
        # Actuate the robot by updating the state
        # Tae a shortcut and just update the state disrectly to be the first from the plan
        self.state = next_state

    def run_agent(self):
       
        # MISC Variables
        step_num = 0
        computational_time = []
        plans = []

        # INITIALIZE LOOP
        while not rospy.is_shutdown():
            
            
            # Get self.state and turn the torch tensor to a list
            robot_position = self.state.tolist()
            
            
            
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = self.state[0].item()
            pose.pose.position.y = self.state[1].item()
            pose.pose.position.z = 0
            self.interface.path.poses.append(pose)
            self.interface.path_publisher.publish(self.interface.path)
            

            self.interface.trajectory.append([robot_position[0], robot_position[1]])

            # compute action

            if step_num == 0 and self.cfg.mppi.warm_start:
                for _ in range(self.cfg.mppi.warm_start_steps):
                    action, plan, states = self.mppi_planner.compute_action(
                    q=robot_position,
                    qdot=None,
                    obst=None)
            else: 
                start_time = time.time()
                action, plan, states = self.mppi_planner.compute_action(
                    q=robot_position,
                    qdot=None,
                    obst=None)
                end_time = time.time()
                print('Time: ', end_time - start_time, 's', 'Agent ID: ', self.agent_id)

            plans.append(plan.numpy())
            if step_num > 0:
                computational_time.append(end_time - start_time)
            step_num += 1
            self.interface.timesteps += 1
            # visualize trajectory
            self.interface.visualize_trajectory(plan)
            # Plot Plan and predictions 
            #pred, weights = goal_oriented_predictions(self.interface, self.cfg, return_original=True)
            #print(pred.shape, weights.shape, plan.shape)
            #if step_num % 50 == 0:
                #plot_gmm(pred, self.cfg.costfn.sigma_pred, weights, trajectory=plan)
            # actuate robot and sleep
            self.actuate(plan[1])
            
            # Publish to server
            self.interface.state = self.state
            self.interface.publish_state_to_server()



            # check if the target goal is reached
            if self.interface.reset_sim:
                #self.interface.reset_env()
                print("planning computational time is: ", (np.sum(computational_time) / step_num) * 1000, " ms")
                
                computational_time = []
                self.interface.timesteps = 0
                self.interface.trajectory = []
                step_num = 0
                self.mppi_planner = MPPI_Wrapper(self.cfg, self.objective)
                
                # Save a copy of Plans in the plans folder
                #print(np.array(plans).shape, 'plans shape')
                #np.save('/home/roman/ROS/catkin_ws/src/Experiments/src/utils/plans_KL2.npy', np.array(plans))
                plans = []
                # Set state back to initital state
                self.state = torch.tensor(self.cfg.multi_agent.starts[self.agent_id], device=self.cfg.mppi.device).float()
                #break
                #sys.exit(0)
                # Sleep for 1 second
                rospy.sleep(1)




if __name__ == '__main__':
    try:
        planner = MISC_Agent()
        planner.run_agent()

    except rospy.ROSInterruptException:
        pass