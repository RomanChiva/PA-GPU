#!/usr/bin/python3

import numpy as np
from scipy.interpolate import CubicSpline
from utils.cubicspline import CubicSpline2D
import torch
from sklearn.mixture import GaussianMixture
import time
from Experiments.msg import Obstacle, ObstacleArray


class ObjectiveLegibility(object):

    def __init__(self, cfg, obstacles, interface):
        # Create two possible goals used in pred_model
        self.cfg = cfg
        self.goal_index = cfg.costfn.goal_index
        self.other_goal = 1 - self.goal_index
        self.goals = torch.tensor(self.cfg.costfn.goals, dtype=torch.float32, device=self.cfg.mppi.device)
        self.v_ref = torch.tensor(cfg.v_ref, device=cfg.mppi.device)
        self.x_ref = np.array([0, 3, 6, 9, 12, 15, 18, 21, 30, 34, 38])
        self.y_ref = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.reference_spline = CubicSpline2D(self.x_ref, self.y_ref)
        self.obstacles = obstacles
        self.interface = interface
        

    

    # Cost Function for free navigation (Straight to goal)
    def compute_cost(self, state):
       

        ## Send all tensors we need to GPU
        self.trajectory = torch.tensor(self.interface.trajectory, device=self.cfg.mppi.device)
        self.position = torch.tensor([self.interface.odom_msg.pose.pose.position.x,
                                self.interface.odom_msg.pose.pose.position.y], device=self.cfg.mppi.device)
        obstacles = self.interface.obstacles
        # The function already picks a device
        self.obstacle_predictions = self.propagate_positions(obstacles, self.cfg.mppi.horizon)
        

        goal_cost = self.goal_cost(state)
        # Legibility Cost

        entropy_cost = self.entropy_cost(state)
        entropy_cost = entropy_cost.reshape(-1)
        
        # Obstacle Cost
        obstacle_cost = self.obstacle_cost(state)
        obstacle_cost = obstacle_cost.reshape(-1)

        
        

        # Add them
        return goal_cost + obstacle_cost + 10*entropy_cost 
     

    def goal_cost(self, state):
        

        state_goal = state.permute(1, 0, 2)
        # Now reshape to (T*K, nx)
        state_goal = state_goal.reshape(-1, self.cfg.nx)
        pos_goal = state_goal[:, 0:2]
        goal_cost = torch.linalg.norm(pos_goal - self.goals[self.goal_index], axis=1)

        return goal_cost



    def entropy_cost(self, state):

        ## Get Predictions
        state = state.permute(1, 0, 2)

        weights = self.pred_mod_goals(state)
        
        ## Compute entropy of the weights at each timestep and sample along the 2nd dimesnion
        entropy = -torch.sum(weights * torch.log2(weights), dim=2)
        
        return entropy

    def pred_mod_goals(self, state):


        pos = state[:, :, 0:2]
        # Find vectors to goals 1 and 2 (Goal[-] shape[1,2] pos shape [T, K, 2])
        vector_goal1 = self.goals[0] - pos
        vector_goal2 = self.goals[1] - pos

        # Find magnitudes of vectors
        magnitude_goal1 = torch.linalg.norm(vector_goal1, axis=2)
        magnitude_goal2 = torch.linalg.norm(vector_goal2, axis=2)

        # Find distance travelled so far
        distance = self.compute_path_length(self.trajectory)
        

        path_lengths = self.compute_path_lengths(pos)
        
        ## Add length of path so far to path_lengths
        path_lengths = path_lengths + distance

    
        # FInd magnitude of goals from [0,0]
        V_g1_0 = torch.linalg.norm(self.goals[0])
        V_g2_0 = torch.linalg.norm(self.goals[1])
     

        # Compute weights
        weight_1 = torch.exp(V_g1_0 - path_lengths - magnitude_goal1)
        weight_2 = torch.exp(V_g2_0 - path_lengths - magnitude_goal2)
        

        # Normalize the weights
        weights = torch.stack([weight_1, weight_2], dim=2)
        weights = weights / weights.sum(dim=2).unsqueeze(-1)
        

        return weights
    



    def compute_path_length(self, path):
        
        path_diff = path[1:] - path[:-1]
        path_length = torch.norm(path_diff, dim=1).sum()
        return path_length.item()
    

    def compute_path_lengths(self, paths):


        # Subtract the current position from the rest of the path
        paths = paths - self.position.unsqueeze(0).unsqueeze(0)

        # append row of zeros to the beginning of the path
        paths = torch.cat([torch.zeros_like(paths[0:1]), paths], dim=0)

        # Compute the differences between consecutive points in each path
        path_diffs = torch.diff(paths, dim=0)
        
        # Compute the length of each segment in each path
        segment_lengths = torch.norm(path_diffs, dim=-1)

        # Compute the cumulative sum of segment lengths along the timesteps dimension
        path_lengths = torch.cumsum(segment_lengths, dim=0)

        return path_lengths
    

    def obstacle_cost(self, state):
        

        pos = state[:, :, 0:2]

        # Compute the distances between the future positions of the obstacles and the trajectory samples
        distances = self.compute_distances(pos, self.obstacle_predictions)
        # Print number of elements below collision threshold
        

    
        # Check if distances are below threshold result in a boolean tensor
        collision = distances < self.cfg.costfn.safety_radius
        collision = collision.float()
        # Print number of elements in array above collision threshold
        
        
        # Multiply to get cost
        collision_cost = collision*self.cfg.costfn.collision_cost
        

        # Sum along objects direction
        collision_cost = collision_cost.sum(-1)

        collision_cost = collision_cost.permute(1, 0)


        return collision_cost



    def compute_distances(self, samples, obstacles):


        # Add new axes to match the shapes
        samples = samples.unsqueeze(2)
        obstacles = obstacles.unsqueeze(0)
        

        # Compute the distances using broadcasting
        distances = torch.sqrt(((samples - obstacles)**2).sum(-1))
        # Sum along the samples direction to get the obstacle cost per sample

        return distances
        
        

    def propagate_positions(self, obstacle_array, K):
        # Convert positions and velocities to PyTorch tensors
        if obstacle_array is None:

            # Return tensor full of Zeros
            return torch.zeros((K, 1, 2), device=self.cfg.mppi.device)
            
        positions = torch.tensor([[
            obstacle.position.position.x,
            obstacle.position.position.y
        ] for obstacle in obstacle_array.obstacles], device=self.cfg.mppi.device)

        velocities = torch.tensor([[
            obstacle.velocity.linear.x,
            obstacle.velocity.linear.y
        ] for obstacle in obstacle_array.obstacles], device=self.cfg.mppi.device)

        # Create a new axis for the timesteps and use broadcasting to propagate the positions
        timesteps = torch.arange(K)[:, None, None]
        future_positions = positions + velocities * timesteps*2

        return future_positions

    