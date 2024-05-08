#!/usr/bin/python3

import numpy as np
from scipy.interpolate import CubicSpline
from utils.cubicspline import CubicSpline2D
import torch
from sklearn.mixture import GaussianMixture
import time
from PredicionModels.utils import *
from PredicionModels.RationalAction import RationalAction
from PredicionModels.ConstantVel import Constant_velocity_prediction
from PredicionModels.GoalOrientedPredictions import goal_oriented_predictions

class ObjectiveLegibility(object):

    def __init__(self, cfg, obstacles, interface):
        # Create two possible goals used in pred_model
        self.cfg = cfg
        self.goal_index = cfg.costfn.goal_index
        self.goals = torch.tensor(self.cfg.costfn.goals).float()
        self.v_ref = torch.tensor(cfg.v_ref, device=cfg.mppi.device)
        self.x_ref = np.array([0, 3, 6, 9, 12, 15, 18, 21, 30, 34, 38])
        self.y_ref = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.reference_spline = CubicSpline2D(self.x_ref, self.y_ref)
        self.obstacles = obstacles
        self.interface = interface
        

    

    # Cost Function for free navigation (Straight to goal)
    def compute_cost(self, state):
        ## The state coming in is a tensor with all the different trajectory samples
        ## Goal Cost
        goal_cost = self.goal_cost(state) 
        obstacle_cost = self.obstacle_cost(state)
        obstacle_cost = obstacle_cost.reshape(-1)

        # KL Cost
        KL = self.KL_Cost(state)
        KL = KL.reshape(-1)
        # Clamp KL to always be below the maximum value from goal cost
        KL = torch.clamp(KL, 0, torch.max(goal_cost))
     

        # Add them
        return goal_cost + obstacle_cost + 2*KL

    def goal_cost(self, state):
        

        state_goal = state.permute(1, 0, 2)
        # Now reshape to (T*K, nx)
        state_goal = state_goal.reshape(-1, self.cfg.nx)
        pos_goal = state_goal[:, 0:2]
        goal_cost = torch.linalg.norm(pos_goal - self.goals[self.goal_index], axis=1)

        return goal_cost
    




    def KL_Cost(self, state):

        ## Specify Distributions
        plan_means = state[:, :, 0:2]
        prediction, weights = RationalAction(state, self.interface, self.cfg)

        # Generate Samples
        samples = GenerateSamplesReparametrizationTrick(plan_means, self.cfg.costfn.sigma_plan, self.cfg.costfn.monte_carlo_samples)
        
        # Score
        score_pred = score_GMM(samples, prediction, self.cfg.costfn.sigma_pred, weights)
        score_plan = multivariate_normal_log_prob(samples, plan_means, self.cfg.costfn.sigma_plan)
        # Compute KL Divergence
        kl_div = torch.mean(score_plan - score_pred, dim=0)

        kl_div = kl_div.permute(1, 0)
        
        return kl_div



    def KL_Cost_reverse(self, state):

        ## Specify Distributions
        plan_means = state[:, :, 0:2] # Get only the positions from state    
        prediction, weights = RationalAction(state, self.interface, self.cfg)
        # Generate Samples

        samples_pred = GenerateSamples(prediction, self.cfg.costfn.sigma_pred, weights, self.cfg.costfn.monte_carlo_samples)
        
        # Score
        score_pred = score_GMM(samples_pred, prediction, self.cfg.costfn.sigma_pred, weights)
        score_plan = multivariate_normal_log_prob(samples_pred, plan_means, self.cfg.costfn.sigma_plan)
        # Compute KL Divergence
        kl_div = torch.mean(score_pred - score_plan, dim=0)


        kl_div = kl_div.permute(1, 0)
        return kl_div
    
    # Reverse Order
    


    def obstacle_cost(self, state):
        

        pos = state[:, :, 0:2]
        horizon = pos.shape[1]
        # Propagate the positions of the obstacles K timesteps into the future
        obstacles = self.interface.obstacles
        future_positions = self.propagate_positions(obstacles, horizon)
        
        # Compute the distances between the future positions of the obstacles and the trajectory samples
        distances = self.compute_distances(pos, future_positions)
        # Print number of elements below collision threshold
        

    
        # Check if distances are below threshold result in a boolean tensor
        collision = distances < self.cfg.costfn.safety_radius
        collision = collision.float()
        # Print number of elements in array above collision threshold
        
        
        # Multiply to get cost
        collision_cost = collision*self.cfg.costfn.collision_cost
        

        # Sum along objects direction
        collision_cost = collision_cost.sum(-1)

        # Sum along the timesteps direction
        collision_cost_2 = collision_cost.sum(-1)
        # Show the number of elements greater than 0
        # Switch the dimensions to match the shape of the state tensor
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
            return torch.zeros((K, 1, 2))
            
        positions = torch.tensor([[
            obstacle.position.position.x,
            obstacle.position.position.y
        ] for obstacle in obstacle_array.obstacles])

        velocities = torch.tensor([[
            obstacle.velocity.linear.x,
            obstacle.velocity.linear.y
        ] for obstacle in obstacle_array.obstacles])

        # Create a new axis for the timesteps and use broadcasting to propagate the positions
        timesteps = torch.arange(K)[:, None, None]
        future_positions = positions + velocities * timesteps*2

        return future_positions
        





    