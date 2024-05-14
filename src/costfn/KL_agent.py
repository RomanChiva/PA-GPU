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
import time

class ObjectiveLegibility(object):

    def __init__(self, cfg, obstacles, interface, ID):
        # Create two possible goals used in pred_model
        self.cfg = cfg
        self.goal_index = ID
        self.goals = torch.tensor(self.cfg.multi_agent.goals, device=self.cfg.mppi.device).float()
        self.interface = interface
        

    

    # Cost Function for free navigation (Straight to goal)
    def compute_cost(self, state):
       
        
         ## Send all tensors we need to GPU
        self.trajectory = torch.tensor(self.interface.trajectory, device=self.cfg.mppi.device)
        self.position = self.trajectory[-1]
        self.psi = self.interface.state[2].to(self.cfg.mppi.device)#torch.tensor(self.interface.state[2], device=self.cfg.mppi.device)
        obstacles = self.interface.get_state_histories()
        
        # The function already picks a device
        self.obstacle_predictions = self.propagate_positions_constant_v(obstacles, self.cfg.mppi.horizon)
        
        ## The state coming in is a tensor with all the different trajectory samples
        ## Goal Cost
        goal_cost = self.goal_cost(state) 
        obstacle_cost = self.obstacle_cost(state)
        obstacle_cost = obstacle_cost.reshape(-1)

        # KL Cost
        #KL = self.KL_Cost(state)
        #KL = KL.reshape(-1)
        # Clamp KL to always be below the maximum value from goal cost
        #KL = torch.clamp(KL, 0, torch.max(goal_cost))
     

        # Add them
        return goal_cost + obstacle_cost# + self.cfg.costfn.KL_weight*KL

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
        prediction, weights = goal_oriented_predictions(self.goals, self.trajectory, self.psi, self.cfg)
        print(prediction.shape, weights.shape, plan_means.shape)
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
        
        


    def propagate_positions_constant_v(self, state_tensor, K):

        timestep = 0.2

        if state_tensor is None:
            return torch.zeros((K, 1, 2), device=self.cfg.mppi.device)
        
        
        # Extract the X, Y, Orientation, and Velocity from the input tensor
        X, Y, theta, V = state_tensor[:, 0], state_tensor[:, 1], state_tensor[:, 2], state_tensor[:, 3]

        # Create a tensor of time steps from 0 to K
        timesteps = torch.arange(K, device=self.cfg.mppi.device).view(-1, 1)*timestep

        # Calculate the new X and Y for all timesteps at once using broadcasting
        X_new = X + V * torch.cos(theta) * timesteps
        Y_new = Y + V * torch.sin(theta) * timesteps

        # Create the propagated states tensor by stacking the new X, Y, and the constant theta and V
        propagated_states = torch.stack([X_new, Y_new], dim=-1)

       
        return propagated_states

    
