import torch
from PredicionModels.utils import *
from PredicionModels.Goal_weights import Compute_Weights_Goals

def RationalAction(state, goals, traj, cfg):

    step = 1/cfg.freq_prop
    # Get velocity -> V_max
    velocity =  0.8*cfg.mppi.u_max[0]*step
    psi_max =   10*cfg.mppi.u_max[1]*step
    # #### All these steps are to make sure the predictions align, and avoid being 1 timestep ahead#####
    pos = state[:, :, 0:2]

   
    pred_goals = []

    # # Loop over each goal
    # for i in range(goals.shape[0]):

    #     # Find vector to goal (Goal[-] shape[1,2] pos shape [T, K, 2])
    #     vector_goal = goals[i] - pos
    #     # Find magnitude of vector
    #     magnitude_goal = torch.linalg.norm(vector_goal, axis=2)
    #     # Find unit vector
    #     unit_goal = vector_goal / magnitude_goal.unsqueeze(-1)
    #     # Find angle to goal
    #     angle_goal = torch.atan2(unit_goal[:, :, 1], unit_goal[:, :, 0])
    #     # Clamp angle to be between -max_w*timestep and max_w*timestep
    #     angle_goal = torch.clamp(angle_goal, -psi_max, psi_max)
    #     # Prediction in constant velocity for goal
    #     pred_goal = pos + torch.stack([velocity * torch.cos(angle_goal), velocity * torch.sin(angle_goal)], dim=2)
        
    #     pred_goals.append(pred_goal)

     # Loop over each goal
    for i in range(goals.shape[0]):

        # Find vector to goal (Goal[-] shape[1,2] pos shape [T, K, 2])
        vector_goal = goals[i] - pos
        # Find magnitude of vector
        magnitude_goal = torch.linalg.norm(vector_goal, axis=2)
        # Find unit vector
        unit_goal = vector_goal / magnitude_goal.unsqueeze(-1)
        # Prediction in constant velocity for goal
        pred_goal = pos + velocity * unit_goal
        
        pred_goals.append(pred_goal)

    # Find weights
    weights = Compute_Weights_Goals(pos, goals, traj,cfg)

    # Convert all to floar32
    predictions = torch.stack(pred_goals,dim=0)
    weights = weights.float()

    return predictions, weights