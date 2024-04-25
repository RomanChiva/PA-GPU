import torch
from PredicionModels.utils import *
from PredicionModels.Goal_weights import Compute_Weights_Goals

def pred_model_goal_oriented(state, interface, cfg, goals):

    step = 0.2
    # Get velocity -> V_max
    velocity =  cfg.mppi.u_max[0]*step
    # Get position XY and make it tensor
    position =  torch.tensor([interface.odom_msg.pose.pose.position.x,
                                interface.odom_msg.pose.pose.position.y], device=cfg.mppi.device)
    #### All these steps are to make sure the predictions align, and avoid being 1 timestep ahead#####
    pos = state[:, :, 0:2]
    # Create a tensor with the current position repeated K times
    current_position = position.repeat(pos.shape[1], 1)
    ## Stack this at first index of pos
    pos = torch.cat([current_position.unsqueeze(0), pos], dim=0)
    # Remove the last element of each trajectory to make it a T-1 trajectory
    pos = pos[:-1, :, :]
    ######################################################
    pred_goals = []

    # Loop over each goal
    for i in range(goals.shape[0]):

        # Find vector to goal (Goal[-] shape[1,2] pos shape [T, K, 2])
        vector_goal = goals[i] - pos
        # Find magnitude of vector
        magnitude_goal = torch.linalg.norm(vector_goal, axis=2)
        # Find unit vector
        unit_goal = vector_goal / magnitude_goal.unsqueeze(-1)
        # Prediction in constant velocity for goal
        pred_goal = pos + unit_goal * velocity
        pred_goals.append(pred_goal)

    # Find weights
    weights = Compute_Weights_Goals(pos, goals, interface)

    # Convert all to floar32
    predictions = torch.stack(pred_goals,dim=0)
    weights = weights.float()

    return predictions, weights