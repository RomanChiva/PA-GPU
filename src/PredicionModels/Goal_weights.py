import torch
from PredicionModels.utils import *

def Compute_Weights_Goals(pos, goals, traj, cfg):
    pos = pos.permute(1,0,2)

    # Initialize a list to store the weights for each goal
    weights_list = []

    # Find distance travelled so far
    distance = compute_path_length(traj)

    path_lengths = compute_path_lengths(pos, torch.tensor(traj[-1]))
    
    # Add length of path so far to path_lengths
    path_lengths = path_lengths + distance


    # Loop over each goal
    for i, goal in enumerate(goals):
        # Find vector to goal
        vector_goal = goal - pos

        # Find magnitude of vector
        magnitude_goal = torch.linalg.norm(vector_goal, axis=2)

        # Find magnitude of goal from [0,0]
        V_g_0 = torch.linalg.norm(goal)

        # Compute weight
        # Raationality parameter 
        r = cfg.costfn.rationality
        weight = torch.exp(r*(V_g_0 - path_lengths - magnitude_goal))
        
        # Add weight to list
        if cfg.costfn.fixed_weights == None:
            weights_list.append(weight)
        else:
            weights_list.append(torch.full_like(weight, cfg.costfn.fixed_weights[i], device=weight.device))


    # Stack the weights along a new dimension
    weights = torch.stack(weights_list, dim=2)

    # Normalize the weights
    weights = weights / weights.sum(dim=2, keepdim=True)
    
    weights = weights.permute(1,0,2)

    return weights


