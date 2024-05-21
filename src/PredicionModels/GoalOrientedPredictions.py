import torch
from PredicionModels.utils import *

def goal_oriented_predictions(goals, interface, cfg, return_original=False):

    v = 1.2
    timestep = 1/cfg.freq_prop
    max_w = 20*cfg.mppi.u_max[1]
    
    
    psi = interface.state[2]
    traj = torch.tensor(interface.trajectory, device = goals.device)

    # Get position XY and make it tensor
    position =  traj[-1]
    

    goal_vectors = goals - position
    #print(goal_vectors, 'Goals Vectors agent:' )
    goal_magnitudes = torch.linalg.norm(goal_vectors, axis=1)
    unit_goals = goal_vectors / goal_magnitudes.unsqueeze(-1)
    #print('Unit Goals Raw:', unit_goals, interface.ID)

    # Find heading angle for each goal realtive to our current heading psi
    #angle_goals = torch.atan2(unit_goals[:,1], unit_goals[:,0]) - psi
    # Cap these to be betwen max_w*timestep and -max_w*timestep
    #angle_goals = torch.clamp(angle_goals, -max_w*timestep, max_w*timestep)
    # Add 2*pi to negative angles
    #angle_goals = torch.where(angle_goals < 0, angle_goals + 2*3.14159, angle_goals)
    
    # Create a unit vector pointing in each of the directions
    #unit_goals = torch.stack([torch.cos(angle_goals), torch.sin(angle_goals)], dim=1)
    ##Unit Goals
    #print('Unit Goals from psi:', unit_goals , interface.ID)
    
    # Multiply by velocity to get displacements
    displacement = unit_goals * v * timestep
    
    # Repeat displacement horizon times and multiply horizon index at each timestep
    displacement = displacement.repeat(cfg.mppi.horizon, 1, 1)
    displacement = displacement * torch.arange(1,cfg.mppi.horizon+1, device=cfg.mppi.device).unsqueeze(-1).unsqueeze(-1).float()
    
    # Add displacement to position
    pred_goals_original = position + displacement
    
    #print(pred_goals_original, 'AgentID:', interface.ID)
    
    # Convert to Right Format (n_goals(1),Samples, horizon , nx)
    pred_goals = pred_goals_original.permute(1,0,2)
    pred_goals = pred_goals.unsqueeze(1).repeat(1,cfg.mppi.num_samples, 1, 1)
    
    
    # Weights
    weights_original = observer_weights_current(traj, cfg, goals)
    
    # Tensor of ones with shape (K,T,1)
    weights = weights_original.unsqueeze(0).unsqueeze(0).repeat(cfg.mppi.num_samples, cfg.mppi.horizon, 1)
    if return_original:
        return pred_goals_original, weights_original
    else:
        
        return pred_goals, weights

    
