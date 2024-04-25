import torch


def goal_oriented_predictions(interface, cfg):



    v = 1
    timestep = 0.2
    max_w = cfg.mppi.u_max[1]
    goals = torch.tensor(cfg.costfn.goals).float()

    

    # Get position XY and make it tensor
    position =  torch.tensor(interface.state[0:2], device=cfg.mppi.device)
    psi = torch.tensor(interface.state[2])

    goal_vectors = goals - position
    goal_magnitudes = torch.linalg.norm(goal_vectors, axis=1)
    unit_goals = goal_vectors / goal_magnitudes.unsqueeze(-1)

    # Find heading angle for each goal realtive to our current heading psi
    angle_goals = torch.atan2(unit_goals[:,1], unit_goals[:,0]) - psi
    # Cap these to be betwen max_w*timestep and -max_w*timestep
    angle_goals = torch.clamp(angle_goals, -max_w*timestep, max_w*timestep)

    # Create a unit vector pointing in each of the directions
    unit_goals = torch.stack([torch.cos(angle_goals), torch.sin(angle_goals)], dim=1)

    # Multiply by velocity to get displacements
    displacement = unit_goals * v * timestep

    # Repeat displacement horizon times and multiply horizon index at each timestep
    displacement = displacement.repeat(cfg.mppi.horizon, 1)
    displacement = displacement * torch.arange(1,cfg.mppi.horizon+1).unsqueeze(-1).float()
    

    # Add displacement to position
    pred_goals = position + displacement


    # Convert to Right Format (n_goals(1),Samples, horizon , nx)
    pred_goals = pred_goals.unsqueeze(0).repeat(1,cfg.mppi.num_samples, 1, 1)
    
    # Weights
    # Tensor of ones with shape (K,T,1)
    weights = torch.ones(cfg.mppi.num_samples, cfg.mppi.horizon,1, device=cfg.mppi.device)

    return pred_goals, weights

    