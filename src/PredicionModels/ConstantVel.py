import torch



def Constant_velocity_prediction(interface, cfg):

    v = 1
    timestep = 0.2

    # Get position XY and make it tensor
    position =  torch.tensor(interface.state[0:2], device=cfg.mppi.device)
    psi = torch.tensor(interface.state[2])

    displacement = torch.tensor([timestep * v * torch.cos(psi),
                                  timestep * v * torch.sin(psi)], device=cfg.mppi.device)

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
    




