import torch
from PredicionModels.ConstantVel import Constant_velocity_prediction
from PredicionModels.utils import *



def local_collision_avoidance(state, goal, obstacle_prediction, cfg, pred_type='angle'): 

    ## Check Collisions
    print('DID WE GET HERE?')
    # FInd ego path and match dimensions
    self_path = Constant_velocity_prediction(state, cfg, return_original=True).unsqueeze(1) # SHape [horizon, 2]
    distances = torch.norm(self_path - obstacle_prediction, dim=-1) # Shape [Horizon, N_obstacles]
    collision = distances < cfg.costfn.safety_radius

    
    if collision.sum() == 0: 
        prediction = straight_path_gen(state[0:2], goal, state[-1], 1/cfg.freq_prop, cfg.mppi.horizon)
        print(prediction)
        prediction = prediction.unsqueeze(0).repeat(cfg.mppi.num_samples, 1, 1)
        weights = weights = torch.ones(cfg.mppi.num_samples, cfg.mppi.horizon,1, device=cfg.mppi.device)
        print('NO COLLISIONS')
        return prediction.unsqueeze(0), weights
    
    ## Pick First One and consttruct interaction goals left/right
    # Find earliest collision with each of the agents
    earliest_collision = find_earliest_collision(collision)
    # Position of first collision: 
    agent_index = torch.argmin(earliest_collision)
    collision_index = earliest_collision[agent_index]
    collision_position = obstacle_prediction[collision_index, agent_index]


    ## Construct paths (Angle or position)

    goal_unit = (goal - state[0:2]) / torch.norm(goal - state[0:2])
    # Perpendicular to goal unit
    goal_perpendicular = torch.tensor([-goal_unit[1], goal_unit[0]], device=goal.device)

    avoid_right = collision_position + cfg.costfn.safety_radius * goal_perpendicular
    avoid_left = collision_position - cfg.costfn.safety_radius * goal_perpendicular
    print(avoid_right, avoid_left, 'Avoid Right and Left')
    if pred_type == 'angle':
        pass
        

    elif pred_type == 'position':

        pred_left = straight_path_gen(state[0:2], avoid_left, state[-1], 1/cfg.freq_prop, cfg.mppi.horizon)
        pred_right = straight_path_gen(state[0:2], avoid_right, state[-1], 1/cfg.freq_prop, cfg.mppi.horizon)

        ### Repeat each path n_samples times
        pred_left = pred_left.unsqueeze(0).repeat(cfg.mppi.num_samples, 1, 1)
        pred_right = pred_right.unsqueeze(0).repeat(cfg.mppi.num_samples, 1, 1)

        # Stack them along 0 dimension
        prediction = torch.stack([pred_left, pred_right], dim=0)
        

    ## Assign weights 
    weights = [0,1]
    weights = torch.tensor(weights, device=goal.device)
    weights = weights.unsqueeze(0).unsqueeze(0).repeat(cfg.mppi.num_samples, cfg.mppi.horizon, 1)

    return prediction, weights

    ## return prediction and weights 



    






    pass