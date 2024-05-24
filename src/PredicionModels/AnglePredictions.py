import torch 
from PredicionModels.utils import *
from PredicionModels.Goal_weights import Compute_Weights_Goals


def Angle_prediction(state, pose, traj, psi, goals,cfg, mode='Simple', angle_limits = False):

    max_w = cfg.mppi.u_max[1]
    timestep = 1/cfg.freq_prop
    print(state.shape, 'State SHape')

    if mode == 'Simple':
        ##Find angles of the goal vectors between pi and -pi 
        goal_vectors = goals - pose
        angle_goals = torch.atan2(goal_vectors[:,1], goal_vectors[:,0])

        if angle_limits:
            # Find heading angle for each goal realtive to our current heading psi
            diff = psi-angle_goals
            # Cap these to be betwen max_w*timestep and -max_w*timestep
            diff = torch.clamp(diff, -max_w*timestep, max_w*timestep)
            # Add 2*pi to negative angles
            #angle_goals = torch.where(angle_goals < 0, angle_goals + 2*3.14159, angle_goals)
            angle_goals = psi + diff

        # Find weights 
        weights = observer_weights_current(traj, cfg, goals)
        print(weights)
        
        # Right Format for each angle (Tensor ones righ shape anc multiplyby angles)
        ones = torch.ones((angle_goals.shape[0], state.shape[0], cfg.mppi.horizon, 1), device=cfg.mppi.device)
        # Multiply with angle_goals in 3rd dimension
        prediction = ones * angle_goals.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        weights = weights.unsqueeze(0).unsqueeze(0).repeat(state.shape[0], cfg.mppi.horizon, 1)

        return prediction, weights

        

    if mode == 'iterative':

        pos = state[:, :, 0:2]
        psi = state[:, :, 2]
        predictions = []
        # Find Goal Angles per timestep for all samples full propagates states
        for goal in goals:
            goal_vectors = goal - pos
            angle_goals = torch.atan2(goal_vectors[:,:,1], goal_vectors[:,:,0])
            

            if angle_limits:
                # Find heading angle for each goal realtive to our current heading psi
                diff = psi-angle_goals
                # Cap these to be betwen max_w*timestep and -max_w*timestep
                diff = torch.clamp(diff, -max_w*timestep, max_w*timestep)
                # Add 2*pi to negative angles
                #angle_goals = torch.where(angle_goals < 0, angle_goals + 2*3.14159, angle_goals)
                angle_goals = psi + diff

            predictions.append(angle_goals)

        # Find weights 
        weights = Compute_Weights_Goals(pos, goals, traj,cfg)
        

        # Right Format for each angle (Tensor ones righ shape anc multiplyby angles)
        predictions = torch.stack(predictions,dim=0).unsqueeze(-1)
        weights = weights.float()
        
        return predictions, weights



        

