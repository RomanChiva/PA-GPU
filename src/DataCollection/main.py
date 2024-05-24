#!/usr/bin/python3

import sys
import os
# Get the parent directory of the current script
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# Add the parent directory to the Python path
sys.path.append(parent_dir)


import rospy
from tf.transformations import euler_from_quaternion
import numpy as np
import time
from planner.MPPI_wrapper import MPPI_Wrapper
from hydra.experimental import initialize, compose
import torch
from omegaconf import OmegaConf
from interface.InterfacePlus import JackalInterfacePlus
from geometry_msgs.msg import PoseStamped
from costfn.KL import ObjectiveLegibility
#from costfn.Entropy_Cost import ObjectiveLegibility
from utils.config_store import *
#from utils.plotter import plot_gmm
from PredicionModels.GoalOrientedPredictions import goal_oriented_predictions
import copy
import pickle
import matplotlib.animation as animation
## MPL
import matplotlib.pyplot as plt
import numpy as np



### LOAD THE DATA
with open('raw_data/FW_73_G_20_20.pkl', 'rb') as f:
    data = pickle.load(f)

cfg = data['cfg']
plans = [torch.tensor(x) for x in data['plans']]
trajectory = data['traj']
trajectory = [torch.tensor(x) for x in data['traj']]
samples = data['samples']
samples = [torch.tensor(x) for x in data['samples']]
           
cost_matrix = data['step_cost_matrix']

step_cost = data['step_cost']
step_cost = [torch.tensor(x) for x in step_cost]





def trajectory_plotter(fig, ax, trajectories, goals):

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # Add more colors if you have more goals

    for traj in trajectories:
        ax.plot(traj[:,0], traj[:,1], 'r--')  # Red dashed line for trajectories
        ax.plot(traj[0,0], traj[0,1], 'go')  # Green circle for start points
        ax.plot(traj[-1,0], traj[-1,1], marker='s', color='b')  # Blue square for end points

    for i, goal in enumerate(goals):
        ax.plot(goal[0], goal[1], marker='<', markersize=10 ,color=colors[i % len(colors)])  # Different color for each goal

    ## Add Grid
    ax.grid(True)
    fig.suptitle('Trajectory Plot')
    ax.xaxis.set_label_text('X')
    ax.yaxis.set_label_text('Y')




def plot_plans(fig, ax, plans):

    # Create a color map
    colors = plt.cm.magma(np.linspace(0, 1, len(plans)))

    # Iterate over timesteps
    for i in range(plans.shape[0]):
        # Extract the plan for this timestep
        plan = plans[i]

        #psi = int(plan[0,2]*(180/np.pi))
        psi_mean = int(torch.mean(plan[:,2])*(180/np.pi))
        #print(psi_mean)
        # Plot the plan with a different color
        ax.plot(plan[:, 0], plan[:, 1], color=colors[i], alpha=1)

    ax.grid(True)
   

def show_mppi_sample(samples, run, timestep):
    
    sample = samples[run][timestep]
    # Samples have shape (T, K, nx) (Horizon, N samples, state)
    fig, ax = plt.subplots()
    
    for i in range(sample.shape[0]):
        ax.plot(sample[i,:,0], sample[i,:,1], alpha=0.3)

    ax.grid(True)
    ax.set_title('Sampled Trajectories T={a}, K={b}'.format(a=sample.shape[1], b=sample.shape[0]))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()


def mppi_animation(samples, run, goals):
    fig, ax = plt.subplots()

    def animate(i):
        ax.clear()
        sample = samples[run][i]
        for j in range(sample.shape[0]):
            ax.plot(sample[j,:,0], sample[j,:,1], alpha=0.3)
        # Plot the goals
        for goal in goals:
            ax.plot(goal[0], goal[1], 'go')
        ax.grid(True)
        ax.set_title('Sampled Trajectories T={a}, K={b}'.format(a=sample.shape[1], b=sample.shape[0]))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        # Set x and y limits
        ax.set_xlim([-5, 20])
        ax.set_ylim([-15, 15])

    ani = animation.FuncAnimation(fig, animate, frames=len(samples[run]), interval=200)
    ani.save('animation.mp4', writer='ffmpeg', fps=10)





### Recover Cost Matrices

def horizon_cost(step_cost_matrices, cfg, cost_type, trim_warmstart = False):

    ## Cost type: 0, goal, 1, KL, 2, obstacle
    costs = []

    # Deal with warmstart
    if trim_warmstart:
        step_cost_matrices = step_cost_matrices[trim_warmstart:]

    for step_cost_matrix in step_cost_matrices:

        if cost_type == 0:
            costs.append(step_cost_matrix[0].reshape(cfg.mppi.horizon, cfg.mppi.num_samples))
        elif cost_type == 1:
            costs.append(step_cost_matrix[1].reshape(cfg.mppi.horizon, cfg.mppi.num_samples))
        elif cost_type == 2:
            costs.append(step_cost_matrix[2].reshape(cfg.mppi.horizon, cfg.mppi.num_samples))

    ## Covert to tensor
    costs = torch.stack(costs, dim=0)
    # Sum along dimension 1
    costs = torch.sum(costs, dim=1)
    
    return costs




def importance_sampling(costs):
    # Find minimum along direction of samples 
    min_costs, _ = torch.min(costs, dim=1)
    # Subtract mean costs per timestep
    normalized_costs = costs - min_costs.unsqueeze(1)
    weights = torch.exp(-(1/cfg.mppi.lambda_)*normalized_costs)

    weights = weights / torch.sum(weights, dim=1, keepdim=True)
    print(weights.shape, 'WEIGHTS SHAPE')
    # Count how many are not zero per row
    print(torch.sum(weights != 0, dim=1))
    return weights





def APPROX_optimal(samples, weights):

    ## ASSUME INVERTIBLE DYNAMICS (DO THE AVERAGING IN STATE SPACE NOT ACTION SPACE)

    # Find weighted average of samples at each timestep
    optimal = torch.sum(samples*weights.unsqueeze(-1).unsqueeze(-1), dim=1)

    return optimal


goal_costs = horizon_cost(cost_matrix[0], cfg, 0, trim_warmstart=9)
kl_costs = horizon_cost(cost_matrix[0], cfg, 1, trim_warmstart=9)

goal_weights = importance_sampling(goal_costs)

Goal_optimal = APPROX_optimal(samples[0], goal_weights)

KL_weights = importance_sampling(kl_costs)
KL_optimal = APPROX_optimal(samples[0], KL_weights)






## Make two plots side by side

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Plot goal oct plans on left
plot_plans(fig, axs[0], Goal_optimal)
# Plot KL optimal plans on right
plot_plans(fig, axs[1], KL_optimal)

# Plot the trajectoriy on both
trajectory_plotter(fig, axs[0], trajectory, cfg.costfn.goals)
trajectory_plotter(fig, axs[1], trajectory, cfg.costfn.goals)

plt.show()





def weighted_mppi_visualization(samples, weights, t_step):
    # Visualize all the samples color coded with cost for an MPPI timestep
    fig, ax = plt.subplots()
    # Find the maximum cost
    max_cost = torch.max(weights[t_step])
    # Normalize the costs
    norm = plt.Normalize(0, max_cost)
    # Create a color map
    cmap = plt.cm.hsv
    # Create a scalar mappable
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # Set the scalar mappable to the current axis
    sm.set_array([])
    # Sort the samples and weights based on the weights
    sorted_indices = torch.argsort(weights[t_step])
    sorted_samples = samples[t_step][sorted_indices]
    sorted_weights = weights[t_step][sorted_indices]
    print(sorted_weights)
    # Iterate over sorted samples
    for i in range(sorted_samples.shape[0]):
        # Extract the sample
        sample = sorted_samples[i]
        # Extract the cost
        cost = sorted_weights[i]
        # Plot the sample with a color based on cost
        ax.plot(sample[:, 0], sample[:, 1], color=cmap(norm(cost)))
    # Add a color bar
    fig.colorbar(sm)
    # Set the title
    ax.set_title('Samples at timestep {a}'.format(a=t_step))
    # Set the x and y labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # Show the plot
    plt.grid()
    plt.show()

weighted_mppi_visualization(samples[0], goal_weights, 25)