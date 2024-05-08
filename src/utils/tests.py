import torch
import time



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal


def plot_gmm(means, sigma, weights):
    # Create a grid of points
    x = np.linspace(-5, 20, 100)
    y = np.linspace(-5, 20, 100)
    X, Y = np.meshgrid(x, y)

    covs = [np.eye(2) * sigma for _ in means]
  
    Z = np.zeros_like(X)

    for i, weight in enumerate(weights):
        means_i = means[:,i,:]
        for mean, cov in zip(means_i, covs):
            rv = multivariate_normal(mean=mean, cov=cov)
            Z += weight * rv.pdf(np.dstack((X, Y)))


    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the GMM as a surface plot
    ax.plot_surface(X, Y, Z, cmap='viridis')

    # Set the vertical axis to 1
    ax.set_zlim(0, 1)

    # Show the plot
    plt.show()




num_steps = 20


# Define the start, end, and number of steps
start = np.array([0.5, 0.5])
end = np.array([10, 10])


# Create a trajectory from start to end
trajectory1 = np.linspace(start, end, num_steps)

start2 = np.array([0, 0.5])
end2 = np.array([0, 10])


# Create a trajectory from start to end
trajectory2 = np.linspace(start2, end2, num_steps)

# Comnbine both trajectories in a new dimension to make tarray shape 20x2x2
trajectory = np.stack([trajectory1, trajectory2], axis=1)

# Define the parameters of the two Gaussian components
means = trajectory1
sigma = 2
weights = [1]

plot_gmm(means, sigma, weights)