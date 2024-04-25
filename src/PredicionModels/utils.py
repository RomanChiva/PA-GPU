import torch
import numpy as np


def compute_path_length(path):
    path_diff = np.diff(path, axis=0)
    path_length = np.linalg.norm(path_diff, axis=1).sum()
    return path_length


def compute_path_lengths(paths, position):


    # Subtract the current position from the rest of the path
    paths = paths - position.unsqueeze(0).unsqueeze(0)

    # append row of zeros to the beginning of the path
    paths = torch.cat([torch.zeros_like(paths[0:1]), paths], dim=0)

    # Compute the differences between consecutive points in each path
    path_diffs = torch.diff(paths, dim=0)
    
    # Compute the length of each segment in each path
    segment_lengths = torch.norm(path_diffs, dim=-1)

    # Compute the cumulative sum of segment lengths along the timesteps dimension
    path_lengths = torch.cumsum(segment_lengths, dim=0)

    return path_lengths


def GenerateSamplesReparametrizationTrick(means, sigma, n_samples):
    # means: tensor of shape (n_distributions, n_components, n_dimensions)
    # covariances: tensor of shape (n_distributions, n_components, n_dimensions, n_dimensions)
    # n_samples: int, number of samples to generate

    # Expand the means tensor to match the number of samples
    expanded_means = means.unsqueeze(0).expand(n_samples, -1, -1, -1)

    # Sample noise from a standard normal distribution
    noise = torch.randn_like(expanded_means)

    # Scale the noise by the Cholesky decomposition and add the mean
    samples = expanded_means + sigma * noise

    return samples



def merge_gmm(samples, weights_tensor, n_samples):
    # Rearrange the dimensions so that modes are the last axis
    samples = samples.permute(2, 3, 1, 0, 4)  # Shape: (n_samples, n_timesteps, n_paths, n_modes, n_dimensions)

    # Sample from the Categorical distribution based on the weights
    select_distribution = torch.distributions.Categorical(weights_tensor).sample((n_samples,))

    # Permute to ensure select_distribution matches the correct dimensions
    select_distribution = select_distribution.permute(1, 2, 0)  # Shape: (n_timesteps, n_paths, n_samples)

    select_distribution = select_distribution.unsqueeze(-1).unsqueeze(-1)
    select_distribution = select_distribution.expand(-1, -1, -1, -1, 2)
    # Apply the gather operation
    # Gather based on the mode index (last dimension)
    gathered_samples = samples.gather(-2, select_distribution)  # Gather along the mode dimension

    # Since `gather` adds an extra dimension, we need to remove it
    gathered_samples = gathered_samples.squeeze(-2)  # Remove the mode dimension after gathering

    # Permute to revert to the expected output order
    gathered_samples = gathered_samples.permute(2, 0, 1, 3)  # Output shape: (n_samples, n_timesteps, n_paths, n_dimensions)

    return gathered_samples

def GenerateSamples(means, sigma, weights, n_samples):

    # For each mode in the menas (dim 0) we generate n_samples using GenerateSamplesReparametrizationTrick
    samples = []
    for i in range(means.shape[0]):
        samples.append(GenerateSamplesReparametrizationTrick(means[i], sigma, n_samples))
    samples = torch.stack(samples, dim=0)

    # Now we have a crazy large 5 dim tensor (n_modes, n_sample, timestep, path, xy)
    
    # At each timestep select a number of samples from each mode based on the weights
    # The end result is a tensor of shape [samples, timestep, path, xy]
    # The weights tensor has shape, [timestep, path, n_modes]
    samples = merge_gmm(samples, weights, n_samples)

    return samples


def score_GMM(samples, means, covariance, weights):
    # means1, means2: tensors of shape (n_distributions, n_components, n_dimensions)
    # covariances1, covariances2: tensors of shape (n_distributions, n_components, n_dimensions, n_dimensions)
    # weights: tensor of shape (timestep, path_sample, 2)


    scores = [multivariate_normal_log_prob(samples, means[i], covariance) for i in range(means.shape[0])]
    # Make a tensor 
    scores = torch.stack(scores, dim=-1)
    
    weights = weights.unsqueeze(0)
    # Weigh the scores based on the weights tensor
    weighted_score = torch.sum(weights * scores, dim=-1)

    # Weigh the scores based on the weights tensor
    #weighted_score = weights[:,:,0] * score1 + weights[:,:,1] * score2

    return weighted_score

def multivariate_normal_log_prob(x, means, sigma):
    # x: tensor of shape (..., n_dimensions)
    # means: tensor of shape (..., n_dimensions)
    # sigma: float, standard deviation of the distributions

    # Convert sigma to a tensor
    sigma = torch.tensor(sigma, dtype=torch.float32)

    # Compute the log probability of a multivariate normal distribution
    diff = x - means
    exponent = -0.5 * (diff / sigma) ** 2
    log_det = -x.shape[-1] * torch.log(sigma)
    log_2pi = -0.5 * x.shape[-1] * torch.log(torch.tensor(2.0 * 3.1415))

    # Sum the log probabilities over the last dimension
    log_prob = torch.sum(exponent + log_det + log_2pi, dim=-1)

    return log_prob


def observer_weights_current(interface, cfg, goals):
    
    # Get position XY and make it tensor
    position =  torch.tensor(interface.state[0:2], device=cfg.mppi.device)
    # Find distance to goals
    distance_goals = torch.norm(goals - position, dim=-1)

    # Length of trajectory so fat
    traj = interface.trajectory
    distance_path = compute_path_length(traj)

    # Find magnitude of goals from [0,0]
    V_g = torch.norm(goals, dim=-1)

    # Compute weights
    weights = torch.exp(V_g - distance_path - distance_goals)

    return weights