import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, entropy
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy.optimize import differential_evolution

# Define the parameters of the GMM
means = np.array([-3, 3])
std_devs = np.array([1, 1])
weights = np.array([0.5, 0.5])

# Generate the GMM
x = np.linspace(-10, 10, 1000)
gmm = np.zeros_like(x)
for mean, std_dev, weight in zip(means, std_devs, weights):
    gmm += weight * norm.pdf(x, mean, std_dev)

# Define the Gaussian with set variance but variable mean
set_variance = 1.0

# Define the KL divergence function
def kl_divergence(mean):
    gaussian = norm.pdf(x, mean, set_variance)
    return entropy(gmm, gaussian)

# Optimize the position of the mean to minimize the KL divergence
result = minimize(kl_divergence, 0.0)  # Start the optimization from mean=0
optimal_mean = result.x[0]

# Generate the optimal Gaussian
optimal_gaussian = norm.pdf(x, optimal_mean, set_variance)
# Define the KL divergence function with switched terms
def kl_divergence_switched(mean):
    gaussian = norm.pdf(x, mean, set_variance)
    return entropy(gaussian, gmm)

# Optimize the position of the mean to minimize the switched KL divergence
result_switched = basinhopping(kl_divergence_switched, 0.0)  # Start the optimization from mean=0
optimal_mean_switched = result_switched.x[0]

# Generate the optimal Gaussian for the switched KL divergence
optimal_gaussian_switched = norm.pdf(x, optimal_mean_switched, set_variance)

# Plot the GMM, the optimal Gaussian, and the optimal Gaussian for the switched KL divergence
plt.plot(x, gmm, label='GMM')
plt.plot(x, optimal_gaussian, label='Optimal Gaussian')
plt.plot(x, optimal_gaussian_switched, label='Optimal Gaussian (switched KL)')
plt.title(f'Optimal mean: {optimal_mean:.2f}, Optimal mean (switched KL): {optimal_mean_switched:.2f}')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.show()