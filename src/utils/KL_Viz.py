import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, entropy
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy.optimize import differential_evolution


def KL_Visualizer(sigma_pred, sigma_plan, pred_means, weights):

    means = np.array(pred_means)
    std_devs = np.array([sigma_pred, sigma_pred])
    weights = np.array(weights)

    # Generate the GMM
    x = np.linspace(-np.pi, np.pi, 1000)
    gmm = np.zeros_like(x)
    for mean, std_dev, weight in zip(means, std_devs, weights):
        gmm += weight * norm.pdf(x, mean, std_dev)

    # Define the KL divergence function
    def kl_divergence(mean):
        gaussian = norm.pdf(x, mean, sigma_plan)
        return entropy(gaussian, gmm)

    # Optimize the position of the mean to minimize the KL divergence
    result = minimize(kl_divergence, 0.0)  # Start the optimization from mean=0
    optimal_mean = result.x[0]

    # Generate the optimal Gaussian
    optimal_gaussian = norm.pdf(x, optimal_mean, sigma_plan)

    # Define the KL divergence function with switched terms
    def kl_divergence_switched(mean):
        gaussian = norm.pdf(x, mean, sigma_plan)
        return entropy(gmm, gaussian)

    # Optimize the position of the mean to minimize the switched KL divergence
    result_switched = basinhopping(kl_divergence_switched, 0.0)  # Start the optimization from mean=0
    optimal_mean_switched = result_switched.x[0]

    # Generate the optimal Gaussian for the switched KL divergence
    optimal_gaussian_switched = norm.pdf(x, optimal_mean_switched, sigma_plan)


    ## PRINT OUT ENTROPIES
    print('KL_Switch:',(entropy(gmm, optimal_gaussian_switched)))
    print('KL:',(entropy(optimal_gaussian, gmm)))

    # Plot the GMM, the optimal Gaussian, and the optimal Gaussian for the switched KL divergence
    plt.plot(x, gmm, label='GMM')
    plt.plot(x, optimal_gaussian, label='Optimal Gaussian')
    plt.plot(x, optimal_gaussian_switched, label='Optimal Gaussian (switched KL)')
    plt.title(f'Optimal mean: {optimal_mean:.2f}, Optimal mean (switched KL): {optimal_mean_switched:.2f}')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    #### PARAMS
    sigma_plan = 0.05
    sigma_pred = 0.2

    # PRED MEANS
    means = [-0.7, 0.7]
    weights = [0.49, 0.51]
    KL_Visualizer(sigma_pred, sigma_plan, means, weights)