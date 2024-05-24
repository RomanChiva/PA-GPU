from scipy.stats import norm
import numpy as np

def gaussian_pdf(x, mean, std_dev):
    """
    This function returns the likelihood of a point under a Gaussian distribution.

    Parameters:
    x (float): The point to evaluate.
    mean (float): The mean of the Gaussian distribution.
    std_dev (float): The standard deviation of the Gaussian distribution.

    Returns:
    float: The likelihood of x under the Gaussian distribution.
    """
    return norm.pdf(x, loc=mean, scale=std_dev)



## GMM TESTS

x = 3
weights = [0.3, 0.7]

mode1 = gaussian_pdf(x, -0.5, 0.2)
mode2 = gaussian_pdf(x, 0.5, 0.2) 
res = weights[0] * mode1 + weights[1] * mode2
print(res)
log_res = np.log(res)
ego = gaussian_pdf(x, x, 0.1)
print(ego)
log_ego = np.log(ego)

print(log_res, log_ego)

KL = log_ego - log_res
print(KL)


