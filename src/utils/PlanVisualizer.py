import numpy as np



# Open plans_kl.npy
plans = np.load('/home/roman/ROS/catkin_ws/src/Experiments/src/utils/plans_KL2.npy', allow_pickle=True)
# Plans is np array of trajectories shape 89x20x4

# Plot the plans

import matplotlib.pyplot as plt

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_title('Trajectories')
ax.set_xlabel('X')
ax.set_ylabel('Y')

# Iterate over the first dimension of the tensor
for i in range(plans.shape[0]):
    # Extract the i-th trajectory
    trajectory = plans[i]

    # Plot the trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.5)

plt.grid()
plt.show()