import torch
import numpy as np
import matplotlib.pyplot as plt

def modes_data(num_samples):
    # Generate data with three modes in the action space
    modes = [[0.25, 0.25], [0.1, 0.8], [0.75, 0.75]]
    obs = np.random.randint(0, 3, size=(num_samples, 1))
    actions = np.zeros((num_samples, 1, 2), dtype=np.float32)
    for i in range(num_samples):
        # mode = obs[i, 0]
        mode = modes[obs[i, 0]]
        actions[i] = np.random.normal(loc=mode, scale=0.05, size=(1, 2))
    actions = np.clip(actions, 0, 1)
    print(obs.shape)
    return actions.squeeze(1)

modes_data = modes_data(1000)
plt.scatter(modes_data[:, 0], modes_data[:, 1])
plt.savefig('modes_data.png')