from train import modes_data_with_obs, DiffusionModel
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
import pandas as pd

plot_once = False
if __name__ == "__main__":
    print("Generating mode sampling animation...")
    # Load the diffusion model
    total_timesteps = 10
    diffusion_model = DiffusionModel(total_timesteps=total_timesteps)
    diffusion_model.load_state_dict(torch.load('models/make_modes_conditioned_10.pth'))

    # Define modes and number of samples
    # modes = [0, 1, 2]
    mode = 0
    num_samples = 3
    num_inference_steps = total_timesteps
    plot_every_n_steps = 1
    data, obs = modes_data_with_obs(num_samples)

    # Create figure
    # fig, axes = plt.subplots(1, len(modes), figsize=(10 * len(modes), 5))
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Generate samples and intermediate values for each mode
    change_mode_timestep = 0.5
    mode_samples = {}
    mode_intermediate_values = {}
    mode_changed_to = {}
    # for mode in modes:
    obs_tensor = torch.tensor([mode] * num_samples, dtype=torch.float32).unsqueeze(-1)
    samples, intermediate_values, new_obs = diffusion_model.sample(obs_tensor,
                    num_samples=num_samples, num_timesteps=num_inference_steps, change_mode_timestep=change_mode_timestep)
    mode_samples[mode] = samples
    mode_intermediate_values[mode] = intermediate_values[:, ::plot_every_n_steps, :]
    mode_changed_to[mode] = new_obs

    # Function to animate the plot
    def animate(i):
        global plot_once
        # Clear the axes
        # for ax in axes:
        ax.clear()
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.axis('off')
        
        # for mode, ax in zip(modes, axes):
        samples = mode_samples[mode]
        intermediate_values = mode_intermediate_values[mode]
        sample_index = i // (num_inference_steps // plot_every_n_steps)
        intermediate_index = i % (num_inference_steps // plot_every_n_steps)
        # Plot the ground truth data
        # if plot_once == False:
        ax.scatter(data[:, 0], data[:, 1], alpha=0.4, color="#67a9cf", label="Ground Truth Data")
        # write on the top of the datapoint which mode it belongs to
        for i in range(len(data)):
            ax.text(data[i][0], data[i][1] + 0.05, f"Mode: {obs[i].item()}", fontsize=8)
        # Plot the intermediate values
        ax.plot(
            intermediate_values[sample_index][:intermediate_index, 0], 
            intermediate_values[sample_index][:intermediate_index, 1], 
            color='#ef8a62', 
            linewidth=2
        )
        # Plot the generated samples
        ax.scatter(
            samples[:sample_index, 0],
            samples[:sample_index, 1],
            color='#ef8a62', 
            label="Generated Samples",
            alpha=0.9
        )
        # Plot a quiver for the last sample
        if intermediate_index > 0:
            ax.quiver(
                intermediate_values[sample_index][intermediate_index - 1, 0],
                intermediate_values[sample_index][intermediate_index - 1, 1],
                intermediate_values[sample_index][intermediate_index, 0] - intermediate_values[sample_index][intermediate_index - 1, 0],
                intermediate_values[sample_index][intermediate_index, 1] - intermediate_values[sample_index][intermediate_index - 1, 1],
                color='#ef8a62',
                linewidth=2
            )
        # right the timestep on the bottom
        ax.text(0, -0.9, f"Timestep: {intermediate_index}", fontsize=10)
        # write on the bottom what the mode changed to
        ax.text(-0.9, -0.9, f"Mode changed to: {mode_changed_to[mode][sample_index].item()} at timestep {change_mode_timestep * total_timesteps}", fontsize=7)
        ax.legend(loc='upper left', fontsize='small')
        ax.set_title(f'Initial Mode: {mode}')
        plot_once = True

    # Create the animation
    print("Creating animation...")
    print("Number of frames:", num_samples * num_inference_steps // plot_every_n_steps)
    anim = FuncAnimation(fig, animate, frames=num_samples * num_inference_steps // plot_every_n_steps, interval=1)
    # Save the animation
    anim.save(f'plots/mode_sampling_animation_{total_timesteps}_{change_mode_timestep}.gif', writer='ffmpeg', fps=1, dpi=300)