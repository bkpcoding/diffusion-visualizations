import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import itertools
import torch.nn.functional as F
import pandas as pd
import seaborn as sns

# Function for generating mode data with observations
def modes_data_with_obs(num_samples):
    modes = [(0.25, 0.25), (0.1, 0.8), (0.75, 0.75)]
    actions = np.zeros((num_samples, 1, 2), dtype=np.float32)
    observations = np.zeros((num_samples, 1), dtype=np.float32)
    for i in range(num_samples):
        mode_index = i % 3
        mode = modes[mode_index]
        actions[i] = np.random.normal(loc=mode, scale=0.05, size=(1, 2))
        observations[i] = mode_index
    actions = np.clip(actions, 0, 1)
    return actions.squeeze(1), observations.squeeze(1)

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim=10, max_length=1000):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.positional_encodings = self._get_positional_encodings()

    def _get_positional_encodings(self):
        pe = torch.zeros(self.max_length, self.embedding_dim)
        position = torch.arange(0, self.max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2).float() * (-math.log(10000.0) / self.embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, time):
        return self.positional_encodings[time, :]

class FiLM(nn.Module):
    def __init__(self, in_features, out_features):
        super(FiLM, self).__init__()
        self.gamma = nn.Linear(in_features, out_features)
        self.beta = nn.Linear(in_features, out_features)

    def forward(self, x, cond):
        gamma = self.gamma(cond)
        beta = self.beta(cond)
        return gamma * x + beta

class ScoreNetwork(nn.Module):
    def __init__(self, data_dim=2, time_dim=2, obs_dim=1, total_timesteps=1000):
        super(ScoreNetwork, self).__init__()
        self.data_dim = data_dim
        self.time_dim = time_dim
        self.obs_dim = obs_dim
        self.positional_embedding = SinusoidalPositionalEmbedding(time_dim, total_timesteps)

        self.fc1 = nn.Linear(data_dim + time_dim, 128)
        self.film1 = FiLM(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.film2 = FiLM(obs_dim, 128)
        self.fc3 = nn.Linear(128, data_dim)

    def forward(self, x, time, obs):
        time_embedding = self.positional_embedding(time)
        x = torch.cat([x, time_embedding], dim=-1)
        x = F.relu(self.film1(self.fc1(x), obs))
        x = F.relu(self.film2(self.fc2(x), obs))
        x = self.fc3(x)
        return x

class DiffusionModel(nn.Module):
    def __init__(self, total_timesteps=1000, data_dim=2, time_dim=10, obs_dim=1, beta_start=0.0001, beta_end=0.02):
        super(DiffusionModel, self).__init__()
        self.total_timesteps = total_timesteps
        self.score_network = ScoreNetwork(data_dim=data_dim, time_dim=time_dim, obs_dim=obs_dim)
        self.betas = torch.linspace(beta_start, beta_end, total_timesteps)
        self.alphas = 1 - self.betas
        self.cumulative_alphas = torch.cumprod(self.alphas, dim=0)
        self.cumulative_betas = 1 - self.cumulative_alphas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(1 / self.alphas_cumprod - 1)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

    def get_variance(self, t):
        if t == 0:
            return 0
        variance = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])
        variance = variance.clip(1e-20)
        return variance

    def predict_noise(self, x, t, obs):
        return self.score_network(x, t, obs)

    def step(self, model_output, timestep, x_t):
        t = timestep
        s1 = self.sqrt_inv_alphas_cumprod[t]
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        pred_original_sample = s1 * x_t - s2 * model_output
        s1 = self.posterior_mean_coef1[t]
        s2 = self.posterior_mean_coef2[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        pred_prev_sample = s1 * pred_original_sample + s2 * x_t
        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = (self.get_variance(t) ** 0.5) * noise
        pred_prev_sample = pred_prev_sample + variance
        return pred_prev_sample

    def add_noise(self, x_start, x_noise, timesteps):
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * x_start + s2 * x_noise

    def sample(self, obs, num_samples=1000, num_timesteps=1000, device='cpu', change_mode_timestep=None):
        sample = torch.randn(num_samples, 2)
        timesteps = list(range(self.total_timesteps))[::-1]
        interemediate_values = torch.empty(num_samples, num_timesteps, 2)
        obs = obs.to(device)
        timestep_change_mode = change_mode_timestep * num_timesteps
        for i, t in enumerate(timesteps):
            if change_mode_timestep is not None and i == timestep_change_mode:
                old_obs = obs
                while torch.any(obs == old_obs):
                    print('Changing mode')
                    obs = torch.randint(0, 3, (num_samples, 1)).to(device).float()
                    print(f'old obs and new obs: {old_obs}, {obs}')
            t = torch.from_numpy(np.repeat(t, num_samples)).long().to(device)
            with torch.no_grad():
                residual = self.predict_noise(sample, t, obs)
            sample = self.step(residual, t[0], sample)
            interemediate_values[:, i, :] = sample
        return sample, interemediate_values, obs

def train(model, data, obs, num_iterations=1000, batch_size=32, learning_rate=1e-4, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    data_tensor = torch.tensor(data, dtype=torch.float32)
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(-1)
    data_loader = torch.utils.data.DataLoader(list(zip(data_tensor, obs_tensor)), batch_size=batch_size, shuffle=True)
    cyclic_data_iterator = itertools.cycle(data_loader)
    losses = []
    for i in tqdm(range(num_iterations)):
        optimizer.zero_grad()
        noise_free_data, obs_batch = next(cyclic_data_iterator)
        noise_free_data, obs_batch = noise_free_data.to(device), obs_batch.to(device)
        time_steps = torch.randint(0, model.total_timesteps, (noise_free_data.shape[0],)).to(device)
        noise = torch.randn(noise_free_data.shape).to(device)
        noisy_data = model.add_noise(noise_free_data, noise, time_steps)
        predicted_noise = model.predict_noise(noisy_data, time_steps, obs_batch)
        loss = loss_fn(predicted_noise, noise)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if i % 10000 == 0:
            samples, _ = model.sample(obs_batch, num_samples=obs_batch.shape[0], num_timesteps=1000, device=device)
            samples = samples.detach().cpu().numpy()
            plt.figure()
            plt.scatter(data[:, 0], data[:, 1], label='True Data', alpha=0.5)
            plt.scatter(samples[:, 0], samples[:, 1], label='Generated Data', alpha=0.5)
            plt.legend()
            plt.show()
    return model, losses

if __name__ == "__main__":
    # Generate data
    num_samples = 3000
    data, obs = modes_data_with_obs(num_samples)
    total_timesteps = 100

    # Instantiate and train the model
    diffusion_model = DiffusionModel(total_timesteps=total_timesteps, data_dim=2, time_dim=10, obs_dim=1)
    trained_model, losses = train(diffusion_model, data, obs, num_iterations=50000, batch_size=32, learning_rate=1e-4, device='cpu')

    # save the model
    torch.save(trained_model.state_dict(), f'models/make_modes_conditioned_{total_timesteps}.pth')

    # Plot training loss
    plt.plot(losses)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()

    # Generate and visualize samples conditioned on each mode
    obs_modes = [0, 1, 2]
    for obs_mode in obs_modes:
        obs_tensor = torch.tensor([obs_mode] * 500, dtype=torch.float32).unsqueeze(-1)
        samples, _ = trained_model.sample(obs_tensor, num_samples=500, num_timesteps=1000, device='cpu')
        samples = samples.detach().cpu().numpy()
        plt.figure()
        plt.scatter(data[:, 0], data[:, 1], label='True Data', alpha=0.5)
        plt.scatter(samples[:, 0], samples[:, 1], label='Generated Data', alpha=0.5)
        plt.legend()
        plt.title(f'Samples conditioned on mode {obs_mode}')
        plt.savefig(f'mode_{obs_mode}_samples.png')
