import torch
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import minigrid
from torch.optim import Adam
from tqdm import tqdm
import os


def obs_to_tensor(obs, device='cpu'):
    return torch.tensor(obs["image"], dtype=torch.float32).unsqueeze(0).to(device)


def save_checkpoint(encoder, transition_model, step, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        "encoder":    encoder.state_dict(),
        "transition": transition_model.state_dict(),
        "step":       step,
    }, f"{save_dir}/checkpoint_{step:07d}.pt")


def extract_latents_from_encoder(encoder, env, n_samples=3000, seed=0):
    dataset = []
    obs, _ = env.reset(seed=seed)
    for _ in range(n_samples):
        img = obs_to_tensor(obs, device=next(encoder.parameters()).device)
        with torch.no_grad():
            _, mu, _ = encoder(img)
        dataset.append({
            "z":    mu.squeeze(0).cpu().numpy().tolist(),
            "info": {
                "agent_pos": [int(x) for x in env.unwrapped.agent_pos],
                "agent_dir": int(env.unwrapped.agent_dir),
                "carrying":  bool(env.unwrapped.carrying is not None),
            }
        })
        obs, _, done, _, _ = env.step(env.action_space.sample())
        if done:
            obs, _ = env.reset()
    return dataset


def train_world_model(
    encoder,
    transition_model,
    n_actions,
    env_name,
    total_steps=100_000,
    kl_weight=0.001,
    lr=3e-4,
    checkpoint_steps=None,
    device='cpu',
    seed=0,
    observation_transform=None,
    save_dir=None,
    progress_desc='Training',
    loss_callback=None,
):
    if checkpoint_steps is None:
        checkpoint_steps = [1_000, 5_000, 10_000, 25_000, 50_000, 100_000]

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(env_name)
    opt_wm = Adam(list(encoder.parameters()) + list(transition_model.parameters()), lr=lr)

    obs, _ = env.reset(seed=seed)
    step = 0
    training_loss_trace = []

    pbar = tqdm(total=total_steps, desc=progress_desc)

    while step < total_steps:
        img = obs_to_tensor(obs, device)
        if observation_transform is not None:
            img = observation_transform(img)
        z, mu, std = encoder(img)

        action = env.action_space.sample()
        next_obs, _, done, truncated, _ = env.step(action)
        step += 1
        pbar.update(1)

        next_img = obs_to_tensor(next_obs, device)
        if observation_transform is not None:
            next_img = observation_transform(next_img)
        z_next, _, _ = encoder(next_img)

        act_onehot = F.one_hot(torch.tensor(action), n_actions).float().unsqueeze(0).to(device)
        z_next_pred = transition_model(z.detach(), act_onehot)
        transition_mse = F.mse_loss(z_next_pred, z_next.detach())

        kl_divergence = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).mean()
        total_loss = transition_mse + kl_weight * kl_divergence

        opt_wm.zero_grad()
        total_loss.backward()
        opt_wm.step()

        if done or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs

        if step % 100 == 0:
            training_loss_trace.append((step, total_loss.item()))

        if save_dir and step in checkpoint_steps:
            save_checkpoint(encoder, transition_model, step, save_dir)
            pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})

        if loss_callback is not None:
            loss_callback(step, total_loss.item())

    pbar.close()
    env.close()

    return training_loss_trace
