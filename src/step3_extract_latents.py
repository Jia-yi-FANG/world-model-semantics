import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import minigrid
import json
import os
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(__file__))
from step2_train_world_model import ImageEncoder, LATENT_DIM, N_ACTIONS, DEVICE


def load_checkpoint(checkpoint_path):
    encoder = ImageEncoder(LATENT_DIM).to(DEVICE)
    ckpt    = torch.load(checkpoint_path, map_location=DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    return encoder, ckpt["step"]


def extract_latents(encoder, env, n_episodes=200):
    dataset = []

    for ep in tqdm(range(n_episodes), desc="提取隐状态", leave=False):
        obs, _ = env.reset(seed=ep)
        done   = False

        while not done:
            img      = torch.tensor(obs["image"],
                                    dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                z, mu, _ = encoder(img)

            z_vec = mu.squeeze(0).cpu().numpy().tolist()

            dataset.append({
                "z":    z_vec,
                "info": {
                    "agent_pos": [int(x) for x in env.unwrapped.agent_pos],
                    "agent_dir": int(env.unwrapped.agent_dir),
                    "carrying":  bool(env.unwrapped.carrying is not None),
                }
            })

            action = env.action_space.sample()
            obs, _, done, truncated, _ = env.step(action)
            if truncated:
                break

    return dataset


def extract_all_checkpoints(checkpoint_dir=None, output_dir=None):
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(PROJECT_ROOT, "data", "checkpoints", "v3_main")
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "latents", "v3_main")
    os.makedirs(output_dir, exist_ok=True)
    env = gym.make("MiniGrid-Empty-8x8-v0")

    ckpt_files = sorted([
        f for f in os.listdir(checkpoint_dir)
        if f.endswith(".pt")
    ])

    print(f"找到 {len(ckpt_files)} 个 checkpoints")

    for ckpt_file in ckpt_files:
        ckpt_path   = os.path.join(checkpoint_dir, ckpt_file)
        encoder, step = load_checkpoint(ckpt_path)

        print(f"\n处理 checkpoint: step={step}")
        dataset = extract_latents(encoder, env, n_episodes=200)

        out_path = os.path.join(output_dir, f"latents_{step:07d}.json")
        with open(out_path, "w") as f:
            json.dump(dataset, f)

        print(f"  保存 {len(dataset)} 条记录 → {out_path}")

    env.close()
    print("\n全部提取完成。")


if __name__ == "__main__":
    extract_all_checkpoints()
