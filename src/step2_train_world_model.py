import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json, os

from training_utils import train_world_model

LATENT_DIM  = 32
HIDDEN_DIM  = 128
N_ACTIONS   = 7
LR_WM       = 3e-4
TOTAL_STEPS = 100_000
ENV_NAME    = "MiniGrid-Empty-8x8-v0"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR     = os.path.join(PROJECT_ROOT, "data", "checkpoints", "v3_main")

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print(f"使用设备: {DEVICE}  环境: {ENV_NAME}")

CHECKPOINT_STEPS = [1_000, 5_000, 10_000, 25_000, 50_000, 100_000]


class ImageEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu  = nn.Linear(32 * 7 * 7, latent_dim)
        self.fc_std = nn.Linear(32 * 7 * 7, latent_dim)

    def forward(self, img):
        x = img.permute(0, 3, 1, 2).float() / 10.0
        h = self.conv(x)
        mu  = self.fc_mu(h)
        std = F.softplus(self.fc_std(h)) + 1e-4
        z   = mu + std * torch.randn_like(std)
        return z, mu, std


class TransitionModel(nn.Module):
    def __init__(self, latent_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + n_actions, HIDDEN_DIM),
            nn.ELU(),
            nn.Linear(HIDDEN_DIM, latent_dim),
        )

    def forward(self, z, action_onehot):
        return self.net(torch.cat([z, action_onehot], dim=-1))


def train():
    os.makedirs(SAVE_DIR, exist_ok=True)

    encoder  = ImageEncoder(LATENT_DIM).to(DEVICE)
    transit  = TransitionModel(LATENT_DIM, N_ACTIONS).to(DEVICE)

    training_loss_trace = train_world_model(
        encoder=encoder,
        transition_model=transit,
        n_actions=N_ACTIONS,
        env_name=ENV_NAME,
        total_steps=TOTAL_STEPS,
        kl_weight=0.001,
        lr=LR_WM,
        checkpoint_steps=CHECKPOINT_STEPS,
        device=DEVICE,
        seed=0,
        observation_transform=None,
        save_dir=SAVE_DIR,
        progress_desc='Training [β=0.001]',
    )

    out_dir  = os.path.join(PROJECT_ROOT, "02_training")
    os.makedirs(out_dir, exist_ok=True)
    metadata = {
        "version": "v2",
        "env": ENV_NAME,
        "device": DEVICE,
        "exploration": "random",
        "total_steps": TOTAL_STEPS,
        "latent_dim": LATENT_DIM,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "checkpoints": [
            {"step": s, "path": f"{SAVE_DIR}/checkpoint_{s:07d}.pt"}
            for s in CHECKPOINT_STEPS
        ]
    }
    with open(os.path.join(out_dir, "checkpoint_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps_l, losses = zip(*training_loss_trace) if training_loss_trace else ([], [])
    plt.figure(figsize=(10, 4))
    plt.plot(steps_l, losses, linewidth=0.8, color="#e05c5c", alpha=0.7)
    for s in CHECKPOINT_STEPS:
        plt.axvline(x=s, color="#cccccc", linestyle="--", linewidth=0.7)
    plt.xlabel("Training Steps")
    plt.ylabel("World Model Loss (MSE + KL)")
    plt.title(f"World Model Training Loss — {ENV_NAME} + Random Exploration")
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, "results", "figures", "loss_curve.png"), dpi=150)
    plt.close()
    print("loss_curve.png 已保存")
    print("reward_curve.png 跳过（纯随机探索无奖励信号）")

    return encoder, transit


if __name__ == "__main__":
    train()
