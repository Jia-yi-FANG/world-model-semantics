import os, sys, json
import numpy as np
from scipy import stats

import torch
import torch.nn as nn
import gymnasium as gym
import minigrid
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from step2_train_world_model import (
    ImageEncoder, TransitionModel, LATENT_DIM, N_ACTIONS,
    DEVICE, LR_WM
)
from training_utils import train_world_model

ENV_NAME      = "MiniGrid-FourRooms-v0"
BETA_COLLAPSE = 0.1
BETA_STABLE   = 0.001
TOTAL_STEPS   = 100_000
CHECKPOINT_STEPS = [5_000, 10_000, 25_000, 50_000, 75_000, 100_000]
EVAL_EPISODES = 150
SEED          = 42

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR      = os.path.join(PROJECT_ROOT, 'results')
FIG_OUT      = os.path.join(OUT_DIR, 'figures', 'fourrooms_knockout.png')
DATA_OUT     = os.path.join(OUT_DIR, 'data', 'fourrooms_knockout.json')


def collect_eval_data(encoder, n_episodes=EVAL_EPISODES):
    env = gym.make(ENV_NAME)
    latents, dirs, xs, ys = [], [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            img = torch.tensor(obs['image'], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                _, mu, _ = encoder(img)
            latents.append(mu.cpu().numpy()[0])
            dirs.append(int(env.unwrapped.agent_dir))
            pos = env.unwrapped.agent_pos
            xs.append(int(pos[0])); ys.append(int(pos[1]))
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            done = term or trunc
    env.close()
    return np.array(latents), np.array(dirs), np.array(xs), np.array(ys)


def eval_metrics(encoder):
    latents, dirs, xs, ys = collect_eval_data(encoder)
    X_tr, X_te, y_tr, y_te = train_test_split(latents, dirs, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=500)
    clf.fit(X_tr, y_tr)
    dir_acc = float(clf.score(X_te, y_te))
    reg = Ridge()
    reg.fit(X_tr, ys[:len(X_tr)])
    y_r2 = float(reg.score(X_te, ys[len(X_tr):]))
    sample = latents[::max(1, len(latents)//200)][:200]
    diffs = []
    for i in range(len(sample)):
        for j in range(i+1, min(i+10, len(sample))):
            diffs.append(np.linalg.norm(sample[i] - sample[j]))
    pairwise = float(np.mean(diffs)) if diffs else 0.0
    return dir_acc, y_r2, pairwise


def train_run(beta, label):
    print(f"\n{'='*50}")
    print(f"Training β={beta} ({label}) in {ENV_NAME}")
    print(f"{'='*50}")

    ckpt_dir = os.path.join(PROJECT_ROOT, "data", "checkpoints",
                            f"fourrooms_{'stable' if beta < 0.01 else 'collapse'}")

    encoder  = ImageEncoder(LATENT_DIM).to(DEVICE)
    transit  = TransitionModel(LATENT_DIM, N_ACTIONS).to(DEVICE)

    train_world_model(
        encoder=encoder,
        transition_model=transit,
        n_actions=N_ACTIONS,
        env_name=ENV_NAME,
        total_steps=TOTAL_STEPS,
        kl_weight=beta,
        lr=LR_WM,
        checkpoint_steps=CHECKPOINT_STEPS,
        device=DEVICE,
        seed=SEED,
        observation_transform=None,
        save_dir=ckpt_dir,
        progress_desc=f'FourRooms β={beta}',
    )

    records = {'steps': [], 'pred_loss': [], 'dir_acc': [], 'y_r2': [], 'pairwise': []}
    for s in CHECKPOINT_STEPS:
        ckpt = torch.load(f"{ckpt_dir}/checkpoint_{s:07d}.pt", map_location=DEVICE)
        encoder.load_state_dict(ckpt["encoder"])
        dir_acc, y_r2, pairwise = eval_metrics(encoder)
        records['steps'].append(s)
        records['pred_loss'].append(0.0)
        records['dir_acc'].append(dir_acc)
        records['y_r2'].append(y_r2)
        records['pairwise'].append(pairwise)
        collapsed = pairwise < 0.005
        print(f"  step {s:>6,}: dir={dir_acc:.3f} y_r2={y_r2:.3f} "
              f"dist={pairwise:.4f}"
              f"{' [COLLAPSED]' if collapsed else ''}")

    return records


def plot_results(stable, collapse):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f'Double Knockout — {ENV_NAME}', fontsize=13, fontweight='bold')

    BLUE = '#2166AC'; RED = '#D6604D'
    steps_s = [s/1000 for s in stable['steps']]
    steps_c = [s/1000 for s in collapse['steps']]

    ax = axes[0]
    ax.plot(steps_s, stable['dir_acc'],  color=BLUE, marker='o', label=r'$\beta=0.001$ (stable)')
    ax.plot(steps_c, collapse['dir_acc'], color=RED,  marker='s', linestyle='--', label=r'$\beta=0.1$ (collapse)')
    ax.axhline(0.25, color='gray', linestyle=':', label='Chance (25%)')
    ax.set_xlabel('Training Steps (k)'); ax.set_ylabel('Direction Accuracy')
    ax.set_title('(a) Direction Accuracy')
    ax.legend(fontsize=8); ax.set_ylim(0.1, 0.85)

    ax = axes[1]
    ax.plot(steps_s, stable['y_r2'],   color=BLUE, marker='o', label=r'$\beta=0.001$')
    ax.plot(steps_c, collapse['y_r2'], color=RED,  marker='s', linestyle='--', label=r'$\beta=0.1$')
    ax.axhline(0.0, color='gray', linestyle=':', label='Random baseline')
    ax.set_xlabel('Training Steps (k)'); ax.set_ylabel(r'Y-Position $R^2$')
    ax.set_title(r'(b) Y-Position $R^2$')
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(steps_s, stable['pairwise'],   color=BLUE, marker='o', label=r'$\beta=0.001$')
    ax.plot(steps_c, collapse['pairwise'], color=RED,  marker='s', linestyle='--', label=r'$\beta=0.1$')
    ax.axhline(0.0, color='gray', linestyle=':')
    ax.set_xlabel('Training Steps (k)'); ax.set_ylabel('Mean Pairwise Distance')
    ax.set_title('(c) Latent Pairwise Distance')
    ax.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
    plt.savefig(FIG_OUT, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved: {FIG_OUT}")


def main():
    stable  = train_run(BETA_STABLE,   'stable')
    collapse = train_run(BETA_COLLAPSE, 'collapse')

    final_s = {k: v[-1] for k, v in stable.items()  if k != 'steps'}
    final_c = {k: v[-1] for k, v in collapse.items() if k != 'steps'}
    print(f"\n=== FINAL RESULTS ({ENV_NAME}) ===")
    print(f"β=0.001: dir={final_s['dir_acc']:.3f}, y_r2={final_s['y_r2']:.3f}, dist={final_s['pairwise']:.4f}")
    print(f"β=0.1:   dir={final_c['dir_acc']:.3f}, y_r2={final_c['y_r2']:.3f}, dist={final_c['pairwise']:.4f}")
    collapsed = final_c['pairwise'] < 0.005
    knockout  = collapsed and (final_c['dir_acc'] < 0.35) and (final_s['dir_acc'] > 0.45)
    print(f"\nDouble Knockout replicated: {'YES ✓' if knockout else 'PARTIAL / NO'}")

    out = {
        'env': ENV_NAME,
        'stable_beta': BETA_STABLE,
        'collapse_beta': BETA_COLLAPSE,
        'stable':  stable,
        'collapse': collapse,
        'knockout_replicated': bool(knockout),
    }
    os.makedirs(os.path.dirname(DATA_OUT), exist_ok=True)
    with open(DATA_OUT, 'w') as f:
        json.dump(out, f, indent=2)

    plot_results(stable, collapse)
    print(f"Data saved: {DATA_OUT}")


if __name__ == '__main__':
    main()
