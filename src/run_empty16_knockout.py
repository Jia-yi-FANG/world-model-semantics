import os, sys, json
import numpy as np
from scipy import stats

import torch, torch.nn as nn
import gymnasium as gym, minigrid
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from step2_train_world_model import (
    ImageEncoder, TransitionModel, LATENT_DIM, N_ACTIONS,
    DEVICE, LR_WM
)
from training_utils import train_world_model

ENV_NAME         = "MiniGrid-Empty-16x16-v0"
BETA_STABLE      = 0.001
BETA_COLLAPSE    = 0.1
TOTAL_STEPS      = 100_000
CHECKPOINT_STEPS = [5_000, 10_000, 25_000, 50_000, 75_000, 100_000]
SEED             = 42

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_OUT  = os.path.join(PROJECT_ROOT, 'results', 'figures', 'empty16_knockout.png')
DATA_OUT = os.path.join(PROJECT_ROOT, 'results', 'data',    'empty16_knockout.json')


def eval_metrics(encoder, n_ep=150):
    env = gym.make(ENV_NAME)
    lats, dirs, xs, ys = [], [], [], []
    for _ in range(n_ep):
        obs, _ = env.reset()
        done = False
        while not done:
            img = torch.tensor(obs['image'], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                _, mu, _ = encoder(img)
            lats.append(mu.cpu().numpy()[0])
            dirs.append(int(env.unwrapped.agent_dir))
            p = env.unwrapped.agent_pos
            xs.append(int(p[0])); ys.append(int(p[1]))
            obs, _, t, tr, _ = env.step(env.action_space.sample())
            done = t or tr
    env.close()
    L, D = np.array(lats), np.array(dirs)
    X_tr, X_te, y_tr, y_te = train_test_split(L, D, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=500).fit(X_tr, y_tr)
    dir_acc = float(clf.score(X_te, y_te))
    reg = Ridge().fit(X_tr, np.array(ys)[:len(X_tr)])
    y_r2 = float(reg.score(X_te, np.array(ys)[len(X_tr):]))
    samp = L[::max(1,len(L)//200)][:200]
    pw = float(np.mean([np.linalg.norm(samp[i]-samp[j])
                        for i in range(len(samp))
                        for j in range(i+1,min(i+10,len(samp)))])) if len(samp)>1 else 0.
    return dir_acc, y_r2, pw


def train_run(beta, label):
    print(f"\n{'='*55}")
    print(f"Training β={beta} ({label})  env={ENV_NAME}")
    print(f"{'='*55}")

    ckpt_dir = os.path.join(PROJECT_ROOT, "data", "checkpoints",
                            f"empty16_{'stable' if beta < 0.01 else 'collapse'}")

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
        progress_desc=f'Empty16 β={beta}',
    )

    checkpoint_records = {k:[] for k in ['steps','pred_loss','dir_acc','y_r2','pairwise']}
    for s in CHECKPOINT_STEPS:
        ckpt = torch.load(f"{ckpt_dir}/checkpoint_{s:07d}.pt", map_location=DEVICE)
        encoder.load_state_dict(ckpt["encoder"])
        da, yr, pw = eval_metrics(encoder)
        checkpoint_records['steps'].append(s)
        checkpoint_records['pred_loss'].append(0.0)
        checkpoint_records['dir_acc'].append(da)
        checkpoint_records['y_r2'].append(yr)
        checkpoint_records['pairwise'].append(pw)
        coll = pw < 0.005
        print(f"  step {s:>7,}: dir={da:.3f} y_r2={yr:.3f} dist={pw:.4f}"
              f"{' [COLLAPSED]' if coll else ''}")

    return checkpoint_records


def plot(stable, collapse):
    BLUE, RED = '#2166AC', '#D6604D'
    fig, axes = plt.subplots(1, 3, figsize=(12,4))
    fig.suptitle(f'Double Knockout Replication — {ENV_NAME}', fontsize=13, fontweight='bold')
    ks = [s/1000 for s in stable['steps']]
    kc = [s/1000 for s in collapse['steps']]
    titles = ['(a) Direction Accuracy', '(b) Y-Position $R^2$', '(c) Latent Pairwise Distance']
    keys   = ['dir_acc', 'y_r2', 'pairwise']
    hlines = [0.25, 0.0, 0.0]
    ylabels= ['Direction Accuracy', r'Y-Position $R^2$', 'Mean Pairwise Distance']
    for ax, title, key, hl, yl in zip(axes, titles, keys, hlines, ylabels):
        ax.plot(ks, stable[key],  color=BLUE, marker='o', label=r'$\beta=0.001$ (stable)')
        ax.plot(kc, collapse[key], color=RED, marker='s', linestyle='--', label=r'$\beta=0.1$ (collapse)')
        ax.axhline(hl, color='gray', linestyle=':', linewidth=0.8)
        ax.set_xlabel('Training Steps (k)'); ax.set_ylabel(yl)
        ax.set_title(title); ax.legend(fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
    plt.savefig(FIG_OUT, dpi=300, bbox_inches='tight')
    print(f"\nFigure: {FIG_OUT}")


def main():
    stable   = train_run(BETA_STABLE,   'stable')
    collapse = train_run(BETA_COLLAPSE, 'collapse')
    fs, fc = {k:v[-1] for k,v in stable.items()  if k!='steps'}, \
             {k:v[-1] for k,v in collapse.items() if k!='steps'}
    print(f"\n=== FINAL ({ENV_NAME}) ===")
    print(f"β=0.001: dir={fs['dir_acc']:.3f}  y_r2={fs['y_r2']:.3f}  dist={fs['pairwise']:.4f}")
    print(f"β=0.1:   dir={fc['dir_acc']:.3f}  y_r2={fc['y_r2']:.3f}  dist={fc['pairwise']:.4f}")
    knockout = (fc['pairwise']<0.005) and (fc['dir_acc']<0.35) and (fs['dir_acc']>0.40)
    print(f"Double Knockout replicated: {'YES' if knockout else 'NO'}")
    out = {'env':ENV_NAME,'stable':stable,'collapse':collapse,
           'knockout_replicated':bool(knockout)}
    os.makedirs(os.path.dirname(DATA_OUT), exist_ok=True)
    with open(DATA_OUT,'w') as f: json.dump(out,f,indent=2)
    plot(stable, collapse)
    print(f"Data: {DATA_OUT}")


if __name__ == '__main__':
    main()
