import torch, torch.nn.functional as F
import numpy as np, sys, os, json
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler
import gymnasium as gym, minigrid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(__file__))
from step2_train_world_model import (
    ImageEncoder, TransitionModel, LATENT_DIM, N_ACTIONS,
    DEVICE, LR_WM, ENV_NAME
)
from training_utils import train_world_model

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOTAL_STEPS      = 100_000
KL_WEIGHT        = 0.001
DENSE_STEPS      = list(range(5_000, 105_000, 5_000))
CKPT_DIR         = os.path.join(PROJECT_ROOT, "data", "checkpoints", "v3_dense")
EVAL_EPISODES    = 100


def eval_checkpoint(encoder, transit, env, n_episodes=EVAL_EPISODES):
    encoder.eval(); transit.eval()
    zs, dirs, pred_losses = [], [], []

    obs, _ = env.reset(seed=42)
    with torch.no_grad():
        for _ in range(n_episodes * 20):
            img = torch.tensor(obs["image"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            _, mu, _ = encoder(img)

            action = env.action_space.sample()
            next_obs, _, done, _, _ = env.step(action)

            next_img = torch.tensor(next_obs["image"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            _, mu_next, _ = encoder(next_img)
            a_oh = F.one_hot(torch.tensor(action), N_ACTIONS).float().unsqueeze(0).to(DEVICE)
            loss = F.mse_loss(transit(mu, a_oh), mu_next).item()

            zs.append(mu.squeeze(0).cpu().numpy())
            dirs.append(int(env.unwrapped.agent_dir))
            pred_losses.append(loss)
            obs = next_obs if not done else env.reset()[0]

    Z = np.array(zs); y = np.array(dirs)
    Zs = StandardScaler().fit_transform(Z)
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(Zs, y, test_size=0.2,
                                                    random_state=0, stratify=y)
        dir_acc = accuracy_score(y_te,
                  LogisticRegression(max_iter=500).fit(X_tr, y_tr).predict(X_te))
    except Exception:
        dir_acc = 0.25

    return float(np.mean(pred_losses)), float(dir_acc)


def run():
    os.makedirs(CKPT_DIR, exist_ok=True)

    trained = all(os.path.exists(f"{CKPT_DIR}/checkpoint_{s:07d}.pt") for s in DENSE_STEPS)

    if not trained:
        print(f"训练 {len(DENSE_STEPS)} 个 checkpoint（每 5000 步）...")
        encoder   = ImageEncoder(LATENT_DIM).to(DEVICE)
        transit   = TransitionModel(LATENT_DIM, N_ACTIONS).to(DEVICE)

        train_world_model(
            encoder=encoder,
            transition_model=transit,
            n_actions=N_ACTIONS,
            env_name=ENV_NAME,
            total_steps=TOTAL_STEPS,
            kl_weight=KL_WEIGHT,
            lr=LR_WM,
            checkpoint_steps=DENSE_STEPS,
            device=DEVICE,
            seed=0,
            observation_transform=None,
            save_dir=CKPT_DIR,
            progress_desc='Training [dense]',
        )
        print("训练完成")
    else:
        print("Dense checkpoints 已存在，跳过训练")

    print(f"\n=== H6 高密度分析（{len(DENSE_STEPS)} 个数据点）===")
    print(f"{'Step':>8} | {'pred_loss':>10} | {'dir_acc':>8}")
    print("-" * 35)

    env_eval = gym.make(ENV_NAME)
    pred_losses, dir_accs = [], []

    for s in DENSE_STEPS:
        ckpt = torch.load(f"{CKPT_DIR}/checkpoint_{s:07d}.pt", map_location=DEVICE)
        encoder = ImageEncoder(LATENT_DIM).to(DEVICE)
        transit = TransitionModel(LATENT_DIM, N_ACTIONS).to(DEVICE)
        encoder.load_state_dict(ckpt["encoder"])
        transit.load_state_dict(ckpt["transition"])

        pred_loss, dir_acc = eval_checkpoint(encoder, transit, env_eval)
        pred_losses.append(pred_loss)
        dir_accs.append(dir_acc)
        print(f"{s:>8} | {pred_loss:>10.4f} | {dir_acc:>8.4f}")

    env_eval.close()

    r_s, p_s = spearmanr(pred_losses, dir_accs)
    r_p, p_p = pearsonr(pred_losses, dir_accs)

    from scipy.stats import pearsonr as pr
    steps_arr = np.array(DENSE_STEPS).reshape(-1, 1)
    pl_resid = pred_losses - Ridge().fit(steps_arr, pred_losses).predict(steps_arr)
    da_resid = dir_accs - Ridge().fit(steps_arr, dir_accs).predict(steps_arr)
    r_partial, p_partial = spearmanr(pl_resid, da_resid)

    print(f"\n{'='*45}")
    print(f"Spearman r = {r_s:.4f}   p = {p_s:.4f}")
    print(f"Pearson  r = {r_p:.4f}   p = {p_p:.4f}")
    print(f"Partial  r = {r_partial:.4f}   p = {p_partial:.4f}  (detrended by step)")
    print(f"n = {len(DENSE_STEPS)} 个数据点")

    if p_s < 0.05:
        verdict = f"H6 显著！r={r_s:.3f} (p={p_s:.3f})"
    elif p_s < 0.1:
        verdict = f"H6 边界显著 r={r_s:.3f} (p={p_s:.3f})"
    else:
        verdict = f"H6 仍不显著 r={r_s:.3f} (p={p_s:.3f})"
    print(f"结论：{verdict}")

    old_6pt_path = os.path.join(PROJECT_ROOT, "results", "data", "mechanism_correlation.json")
    if os.path.exists(old_6pt_path):
        with open(old_6pt_path) as f:
            old = json.load(f)
        print(f"\n原始 6 点：r={old['spearman_r']:.4f}, p={old['spearman_p']:.4f}")
        print(f"新增 20 点：r={r_s:.4f}, p={p_s:.4f}")

    h6_dense_result = {
        "n_checkpoints": len(DENSE_STEPS),
        "checkpoint_steps": DENSE_STEPS,
        "pred_losses": pred_losses,
        "dir_accs": dir_accs,
        "spearman_r": float(r_s), "spearman_p": float(p_s),
        "pearson_r":  float(r_p), "pearson_p":  float(p_p),
        "partial_spearman_r": float(r_partial), "partial_spearman_p": float(p_partial),
        "verdict": verdict
    }
    out = os.path.join(PROJECT_ROOT, "results", "data", "h6_dense_20pt.json")
    with open(out, "w") as f:
        json.dump(h6_dense_result, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(pred_losses, dir_accs, c=range(len(DENSE_STEPS)),
                    cmap="Blues", s=60, zorder=3)
    z = np.polyfit(pred_losses, dir_accs, 1)
    xline = np.linspace(min(pred_losses), max(pred_losses), 100)
    axes[0].plot(xline, np.poly1d(z)(xline), 'r--', alpha=0.5, label='trend')
    axes[0].set_xlabel("Prediction Loss")
    axes[0].set_ylabel("Direction Accuracy")
    axes[0].set_title(f"H6: Spearman r={r_s:.3f} (p={p_s:.3f})\n"
                       f"Partial r={r_partial:.3f} (p={p_partial:.3f}, detrended)")
    axes[0].axhline(0.25, color='gray', linestyle=':', alpha=0.5, label='random')
    axes[0].legend()

    axes[1].plot(DENSE_STEPS, pred_losses, 'o-', color='#e05c5c',
                 linewidth=1.5, markersize=4, label='Pred Loss')
    ax2 = axes[1].twinx()
    ax2.plot(DENSE_STEPS, dir_accs, 's--', color='#2563cc',
             linewidth=1.5, markersize=4, label='Dir Acc')
    axes[1].set_xlabel("Training Steps"); axes[1].set_ylabel("Prediction Loss", color='#e05c5c')
    ax2.set_ylabel("Direction Accuracy", color='#2563cc')
    axes[1].set_title("Prediction Loss & Direction Accuracy (20 checkpoints)")

    plt.tight_layout()
    fig_out = os.path.join(PROJECT_ROOT, "results", "figures", "h6_dense_20pt.png")
    plt.savefig(fig_out, dpi=150)
    print(f"\nh6_dense_20pt.png 保存完成")
    print(f"h6_dense_20pt.json 保存完成")

    return h6_dense_result


if __name__ == "__main__":
    run()
