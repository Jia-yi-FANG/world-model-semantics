import torch, torch.nn.functional as F
import numpy as np, sys, os, json
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import gymnasium as gym, minigrid

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(__file__))
from step2_train_world_model import (
    ImageEncoder, TransitionModel, LATENT_DIM, N_ACTIONS, DEVICE,
    LR_WM, TOTAL_STEPS, ENV_NAME, CHECKPOINT_STEPS
)
from training_utils import train_world_model

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASK_RATIO   = 0.5
KL_WEIGHT    = 0.001
CKPT_DIR     = os.path.join(PROJECT_ROOT, "data", "checkpoints", "partial_obs")
LATENT_DIR   = os.path.join(PROJECT_ROOT, "data", "latents", "partial_obs")


def apply_partial_obs(img_tensor, mask_ratio=MASK_RATIO):
    mask = torch.rand_like(img_tensor[:, :, :, 0]) > mask_ratio
    mask = mask.unsqueeze(-1).expand_as(img_tensor)
    return img_tensor * mask.float()


def train_partial_obs():
    os.makedirs(CKPT_DIR, exist_ok=True)
    if all(os.path.exists(f"{CKPT_DIR}/checkpoint_{s:07d}.pt") for s in CHECKPOINT_STEPS):
        print("partial_obs checkpoints 已存在，跳过训练")
        return

    encoder  = ImageEncoder(LATENT_DIM).to(DEVICE)
    transit  = TransitionModel(LATENT_DIM, N_ACTIONS).to(DEVICE)

    train_world_model(
        encoder=encoder,
        transition_model=transit,
        n_actions=N_ACTIONS,
        env_name=ENV_NAME,
        total_steps=TOTAL_STEPS,
        kl_weight=KL_WEIGHT,
        lr=LR_WM,
        checkpoint_steps=CHECKPOINT_STEPS,
        device=DEVICE,
        seed=0,
        observation_transform=lambda img: apply_partial_obs(img, mask_ratio=MASK_RATIO),
        save_dir=CKPT_DIR,
        progress_desc=f'Training [partial_obs {int(MASK_RATIO*100)}%]',
    )
    print(f"训练完成 → {CKPT_DIR}/")


def run_h6_partial():
    os.makedirs(LATENT_DIR, exist_ok=True)
    env = gym.make(ENV_NAME)
    pred_losses, dir_accs = [], []

    print(f"\n=== H6 对照实验 v2（部分可观测 {int(MASK_RATIO*100)}%）===")
    print(f"{'Step':>8} | {'pred_loss':>10} | {'dir_acc':>8}")
    print("-" * 35)

    for s in CHECKPOINT_STEPS:
        ckpt = torch.load(f"{CKPT_DIR}/checkpoint_{s:07d}.pt", map_location=DEVICE)
        encoder = ImageEncoder(LATENT_DIM).to(DEVICE)
        transit = TransitionModel(LATENT_DIM, N_ACTIONS).to(DEVICE)
        encoder.load_state_dict(ckpt["encoder"])
        transit.load_state_dict(ckpt["transition"])
        encoder.eval(); transit.eval()

        zs, dirs, losses = [], [], []
        obs, _ = env.reset(seed=99)
        with torch.no_grad():
            for _ in range(800):
                img = torch.tensor(obs["image"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                img_masked = apply_partial_obs(img)
                _, mu, _ = encoder(img_masked)

                action = env.action_space.sample()
                next_obs, _, done, _, _ = env.step(action)
                next_img        = torch.tensor(next_obs["image"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                next_img_masked = apply_partial_obs(next_img)
                _, mu_next, _   = encoder(next_img_masked)

                a_oh = F.one_hot(torch.tensor(action), N_ACTIONS).float().unsqueeze(0)
                loss = F.mse_loss(transit(mu, a_oh), mu_next).item()

                img_clean = torch.tensor(obs["image"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                _, mu_clean, _ = encoder(img_clean)
                zs.append(mu_clean.squeeze(0).numpy())
                dirs.append(int(env.unwrapped.agent_dir))
                losses.append(loss)
                obs = next_obs if not done else env.reset()[0]

        pred_loss = float(np.mean(losses))
        Z = np.array(zs); y = np.array(dirs)
        Zs = StandardScaler().fit_transform(Z)
        X_tr, X_te, y_tr, y_te = train_test_split(Zs, y, test_size=0.2,
                                                    random_state=0, stratify=y)
        acc = accuracy_score(y_te,
              LogisticRegression(max_iter=1000).fit(X_tr, y_tr).predict(X_te))
        pred_losses.append(pred_loss); dir_accs.append(acc)
        print(f"{s:>8} | {pred_loss:>10.4f} | {acc:>8.4f}")

    env.close()

    r_s, p_s = spearmanr(pred_losses, dir_accs)
    print(f"\nSpearman r = {r_s:.4f}  p = {p_s:.4f}")
    verdict = "H6 阳性" if r_s < -0.4 and p_s < 0.1 else "H6 仍不显著"
    print(f"结论：{verdict}")

    h6_partial_result = {"condition": "partial_obs", "mask_ratio": MASK_RATIO,
              "steps": CHECKPOINT_STEPS, "pred_losses": pred_losses,
              "dir_accs": dir_accs, "spearman_r": float(r_s), "spearman_p": float(p_s),
              "verdict": verdict}

    out = os.path.join(PROJECT_ROOT, "results", "data", "h6_contrast_partial_obs.json")
    with open(out, "w") as f: json.dump(h6_partial_result, f, indent=2)

    print("\n--- 三组 H6 对比 ---")
    for cond, fname in [("基线（确定性）", "mechanism_correlation.json"),
                        ("噪声（sigma=0.3）", "h6_contrast_noisy.json"),
                        ("部分可观测（50%）", "h6_contrast_partial_obs.json")]:
        p = os.path.join(PROJECT_ROOT, "results", "data", fname)
        if os.path.exists(p):
            with open(p) as f: d = json.load(f)
            print(f"  {cond}: r={d['spearman_r']:.3f}, p={d['spearman_p']:.3f}")

    return h6_partial_result


if __name__ == "__main__":
    print("Step 1/2: 训练部分可观测世界模型...")
    train_partial_obs()
    print("\nStep 2/2: H6 分析...")
    run_h6_partial()
