import torch, torch.nn.functional as F
import numpy as np, sys, os, json
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import gymnasium as gym, minigrid

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(__file__))
from step2_train_world_model import (
    ImageEncoder, TransitionModel, LATENT_DIM, N_ACTIONS, DEVICE,
    LR_WM, TOTAL_STEPS, ENV_NAME, CHECKPOINT_STEPS
)
from training_utils import train_world_model, extract_latents_from_encoder

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOISE_SIGMA  = 0.3
KL_WEIGHT    = 0.001
CKPT_DIR     = os.path.join(PROJECT_ROOT, "checkpoints_noisy")
LATENT_DIR   = os.path.join(PROJECT_ROOT, "latents_noisy")


def add_noise(img_tensor, sigma=NOISE_SIGMA):
    noise = torch.randn_like(img_tensor) * sigma
    return torch.clamp(img_tensor + noise, 0, 10)


def train_noisy():
    os.makedirs(CKPT_DIR, exist_ok=True)
    if all(os.path.exists(f"{CKPT_DIR}/checkpoint_{s:07d}.pt") for s in CHECKPOINT_STEPS):
        print("noisy checkpoints 已存在，跳过训练")
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
        observation_transform=lambda img: add_noise(img, sigma=NOISE_SIGMA),
        save_dir=CKPT_DIR,
        progress_desc='Training [noisy]',
    )
    print(f"训练完成 → {CKPT_DIR}/")


def extract_noisy():
    os.makedirs(LATENT_DIR, exist_ok=True)
    env = gym.make(ENV_NAME)

    for s in CHECKPOINT_STEPS:
        out = f"{LATENT_DIR}/latents_{s:07d}.json"
        if os.path.exists(out):
            continue
        encoder = ImageEncoder(LATENT_DIM).to(DEVICE)
        ckpt = torch.load(f"{CKPT_DIR}/checkpoint_{s:07d}.pt", map_location=DEVICE)
        encoder.load_state_dict(ckpt["encoder"]); encoder.eval()

        dataset = extract_latents_from_encoder(encoder, env, n_samples=3000, seed=999)
        with open(out, "w") as f:
            json.dump(dataset, f)
        print(f"  step={s}: 提取完成")

    env.close()


def run_h6():
    env = gym.make(ENV_NAME)
    pred_losses, dir_accs = [], []

    print("\n=== H6 对照实验（噪声环境）===")
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
                img_noisy = add_noise(img)
                _, mu, _ = encoder(img_noisy)

                action = env.action_space.sample()
                next_obs, _, done, _, _ = env.step(action)
                next_img = torch.tensor(next_obs["image"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                next_img_noisy = add_noise(next_img)
                _, mu_next, _ = encoder(next_img_noisy)

                a_oh = F.one_hot(torch.tensor(action), N_ACTIONS).float().unsqueeze(0)
                mu_pred = transit(mu, a_oh)
                loss = F.mse_loss(mu_pred, mu_next).item()

                zs.append(mu.squeeze(0).numpy())
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

        pred_losses.append(pred_loss)
        dir_accs.append(acc)
        print(f"{s:>8} | {pred_loss:>10.4f} | {acc:>8.4f}")

    env.close()

    r_s, p_s = spearmanr(pred_losses, dir_accs)
    r_p, p_p = __import__('scipy').stats.pearsonr(pred_losses, dir_accs)

    print(f"\nSpearman r = {r_s:.4f}  p = {p_s:.4f}")
    print(f"Pearson  r = {r_p:.4f}  p = {p_p:.4f}")
    verdict = "H6 阳性" if (r_s < -0.4 and p_s < 0.1) else "H6 仍不显著"
    print(f"\n结论：{verdict}")

    h6_noisy_result = {
        "condition": "noisy",
        "noise_sigma": NOISE_SIGMA,
        "steps": CHECKPOINT_STEPS,
        "pred_losses": pred_losses,
        "dir_accs": dir_accs,
        "spearman_r": r_s, "spearman_p": p_s,
        "pearson_r": r_p,  "pearson_p": p_p,
        "verdict": verdict
    }
    os.makedirs(os.path.join(PROJECT_ROOT, "04_analysis"), exist_ok=True)
    with open(os.path.join(PROJECT_ROOT, "04_analysis", "h6_contrast_noisy.json"), "w") as f:
        json.dump(h6_noisy_result, f, indent=2)

    baseline_path = os.path.join(PROJECT_ROOT, "04_analysis", "mechanism_correlation.json")
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            bl = json.load(f)
        print(f"\n--- 基线 vs 噪声对比 ---")
        print(f"基线（确定性环境）：Spearman r={bl['spearman_r']:.4f}, p={bl['spearman_p']:.4f}")
        print(f"噪声（不确定环境）：Spearman r={r_s:.4f}, p={p_s:.4f}")

    return h6_noisy_result


if __name__ == "__main__":
    print("Step 1/3: 训练噪声版世界模型...")
    train_noisy()
    print("\nStep 2/3: 提取隐状态...")
    extract_noisy()
    print("\nStep 3/3: H6 对照分析...")
    run_h6()
    print("\n实验完成。结果见 04_analysis/h6_contrast_noisy.json")
