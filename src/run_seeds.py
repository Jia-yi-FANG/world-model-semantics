import os, sys, json, torch, numpy as np
import gymnasium as gym, minigrid
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(__file__))
from step2_train_world_model import (
    ImageEncoder, TransitionModel, LATENT_DIM, N_ACTIONS, DEVICE,
    LR_WM, TOTAL_STEPS, ENV_NAME, CHECKPOINT_STEPS
)
from training_utils import train_world_model

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = [0, 1, 2]
KL_WEIGHT = 0.001


def train_one_seed(seed):
    ckpt_dir = os.path.join(PROJECT_ROOT, f"checkpoints_seed{seed}")
    if all(os.path.exists(f"{ckpt_dir}/checkpoint_{s:07d}.pt") for s in CHECKPOINT_STEPS):
        print(f"  Seed {seed}: checkpoints 已存在，跳过训练")
        return

    os.makedirs(ckpt_dir, exist_ok=True)

    encoder = ImageEncoder(LATENT_DIM).to(DEVICE)
    transit = TransitionModel(LATENT_DIM, N_ACTIONS).to(DEVICE)

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
        seed=seed,
        observation_transform=None,
        save_dir=ckpt_dir,
        progress_desc=f'Seed {seed}',
    )
    print(f"  Seed {seed}: 训练完成，checkpoints → {ckpt_dir}/")


def extract_one_seed(seed):
    ckpt_dir  = os.path.join(PROJECT_ROOT, f"checkpoints_seed{seed}")
    out_dir   = os.path.join(PROJECT_ROOT, f"latents_seed{seed}")
    os.makedirs(out_dir, exist_ok=True)

    env = gym.make(ENV_NAME)
    for s in CHECKPOINT_STEPS:
        out_path = f"{out_dir}/latents_{s:07d}.json"
        if os.path.exists(out_path):
            continue
        encoder = ImageEncoder(LATENT_DIM).to(DEVICE)
        ckpt = torch.load(f"{ckpt_dir}/checkpoint_{s:07d}.pt", map_location=DEVICE)
        encoder.load_state_dict(ckpt["encoder"])
        encoder.eval()

        dataset = []
        obs, _ = env.reset(seed=seed + 100)
        for _ in range(3000):
            img = torch.tensor(obs["image"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
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
            if done: obs, _ = env.reset()
        with open(out_path, "w") as f:
            json.dump(dataset, f)
    env.close()
    print(f"  Seed {seed}: 隐状态提取完成 → {out_dir}/")


def analyze_one_seed(seed):
    latent_dir = os.path.join(PROJECT_ROOT, f"latents_seed{seed}")
    results = {}
    for s in CHECKPOINT_STEPS:
        with open(f"{latent_dir}/latents_{s:07d}.json") as f:
            data = json.load(f)
        Z    = np.array([d["z"] for d in data])
        dirs = np.array([d["info"]["agent_dir"] for d in data])
        posx = np.array([d["info"]["agent_pos"][0] for d in data])
        posy = np.array([d["info"]["agent_pos"][1] for d in data])

        Zs = StandardScaler().fit_transform(Z)
        X_tr, X_te, y_tr, y_te = train_test_split(Zs, dirs, test_size=0.2,
                                                    random_state=0, stratify=dirs)
        dir_acc = accuracy_score(y_te,
                  LogisticRegression(max_iter=1000).fit(X_tr, y_tr).predict(X_te))
        X_tr, X_te, y_tr, y_te = train_test_split(Zs, posy, test_size=0.2, random_state=0)
        posy_r2 = r2_score(y_te, Ridge().fit(X_tr, y_tr).predict(X_te))
        np.random.seed(42)
        idx = np.random.choice(len(Z), min(500, len(Z)), replace=False)
        Zs_sub = cosine_similarity(Z[idx])
        pos_d  = (np.abs(posx[idx, None] - posx[None, idx]) +
                  np.abs(posy[idx, None] - posy[None, idx]))
        pos_sim = 1.0 / (1.0 + pos_d)
        mask = np.triu(np.ones((len(idx), len(idx)), dtype=bool), k=1)
        rsa_pos = float(spearmanr(Zs_sub[mask], pos_sim[mask]).statistic)

        results[s] = {"dir_acc": float(dir_acc),
                      "posy_r2": float(posy_r2),
                      "rsa_pos": rsa_pos}
        print(f"    step={s:>7d}: dir={dir_acc:.3f}  posy_r2={posy_r2:.3f}  rsa_pos={rsa_pos:.3f}")
    return results


def aggregate(multi_seed_results):
    print("\n" + "="*65)
    print("=== 汇总结果（均值 ± 标准差，3 seeds）===")
    print("="*65)
    print(f"{'Step':>8} | {'Dir Acc':>14} | {'Y-Pos R²':>14} | {'RSA Pos':>14}")
    print("-"*65)

    summary = {}
    for s in CHECKPOINT_STEPS:
        dir_accs  = [multi_seed_results[seed][s]["dir_acc"]  for seed in SEEDS]
        posy_r2s  = [multi_seed_results[seed][s]["posy_r2"]  for seed in SEEDS]
        rsa_poss  = [multi_seed_results[seed][s]["rsa_pos"]  for seed in SEEDS]
        print(f"{s:>8} | {np.mean(dir_accs):.3f}±{np.std(dir_accs):.3f}  "
              f"| {np.mean(posy_r2s):.3f}±{np.std(posy_r2s):.3f}  "
              f"| {np.mean(rsa_poss):.3f}±{np.std(rsa_poss):.3f}")
        summary[s] = {
            "dir_acc_mean":  float(np.mean(dir_accs)),
            "dir_acc_std":   float(np.std(dir_accs)),
            "posy_r2_mean":  float(np.mean(posy_r2s)),
            "posy_r2_std":   float(np.std(posy_r2s)),
            "rsa_pos_mean":  float(np.mean(rsa_poss)),
            "rsa_pos_std":   float(np.std(rsa_poss)),
        }
    print(f"\n随机基线：方向 0.250，位置 R² ≈ 0，RSA ≈ 0.000")

    out = os.path.join(PROJECT_ROOT, "04_analysis", "multi_seed_summary.json")
    with open(out, "w") as f:
        json.dump({"seeds": SEEDS, "kl_weight": KL_WEIGHT,
                   "checkpoint_steps": CHECKPOINT_STEPS,
                   "summary": {str(k): v for k, v in summary.items()}}, f, indent=2)
    print(f"\n结果已保存: {out}")
    return summary


if __name__ == "__main__":
    seed0_dir = os.path.join(PROJECT_ROOT, "checkpoints_seed0")
    os.makedirs(seed0_dir, exist_ok=True)
    src_dir = os.path.join(PROJECT_ROOT, "data", "checkpoints", "v3_main")
    for s in CHECKPOINT_STEPS:
        src = f"{src_dir}/checkpoint_{s:07d}.pt"
        dst = f"{seed0_dir}/checkpoint_{s:07d}.pt"
        if os.path.exists(src) and not os.path.exists(dst):
            import shutil
            shutil.copy2(src, dst)
    print("Seed 0: 复用主实验 checkpoints")

    for seed in [1, 2]:
        print(f"\n--- 训练 seed={seed} ---")
        train_one_seed(seed)

    multi_seed_results = {}
    for seed in SEEDS:
        print(f"\n--- 提取隐状态 seed={seed} ---")
        extract_one_seed(seed)

        print(f"\n--- 分析 seed={seed} ---")
        multi_seed_results[seed] = analyze_one_seed(seed)

    aggregate(multi_seed_results)
    print("\n全部完成！")
