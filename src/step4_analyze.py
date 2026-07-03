import numpy as np
import json
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler
import umap
import torch
import open_clip
from sklearn.metrics.pairwise import cosine_similarity


_clip_model = None
_clip_tokenizer = None

def get_clip():
    global _clip_model, _clip_tokenizer
    if _clip_model is None:
        print("加载 CLIP 模型...")
        _clip_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        _clip_model.eval()
        _clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return _clip_model, _clip_tokenizer


def load_latent_dataset(path):
    with open(path) as f:
        latent_records = json.load(f)
    Z    = np.array([d["z"]               for d in latent_records])
    dirs = np.array([d["info"]["agent_dir"] for d in latent_records])
    posx = np.array([d["info"]["agent_pos"][0] for d in latent_records])
    posy = np.array([d["info"]["agent_pos"][1] for d in latent_records])
    carry= np.array([d["info"]["carrying"] for d in latent_records]).astype(int)
    return Z, dirs, posx, posy, carry, latent_records


def plot_umap(Z, dirs, carry, step, output_dir):
    print(f"  UMAP 可视化 (step={step})...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
    Z_2d    = reducer.fit_transform(Z[:2000])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"隐空间结构 (训练步数={step})", fontsize=14)

    axes[0].scatter(Z_2d[:,0], Z_2d[:,1],
                    c=dirs[:2000], cmap="tab10", s=3, alpha=0.6)
    axes[0].set_title("按智能体方向着色")

    axes[1].scatter(Z_2d[:,0], Z_2d[:,1],
                    c=carry[:2000], cmap="RdYlGn", s=3, alpha=0.6)
    axes[1].set_title("按是否携带物体着色")

    plt.tight_layout()
    out = os.path.join(output_dir, f"umap_{step:07d}.png")
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def linear_probe(Z, dirs, posx, posy, carry):
    scaler = StandardScaler()
    Zs     = scaler.fit_transform(Z)
    probe_results = {}

    def clf_probe(name, labels):
        if len(np.unique(labels)) < 2:
            print(f"    [{name}] 跳过：只有一个类别（{np.unique(labels)[0]}）")
            probe_results[name] = None
            return
        X_tr, X_te, y_tr, y_te = train_test_split(
            Zs, labels, test_size=0.2, random_state=0, stratify=labels)
        acc = accuracy_score(y_te,
              LogisticRegression(max_iter=1000).fit(X_tr, y_tr).predict(X_te))
        probe_results[name] = float(acc)
        print(f"    [{name}] accuracy = {acc:.4f}")

    def reg_probe(name, labels):
        X_tr, X_te, y_tr, y_te = train_test_split(
            Zs, labels, test_size=0.2, random_state=0)
        r2 = r2_score(y_te, Ridge().fit(X_tr, y_tr).predict(X_te))
        probe_results[name] = float(r2)
        print(f"    [{name}] R2 = {r2:.4f}")

    clf_probe("direction_acc", dirs)
    reg_probe("posx_r2",       posx)
    reg_probe("posy_r2",       posy)
    clf_probe("carrying_acc",  carry)

    return probe_results


def make_description(info):
    dir_names = ["facing right", "facing down", "facing left", "facing up"]
    direction = dir_names[info["agent_dir"]]
    carry     = "carrying an object" if info["carrying"] else "hands empty"
    x, y      = info["agent_pos"]
    return f"agent at position ({x},{y}), {direction}, {carry}"


def clip_alignment_score(Z, latent_records, sample_size=1000):
    model, tokenizer = get_clip()

    idx  = np.random.choice(len(Z), min(sample_size, len(Z)), replace=False)
    Z_s  = Z[idx]
    state_descriptions = [make_description(latent_records[i]["info"]) for i in idx]

    text_embeddings = []
    for i in range(0, len(state_descriptions), 64):
        batch  = state_descriptions[i:i+64]
        tokens = tokenizer(batch)
        with torch.no_grad():
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        text_embeddings.append(emb.numpy())
    T = np.concatenate(text_embeddings, axis=0)

    split = int(len(Z_s) * 0.8)
    reg   = Ridge(alpha=1.0).fit(Z_s[:split], T[:split])
    T_pred= reg.predict(Z_s[split:])
    T_true= T[split:]

    sims  = [cosine_similarity([T_pred[i]], [T_true[i]])[0][0]
             for i in range(len(T_pred))]
    return float(np.mean(sims))


def plot_alignment_curve(steps, scores, probe_results, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(steps, scores, "o-", color="#2563cc", linewidth=2, markersize=6)
    axes[0].axhline(y=0.0, color="gray", linestyle="--", alpha=0.5,
                    label="随机基线")
    axes[0].set_xlabel("训练步数（物理交互量）", fontsize=12)
    axes[0].set_ylabel("CLIP 语言对齐分数", fontsize=12)
    axes[0].set_title("物理交互越多，隐空间与语言越对齐", fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    metrics = ["direction_acc", "posx_r2", "posy_r2", "carrying_acc"]
    labels  = ["方向准确率", "X坐标R²", "Y坐标R²", "携带物准确率"]
    baselines = [0.25, 0.0, 0.0, 0.5]

    for i, (m, label, baseline) in enumerate(zip(metrics, labels, baselines)):
        vals = [r[m] for r in probe_results]
        axes[1].plot(steps, vals, "o-", label=label, linewidth=2, markersize=5)
        axes[1].axhline(y=baseline, linestyle=":", alpha=0.4)

    axes[1].set_xlabel("训练步数", fontsize=12)
    axes[1].set_ylabel("预测分数", fontsize=12)
    axes[1].set_title("线性探测：隐空间包含的语义信息", fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir, "alignment_curve.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"对齐曲线图保存到: {out}")


def run_analysis(latent_dir=None, output_dir="./results"):
    if latent_dir is None:
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        latent_dir = os.path.join(PROJECT_ROOT, "data", "latents", "v3_main")
    os.makedirs(output_dir, exist_ok=True)

    latent_files = sorted([
        f for f in os.listdir(latent_dir) if f.endswith(".json")
    ])
    print(f"找到 {len(latent_files)} 个隐状态文件")

    steps         = []
    clip_scores   = []
    probe_results = []

    for fname in latent_files:
        step = int(fname.replace("latents_", "").replace(".json", ""))
        path = os.path.join(latent_dir, fname)

        print(f"\n分析 step={step}...")
        Z, dirs, posx, posy, carry, latent_records = load_latent_dataset(path)

        plot_umap(Z, dirs, carry, step, output_dir)

        probe = linear_probe(Z, dirs, posx, posy, carry)
        print(f"  线性探测: {probe}")

        score = clip_alignment_score(Z, latent_records)
        print(f"  CLIP 对齐分数: {score:.4f}")

        steps.append(step)
        clip_scores.append(score)
        probe_results.append(probe)

    plot_alignment_curve(steps, clip_scores, probe_results, output_dir)

    summary = {
        "steps":       steps,
        "clip_scores": clip_scores,
        "probe":       probe_results,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n分析完成，结果保存在 {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dir",  default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "latents", "v3_main"))
    parser.add_argument("--output_dir",  default="./results")
    args = parser.parse_args()
    run_analysis(args.latent_dir, args.output_dir)
