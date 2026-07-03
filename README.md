# Emergent Semantic Representations in World Models through Physical Interaction without Linguistic Supervision

**[Paper (arXiv)](https://arxiv.org/abs/2605.28865)** | **[PDF](paper/main.pdf)**

> Jiayi Fang · Shanghai University of Finance and Economics

---

## Overview

This repository contains the code and data for the paper:

> *Emergent Semantic Representations in World Models through Physical Interaction without Linguistic Supervision*

We investigate whether a world model trained exclusively through physical random exploration — without any linguistic supervision — spontaneously develops latent representations that encode spatial semantic structure. Using a minimal VAE-based world model on MiniGrid, we show that:

1. **Physical geometry organizes world model representations**: Position RSA rises 5.0× above randomly initialized encoders, demonstrating training-induced structural organization beyond CNN inductive bias.
2. **Double knockout confirms the shared-driver mechanism**: Standard KL regularization (β=0.1) forces the encoder away from geometric structure, collapsing *both* prediction performance and semantic alignment simultaneously — exactly as the shared-driver account predicts.
3. **The double knockout replicates** in a larger environment (Empty-16×16).

---

## Repository Structure

```
├── src/
│   ├── step1_test_env.py          # Environment sanity check
│   ├── step2_train_world_model.py # Main training script (β=0.001 stable, β=0.1 collapse)
│   ├── step3_extract_latents.py   # Extract latent representations at each checkpoint
│   ├── step4_analyze.py           # Linear probing + RSA analysis
│   ├── run_seeds.py               # 3-seed multi-run for statistical validation
│   ├── run_h6_dense.py            # H6: 20-checkpoint prediction–semantics co-evolution
│   ├── run_h6_contrast.py         # H6 contrast: Gaussian noise condition
│   ├── run_h6_partial_obs.py      # H6 contrast: partial observability condition
│   ├── run_empty16_knockout.py    # Double knockout replication in Empty-16×16
│   └── run_fourrooms_knockout.py  # FourRooms exploration (see Limitations)
│
├── paper/
│   ├── main.tex                   # LaTeX source
│   ├── references.bib             # Bibliography
│   ├── main.pdf                   # Compiled preprint
│   └── regen_figures.py           # Reproduce all figures at 300 DPI
│
├── results/
│   ├── figures/                   # All figures used in the paper
│   │   ├── v2_vs_v3_comparison.png      # Figure 1: Double knockout (Empty-8×8)
│   │   ├── h6_dense_20pt.png            # Figure 2: H6 co-evolution (20 checkpoints)
│   │   └── empty16_knockout.png         # Figure 3: Replication (Empty-16×16)
│   └── data/                      # Numerical results (JSON)
│       ├── multi_seed_summary.json      # 3-seed main results
│       ├── h6_dense_20pt.json           # H6 20-checkpoint data
│       ├── h6_contrast_noisy.json       # H6 noise contrast
│       ├── h6_contrast_partial_obs.json # H6 partial-obs contrast
│       ├── random_baseline.json         # Random encoder baseline
│       ├── rsa_scores.json              # RSA scores across checkpoints
│       ├── empty16_knockout.json        # Empty-16×16 knockout results
│       └── fourrooms_knockout.json      # FourRooms results (negative, see Limitations)
│
├── requirements.txt
├── setup.sh                       # Linux/Mac setup
└── setup.bat                      # Windows setup
```

> **Note**: Model checkpoints (`.pt` files) and raw latent data are not included due to file size.
> All results can be reproduced by running the scripts in order (see below).

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Jia-yi-FANG/world-model-semantics
cd world-model-semantics

# Create virtual environment
python3 -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Verify environment
python src/step1_test_env.py    # Should print: Environment test passed OK
```

**Requirements**: Python ≥ 3.9, PyTorch ≥ 2.0 (CPU only, no GPU required).
Each training run takes approximately 30 minutes on a standard laptop CPU.

---

## Reproducing the Results

### Step 1: Train the world model

```bash
# Stable configuration (β=0.001) — main experiment
python src/step2_train_world_model.py

# Collapse configuration (β=0.1) — for double knockout comparison
# Edit KL_WEIGHT = 0.1 in step2_train_world_model.py, then:
python src/step2_train_world_model.py
```

### Step 2: Extract latent representations

```bash
python src/step3_extract_latents.py
```

### Step 3: Linear probing + RSA analysis

```bash
python src/step4_analyze.py
```

### Step 4: Run multi-seed validation (3 seeds)

```bash
python src/run_seeds.py
```

### Step 5: H6 co-evolution experiment (20 checkpoints)

```bash
python src/run_h6_dense.py
python src/run_h6_contrast.py       # Gaussian noise contrast
python src/run_h6_partial_obs.py    # Partial observability contrast
```

### Step 6: Double knockout replication (Empty-16×16)

```bash
python src/run_empty16_knockout.py
```

### Regenerate all figures

```bash
cd paper
python regen_figures.py   # Outputs to results/figures/ at 300 DPI
```

---

## Key Results

| Metric | Trained (β=0.001) | Random Encoder | Random Policy |
|--------|:-----------------:|:--------------:|:-------------:|
| Direction Accuracy | **0.677 ± 0.029** | 0.547 ± 0.029 | 0.250 |
| Y-Position R² | **0.333 ± 0.036** | ≈0 | 0.000 |
| RSA Position | **0.192 ± 0.047** | 0.029 ± 0.031 | ≈0 |

*Results across 3 random seeds at step 100,000.*

**Double Knockout** (β=0.1): Direction accuracy drops from 61.3% → 26.8% (chance) and latent pairwise distance collapses to 0.000 by step 50,000 — *simultaneously* with prediction performance, confirming the shared geometric driver account.

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{fang2026emergent,
  title   = {Emergent Semantic Representations in World Models
             through Physical Interaction without Linguistic Supervision},
  author  = {Fang, Jiayi},
  year    = {2026},
  note    = {arXiv preprint arXiv:2605.28865}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
