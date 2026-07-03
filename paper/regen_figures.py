import json, numpy as np, os, sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG  = os.path.join(BASE, 'results', 'figures')

mpl.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['DejaVu Serif', 'Times New Roman', 'Times'],
    'font.size':         11,
    'axes.titlesize':    12,
    'axes.labelsize':    11,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'legend.fontsize':   9.5,
    'figure.dpi':        300,
    'savefig.dpi':       300,
    'lines.linewidth':   2.0,
    'lines.markersize':  6,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.linestyle':    '--',
    'axes.axisbelow':    True,
})

BLUE   = '#2166AC'
RED    = '#D6604D'
ORANGE = '#E07F00'
GRAY   = '#888888'
LGRAY  = '#CCCCCC'


def fig_v2_vs_v3():
    steps = [1000, 5000, 10000, 25000, 50000, 100000]
    v2_dir  = [0.557, 0.574, 0.599, 0.613, 0.367, 0.268]
    v3_dir  = [0.568, 0.510, 0.652, 0.690, 0.637, 0.646]
    v2_posy = [0.230, 0.235, 0.223, 0.215, 0.077, 0.035]
    v3_posy = [0.186, 0.268, 0.223, 0.295, 0.293, 0.401]
    v2_dist = [0.0278,0.0143,0.0071,0.0017,0.0000,0.0000]
    v3_dist = [0.0291,0.0181,0.0068,0.0054,0.0025,0.0019]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    fig.subplots_adjust(wspace=0.38)

    kw_v2 = dict(color=RED,  linewidth=2.2, markersize=6,
                  linestyle='--', marker='o', label=r'$\beta=0.1$ (collapse)')
    kw_v3 = dict(color=BLUE, linewidth=2.2, markersize=6,
                  linestyle='-',  marker='s', label=r'$\beta=0.001$ (stable)')

    ax = axes[0]
    ax.plot(steps, v2_dir, **kw_v2)
    ax.plot(steps, v3_dir, **kw_v3)
    ax.axhline(0.25, color=GRAY,   ls=':', lw=1.5, label='Chance (25%)')
    ax.axhline(0.60, color=ORANGE, ls=':', lw=1.2, alpha=0.7, label='H1 target (60%)')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Direction Accuracy')
    ax.set_title('(a) Direction Accuracy\n(H1: Semantic Structure)', pad=8)
    ax.set_ylim(0.15, 0.82)
    ax.legend(loc='lower right', framealpha=0.9, fontsize=9)
    ax.set_xticklabels([f'{x//1000}k' for x in steps], rotation=30)
    ax.set_xticks(steps)

    ax = axes[1]
    ax.plot(steps, v2_posy, **kw_v2)
    ax.plot(steps, v3_posy, **kw_v3)
    ax.axhline(0.0, color=GRAY, ls=':', lw=1.5, label='Random baseline')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel(r'$Y$-Position $R^2$')
    ax.set_title('(b) Y-Position $R^2$\n(H3: Monotonic Growth)', pad=8)
    ax.set_ylim(-0.03, 0.52)
    ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
    ax.set_xticklabels([f'{x//1000}k' for x in steps], rotation=30)
    ax.set_xticks(steps)

    ax = axes[2]
    ax.plot(steps, v2_dist, **kw_v2)
    ax.plot(steps, v3_dist, **kw_v3)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Mean Pairwise Distance')
    ax.set_title('(c) Latent Pairwise Distance\n(Collapse Diagnosis)', pad=8)
    ax.annotate('Complete\nCollapse',
                xy=(50000, 0.0000), xytext=(30000, 0.012),
                arrowprops=dict(arrowstyle='->', color=RED, lw=1.5),
                color=RED, fontsize=9, ha='center')
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax.set_xticklabels([f'{x//1000}k' for x in steps], rotation=30)
    ax.set_xticks(steps)

    out = os.path.join(FIG, 'v2_vs_v3_comparison.png')
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"OK v2_vs_v3_comparison.png ({os.path.getsize(out)//1024}KB)")


def fig_h6():
    with open(os.path.join(BASE,'results','data','h6_dense_20pt.json')) as f:
        d = json.load(f)
    with open(os.path.join(BASE,'results','data','random_baseline.json')) as f:
        rb = json.load(f)
    with open(os.path.join(BASE,'results','data','rsa_scores.json')) as f:
        rsa_data = json.load(f)

    steps  = d['checkpoint_steps']
    losses = d['pred_losses']
    accs   = d['dir_accs']
    rb_dir = rb['dir_acc_mean']
    rb_std = rb['dir_acc_std']
    r_s, p_s = d['spearman_r'], d['spearman_p']

    rsa_6steps = [1000, 5000, 10000, 25000, 50000, 100000]
    rsa_6vals  = rsa_data['rsa_position']
    rsa_interp = np.interp(steps, rsa_6steps, rsa_6vals)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.subplots_adjust(wspace=0.42)

    ax = axes[0]
    sc = ax.scatter(losses, accs,
                    c=np.array(steps)/max(steps),
                    cmap='Blues', s=55, zorder=3,
                    edgecolors='#333333', linewidths=0.4, vmin=0, vmax=1)
    z = np.polyfit(losses, accs, 1)
    xl = np.linspace(min(losses), max(losses), 100)
    ax.plot(xl, np.poly1d(z)(xl), color=RED, ls='--', lw=1.8,
            alpha=0.7, label='Linear trend', zorder=2)
    ax.axhline(0.25, color=GRAY,   ls=':',  lw=1.5, label='Chance level (25%)')
    ax.axhline(rb_dir, color=ORANGE, ls='--', lw=2.0,
               label=f'Random-encoder baseline ({rb_dir:.3f})')
    ax.fill_between(xl, rb_dir - rb_std, rb_dir + rb_std,
                    color=ORANGE, alpha=0.08)

    ax.xaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, _: f'{x*1000:.1f}$\\times$10$^{{-3}}$' if x > 0 else '0')
    )
    ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.4f'))
    ax.set_xlabel('Prediction Loss (MSE)')
    ax.set_ylabel('Direction Accuracy')
    ax.set_title(f'(a) H6: Prediction--Semantics Co-evolution\n'
                 f'Spearman $r={r_s:.3f}$, $p={p_s:.3f}$ ($n=20$)', pad=8)
    ax.legend(loc='lower left', framealpha=0.9, fontsize=9)
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label('Training progress', fontsize=9)
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(['5k', '50k', '100k'])

    ax1 = axes[1]
    l1, = ax1.plot(steps, [l*1e3 for l in losses],
                   color=RED, lw=2.0, marker='o', ms=4,
                   label='Pred. Loss ($\\times 10^{-3}$) $\\downarrow$')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Prediction Loss ($\\times 10^{-3}$)', color=RED)
    ax1.tick_params(axis='y', colors=RED)
    ax1.set_xticks(steps[::4])
    ax1.set_xticklabels([f'{x//1000}k' for x in steps[::4]])

    ax2 = ax1.twinx()
    l2, = ax2.plot(steps, accs,
                   color=BLUE, lw=2.0, marker='s', ms=4, ls='--',
                   label='Direction Acc. $\\uparrow$')
    l3, = ax2.plot(steps, rsa_interp,
                   color='#5E4FA2', lw=1.8, marker='^', ms=4, ls=':',
                   label='RSA Position $\\uparrow$', alpha=0.9)
    l4  = ax2.axhline(rb_dir, color=ORANGE, ls='--', lw=1.8,
                      label=f'Random-enc. ({rb_dir:.3f})', alpha=0.85)
    ax2.set_ylabel('Direction Accuracy / RSA Score', color=BLUE)
    ax2.tick_params(axis='y', colors=BLUE)
    ax2.set_ylim(0.15, 0.75)
    ax1.set_title('(b) Temporal Co-evolution\n(20 checkpoints, every 5k steps)', pad=8)

    handles = [l1, l2, l3, l4]
    labels  = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc='upper right',
               framealpha=0.9, fontsize=8.5, ncol=2)

    out = os.path.join(FIG, 'h6_dense_20pt.png')
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"OK h6_dense_20pt.png ({os.path.getsize(out)//1024}KB)")


def fig_rsa():
    with open(os.path.join(BASE,'results','data','rsa_scores.json')) as f:
        d = json.load(f)
    steps = [1000,5000,10000,25000,50000,100000]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.subplots_adjust(wspace=0.38)

    for ax, key, label, color, panel in [
        (axes[0], 'rsa_direction', 'Direction RSA', BLUE,  '(a)'),
        (axes[1], 'rsa_position',  'Position RSA',  RED,   '(b)'),
    ]:
        vals = d[key]
        ax.plot(steps, vals, color=color, lw=2.2, marker='o', ms=6)
        ax.fill_between(steps, 0, vals, color=color, alpha=0.08)
        ax.axhline(0.0, color=GRAY, ls='--', lw=1.5, label='Random baseline (0.0)')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('RSA Score (Spearman $r$)')
        ax.set_title(f'{panel} {label}\nvs. Physical Semantic Similarity', pad=8)
        ax.set_xticks(steps)
        ax.set_xticklabels([f'{x//1000}k' for x in steps], rotation=30)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.set_ylim(-0.02, max(vals)*1.3)

    out = os.path.join(FIG, 'rsa_curve.png')
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"OK rsa_curve.png ({os.path.getsize(out)//1024}KB)")


if __name__ == '__main__':
    print("Generating paper figures (300 DPI)...")
    fig_v2_vs_v3()
    fig_h6()
    fig_rsa()
    print("Done")
