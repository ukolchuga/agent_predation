import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training(log_csv: str, out_png: str, title: str = ""):
    with open(log_csv) as f:
        rows = list(csv.DictReader(f))
    col = lambda k: [float(r[k]) for r in rows]
    it = col("iteration")

    fig, ax = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    panels = [("pnl_mean", "victim PnL per episode"), ("mean_frac", "mean position / max_pos"),
              ("ep_cost", "episode cost"), ("std", "policy std")]
    for a, (key, label) in zip(ax.flat, panels):
        a.plot(it, col(key), lw=1.5)
        a.set_title(label, fontsize=10)
        a.grid(alpha=0.3)
    ax[0, 1].set_ylim(0, 1.02)
    for a in ax[1]:
        a.set_xlabel("iteration")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
