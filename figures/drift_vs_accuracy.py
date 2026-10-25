#!/usr/bin/env python3
r"""figures/drift_vs_accuracy.py — paired projection drift along v_leak vs accuracy, one point per
checkpoint (the paper's main-figure panel (b), without the base-Think anchor).

Reads the CSV from scripts/09_collect_results.py. Colour = base model, marker = method; one OLS
line per base model; dashed verticals = each base model's NoThink accuracy.

  python figures/drift_vs_accuracy.py --csv results/drift_vs_acc.csv --out figures/drift_vs_accuracy.pdf
"""
import argparse, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "dejavuserif", "font.size": 12, "axes.titlesize": 13,
    "axes.labelsize": 12.5, "axes.linewidth": 0.9, "axes.edgecolor": "#333333",
    "legend.fontsize": 9.5, "legend.framealpha": 0.92, "legend.edgecolor": "#cccccc",
    "grid.color": "#d9d9d9", "grid.linewidth": 0.7, "savefig.dpi": 200, "savefig.bbox": "tight",
})
MODEL_COLOR = {"Qwen3-8B": "#D64550", "Qwen3-4B": "#DFA22F", "MiniCPM4.1-8B": "#4C9F70"}
METHOD_MARK = {"GRPO": "o", "SFT": "s", "OPSD": "^", "OPSD-LoRA": "v"}
METHOD_LABEL = {"GRPO": "GRPO", "SFT": "SFT", "OPSD": "OPSD (full)", "OPSD-LoRA": "OPSD (LoRA)"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    rows = list(csv.DictReader(open(args.csv)))
    ck = [r for r in rows if r["method"] in METHOD_MARK]
    base = {r["model"]: float(r["acc"]) for r in rows if r["method"] == "base-NoThink"}

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.axhline(0, color="0.4", lw=0.8, zorder=1)
    for m, col in MODEL_COLOR.items():
        if m in base:
            ax.axvline(base[m], color=col, ls="--", lw=1.4, alpha=0.9, zorder=1)
        pts = [(float(r["acc"]), float(r["cohens_d"])) for r in ck if r["model"] == m]
        if len(pts) > 1:
            x, y = np.array(pts).T
            k, b = np.polyfit(x, y, 1)
            xs = np.linspace(0.05, 0.6, 50)
            ax.plot(xs, k * xs + b, color=col, lw=1.6, alpha=0.45, zorder=2)
    for r in ck:
        ax.scatter(float(r["acc"]), float(r["cohens_d"]), s=46, marker=METHOD_MARK[r["method"]],
                   color=MODEL_COLOR[r["model"]], edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_xlim(0.05, 0.6)
    ax.set_xlabel("Accuracy"); ax.set_ylabel("Cohen's $d$")
    ax.set_title(r"Paired projection drift, $\hat{v}^{(\ell)}_{\mathrm{leak}}$  ($\ell=20$)")
    ax.grid(alpha=0.6, zorder=0)
    hs = [plt.Line2D([], [], marker="o", ls="", color=c, markeredgecolor="white", markersize=8, label=m)
          for m, c in MODEL_COLOR.items()]
    hs += [plt.Line2D([], [], marker=mk, ls="", color="0.7", markeredgecolor="0.3", markersize=8,
                      label=METHOD_LABEL[k]) for k, mk in METHOD_MARK.items()]
    hs.append(plt.Line2D([], [], color="0.4", ls="--", lw=1.4, label="NoThink (base)"))
    ax.legend(handles=hs, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
    fig.savefig(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
