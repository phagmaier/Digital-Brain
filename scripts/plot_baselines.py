# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "pandas"]
# ///
"""Plot Stage 1 external baseline results.

    uv run scripts/baselines.py
    uv run scripts/plot_baselines.py   # -> baseline.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLORS = {
    "tabular": "#7f7f7f",
    "esn": "#2ca02c",
    "bptt_rnn": "#1f77b4",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=Path("baseline.csv"))
    ap.add_argument("--curves", type=Path, default=Path("baseline_curves.csv"))
    ap.add_argument("-o", "--out", type=Path, default=Path("baseline.png"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    curves = pd.read_csv(args.curves) if args.curves.exists() else None

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # Immediate accuracy by model
    ax = axes[0, 0]
    imm = df[df["protocol"] == "immediate"]
    models = list(imm["model"].unique())
    means = [imm[imm["model"] == m]["final_accuracy"].mean() for m in models]
    stds = [imm[imm["model"] == m]["final_accuracy"].std(ddof=1) for m in models]
    ax.bar(models, means, yerr=stds, color=[COLORS.get(m, "C0") for m in models], capsize=4)
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("final accuracy")
    ax.set_title("Immediate association")

    # Delay accuracy
    ax = axes[0, 1]
    dly = df[df["protocol"].str.startswith("delay")]
    means = [dly[dly["model"] == m]["final_accuracy"].mean() for m in models]
    stds = [dly[dly["model"] == m]["final_accuracy"].std(ddof=1) for m in models]
    ax.bar(models, means, yerr=stds, color=[COLORS.get(m, "C0") for m in models], capsize=4)
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("final accuracy")
    ax.set_title("Delayed association")

    # Forgetting retest
    ax = axes[1, 0]
    frg = df[df["protocol"] == "forgetting"]
    means = [frg[frg["model"] == m]["final_accuracy"].mean() for m in models]
    stds = [frg[frg["model"] == m]["final_accuracy"].std(ddof=1) for m in models]
    ax.bar(models, means, yerr=stds, color=[COLORS.get(m, "C0") for m in models], capsize=4)
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("A-retest after B-disuse")
    ax.set_title("Forgetting (no consolidation)")

    # Shift curves
    ax = axes[1, 1]
    if curves is not None and not curves.empty:
        for m, g in curves.groupby("model"):
            mean = g.groupby("episode")["block_accuracy"].mean()
            ax.plot(mean.index, mean.values, color=COLORS.get(m, "C0"), lw=2, label=m)
        flip = curves.loc[curves["phase"] == "post_shift", "episode"].min()
        if pd.notna(flip):
            ax.axvline(flip, color="gray", ls="--", lw=1)
        ax.legend(frameon=False, fontsize=8)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("episode")
    ax.set_ylabel("block accuracy")
    ax.set_title("Distribution shift (A↔B flip)")

    fig.suptitle("Stage 1 external baselines", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
