# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "pandas"]
# ///
"""Plot the Phase 3 learning curves from train.csv.

Run `zig build train` first, then:

    uv run scripts/plot_learning.py            # -> train.png

Each thin line is one seed's block accuracy over training; the bold line is the
mean across seeds. The dashed line is chance (0.5). The story: every seed starts
at chance and climbs above it -- the reward-modulated rule is learning the
input->action association.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=Path("train.csv"))
    ap.add_argument("-o", "--out", type=Path, default=Path("train.png"))
    args = ap.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"error: {args.csv} not found -- run `zig build train` first")

    df = pd.read_csv(args.csv)

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)

    for seed, g in df.groupby("seed"):
        ax.plot(g["episode"], g["block_accuracy"], lw=0.8, alpha=0.45,
                color="#1f77b4", label="_nolegend_")

    mean = df.groupby("episode")["block_accuracy"].mean()
    ax.plot(mean.index, mean.values, lw=2.4, color="#d62728", label="mean across seeds")

    ax.axhline(0.5, ls="--", lw=1.0, color="#666666", label="chance")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("episode")
    ax.set_ylabel("block accuracy")
    n_seeds = df["seed"].nunique()
    ax.set_title(f"Phase 3: two-choice association learning ({n_seeds} seeds, thin = per seed)")
    ax.legend(loc="lower right", framealpha=0.9)

    fig.savefig(args.out, dpi=130)
    final = df[df["episode"] == df["episode"].max()]["block_accuracy"]
    print(f"wrote {args.out}  (final block accuracy: mean={final.mean():.3f}, "
          f"min={final.min():.3f} over {n_seeds} seeds)")


if __name__ == "__main__":
    main()
