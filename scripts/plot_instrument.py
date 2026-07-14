# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "pandas"]
# ///
"""Plot Stage 1 instrumentation artefacts.

Run `zig build instrument -Doptimize=ReleaseFast` first, then:

    uv run scripts/plot_instrument.py   # -> instrument.png

Panels:
  1. Online-update cost ratio (local / dense) per seed
  2. Sparsity: mean firing rate per seed
  3. Forgetting curves: A-retest during block-B disuse (consolidation on vs off)
  4. Distribution shift: block accuracy pre/post mapping flip
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ON = "#1f77b4"
OFF = "#d62728"
SHIFT = "#2ca02c"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cost", type=Path, default=Path("instrument_cost.csv"))
    ap.add_argument("--forgetting", type=Path, default=Path("instrument_forgetting.csv"))
    ap.add_argument("--shift", type=Path, default=Path("instrument_shift.csv"))
    ap.add_argument("-o", "--out", type=Path, default=Path("instrument.png"))
    args = ap.parse_args()

    cost = pd.read_csv(args.cost)
    forget = pd.read_csv(args.forgetting)
    shift = pd.read_csv(args.shift)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    ax.bar(cost["seed"].astype(str), cost["cost_ratio"], color=ON)
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="soft band 0.5")
    ax.set_xlabel("seed")
    ax.set_ylabel("local / dense ops")
    ax.set_title("Online-update cost ratio")
    ax.set_ylim(0, max(0.6, cost["cost_ratio"].max() * 1.2))
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.bar(cost["seed"].astype(str), cost["mean_firing_rate"], color=ON)
    ax.axhline(0.05, color="gray", ls="--", lw=1, label="target_rate 0.05")
    ax.set_xlabel("seed")
    ax.set_ylabel("spikes / neuron / step")
    ax.set_title("Activity sparsity (final window)")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    for cond, color, label in (
        ("consolidation_on", ON, "consolidation ON"),
        ("consolidation_off", OFF, "consolidation OFF"),
    ):
        sub = forget[forget["condition"] == cond]
        for seed, g in sub.groupby("seed"):
            ax.plot(
                g["block_b_episode"],
                g["retest_a"],
                color=color,
                alpha=0.25,
                lw=1,
            )
        mean = sub.groupby("block_b_episode")["retest_a"].mean()
        ax.plot(mean.index, mean.values, color=color, lw=2.5, label=label)
    ax.set_xlabel("block-B episode")
    ax.set_ylabel("frozen A-retest accuracy")
    ax.set_title("Forgetting curves (A after B-only disuse)")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    for seed, g in shift.groupby("seed"):
        ax.plot(g["episode"], g["block_accuracy"], color=SHIFT, alpha=0.25, lw=1)
    mean = shift.groupby("episode")["block_accuracy"].mean()
    ax.plot(mean.index, mean.values, color=SHIFT, lw=2.5, label="mean")
    flip = shift.loc[shift["phase"] == "post_shift", "episode"].min()
    if pd.notna(flip):
        ax.axvline(flip, color="gray", ls="--", lw=1, label=f"mapping flip @ {int(flip)}")
    ax.set_xlabel("episode")
    ax.set_ylabel("block accuracy")
    ax.set_title("Distribution shift (A↔B mapping flip)")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(frameon=False)

    fig.suptitle("Stage 1 instrumentation", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
