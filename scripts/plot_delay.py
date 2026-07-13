# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "pandas"]
# ///
"""Plot the Phase 4 delayed-association results.

Run `zig build delay` first (writes delay.csv + retention.csv), then:

    uv run scripts/plot_delay.py            # -> delay.png

Top panel: final accuracy vs delay, working-memory condition vs reservoir-only.
The memory curve stays high across long delays; the reservoir-only curve decays
toward chance -- the gap is the working-memory mechanism doing the retaining.

Bottom panel: the recurrent-state analysis. Mean firing rate of the stimulus's
OWN input assembly vs the OTHER assembly, across one trial. The own assembly
stays lit through the delay while the other stays silent -- stimulus-specific
retention, read directly off the neurons.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

MEM_COLOR = "#1f77b4"
RES_COLOR = "#d62728"
OWN_COLOR = "#1f77b4"
OTHER_COLOR = "#d62728"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--delay-csv", type=Path, default=Path("delay.csv"))
    ap.add_argument("--retention-csv", type=Path, default=Path("retention.csv"))
    ap.add_argument("-o", "--out", type=Path, default=Path("delay.png"))
    args = ap.parse_args()

    if not args.delay_csv.exists():
        raise SystemExit(f"error: {args.delay_csv} not found -- run `zig build delay` first")

    fig, (ax_acc, ax_ret) = plt.subplots(
        2, 1, figsize=(11, 8), constrained_layout=True,
    )

    # --- accuracy vs delay ------------------------------------------------
    df = pd.read_csv(args.delay_csv)
    means = df.groupby(["condition", "delay"])["accuracy"].mean().reset_index()
    labels = {"memory": ("working memory", MEM_COLOR), "reservoir_only": ("reservoir only", RES_COLOR)}
    for cond, g in means.groupby("condition"):
        label, color = labels.get(cond, (cond, "#333333"))
        ax_acc.plot(g["delay"], g["accuracy"], "o-", color=color, lw=2, label=label)
    ax_acc.axhline(0.5, ls="--", lw=1.0, color="#666666", label="chance")
    ax_acc.set_ylim(0.4, 1.02)
    ax_acc.set_xlabel("delay (timesteps with no input)")
    ax_acc.set_ylabel("final accuracy")
    ax_acc.set_title("Phase 4: retention vs delay")
    ax_acc.legend(loc="lower left", framealpha=0.9)

    # --- recurrent-state retention ---------------------------------------
    if args.retention_csv.exists():
        r = pd.read_csv(args.retention_csv)
        ax_ret.plot(r["t"], r["own_assembly_rate"], color=OWN_COLOR, lw=1.6,
                    label="stimulated (own) assembly")
        ax_ret.plot(r["t"], r["other_assembly_rate"], color=OTHER_COLOR, lw=1.6,
                    label="other assembly")
        # Shade the phases.
        for phase, color in (("stimulus", "#cccccc"), ("delay", "#ffe08a"), ("readout", "#bfe3c0")):
            seg = r[r["phase"] == phase]
            if not seg.empty:
                ax_ret.axvspan(seg["t"].min(), seg["t"].max() + 1, color=color, alpha=0.3)
                ax_ret.text(seg["t"].mean(), ax_ret.get_ylim()[1], phase, ha="center",
                            va="top", fontsize=9, color="#555555")
        ax_ret.set_xlabel("timestep within trial")
        ax_ret.set_ylabel("assembly firing rate")
        ax_ret.set_title("recurrent-state analysis: stimulus-specific persistence")
        ax_ret.legend(loc="upper right", framealpha=0.9)
    else:
        ax_ret.set_visible(False)

    fig.savefig(args.out, dpi=130)
    tail = means[means["delay"] == means["delay"].max()]
    summary = ", ".join(f"{row.condition}={row.accuracy:.3f}" for row in tail.itertuples())
    print(f"wrote {args.out}  (accuracy at delay {means['delay'].max()}: {summary})")


if __name__ == "__main__":
    main()
