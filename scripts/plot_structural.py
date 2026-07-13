# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "pandas"]
# ///
"""Plot the Phase 5 structural-plasticity timeline.

Run `zig build grow` first (writes structural.csv), then:

    uv run scripts/plot_structural.py        # -> structural.png

Top panel: cumulative grows and prunes over training -- the connections turning
over. Growth explores (rising) then self-limits at the target out-degree; a
trickle of pruning removes genuinely-disused edges.

Bottom panel: the live structural-edge population (left axis) against the
population firing rate at each structural event (right axis). The point of the
figure: the graph is being rewired (live count moves, churn accumulates) while
the firing rate stays in a stable band -- Phase 5's exit criterion, made visible.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

GROW_COLOR = "#2ca02c"
PRUNE_COLOR = "#d62728"
LIVE_COLOR = "#1f77b4"
RATE_COLOR = "#ff7f0e"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=Path("structural.csv"))
    ap.add_argument("-o", "--out", type=Path, default=Path("structural.png"))
    args = ap.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"error: {args.csv} not found -- run `zig build grow` first")

    df = pd.read_csv(args.csv)

    fig, (ax_churn, ax_live) = plt.subplots(
        2, 1, figsize=(11, 8), constrained_layout=True, sharex=True,
    )

    # --- cumulative churn -------------------------------------------------
    ax_churn.plot(df["episode"], df["cum_grown"], "o-", color=GROW_COLOR, lw=2,
                  label="cumulative grown")
    ax_churn.plot(df["episode"], df["cum_pruned"], "o-", color=PRUNE_COLOR, lw=2,
                  label="cumulative pruned")
    ax_churn.set_ylabel("connections")
    ax_churn.set_title("Phase 5: connections change over training (representative seed)")
    ax_churn.legend(loc="upper left", framealpha=0.9)

    # --- live population vs firing rate -----------------------------------
    ax_live.plot(df["episode"], df["live_structural"], "o-", color=LIVE_COLOR, lw=2,
                 label="live structural edges")
    ax_live.set_xlabel("episode")
    ax_live.set_ylabel("live structural edges", color=LIVE_COLOR)
    ax_live.tick_params(axis="y", labelcolor=LIVE_COLOR)

    ax_rate = ax_live.twinx()
    ax_rate.plot(df["episode"], df["mean_rate"], "s--", color=RATE_COLOR, lw=1.5,
                 label="mean firing rate")
    ax_rate.set_ylabel("mean firing rate (EMA)", color=RATE_COLOR)
    ax_rate.tick_params(axis="y", labelcolor=RATE_COLOR)
    ax_rate.set_ylim(0, max(0.25, df["mean_rate"].max() * 1.2))

    ax_live.set_title("rewiring while activity stays stable")

    fig.savefig(args.out, dpi=130)
    print(
        f"wrote {args.out}  "
        f"(grown={df['cum_grown'].iloc[-1]}, pruned={df['cum_pruned'].iloc[-1]}, "
        f"live {df['live_structural'].iloc[0]}->{df['live_structural'].iloc[-1]})"
    )


if __name__ == "__main__":
    main()
