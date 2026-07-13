# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "pandas"]
# ///
"""Plot the Phase 6 consolidation / continual-learning results.

Run `zig build continual` first (writes continual.csv), then:

    uv run scripts/plot_continual.py         # -> continual.png

Top panel: A-retest accuracy after block B (train A -> train B -> retest A), with
consolidation on vs off, per seed. With consolidation the previously-useful A
pathway survives block B's disuse, so A is still read out correctly; without it,
A is forgotten.

Bottom panel: survival of the two permanence bands (consolidation on). Synapses
classified at the end of block A as *consolidated* (previously useful) almost all
survive block B; those classified *tentative* (never consolidated) are pruned away
-- the exit criterion, made visible.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ON_COLOR = "#1f77b4"
OFF_COLOR = "#d62728"
CONS_COLOR = "#2ca02c"
TENT_COLOR = "#d62728"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=Path("continual.csv"))
    ap.add_argument("-o", "--out", type=Path, default=Path("continual.png"))
    args = ap.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"error: {args.csv} not found -- run `zig build continual` first")

    df = pd.read_csv(args.csv)
    seeds = sorted(df["seed"].unique())
    x = range(len(seeds))
    w = 0.38

    fig, (ax_acc, ax_surv) = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)

    # --- A-retest accuracy: on vs off -------------------------------------
    on = df[df["consolidation"] == "on"].set_index("seed")
    off = df[df["consolidation"] == "off"].set_index("seed")
    ax_acc.bar([i - w / 2 for i in x], [on.loc[s, "retest_a"] for s in seeds], w,
               color=ON_COLOR, label="consolidation on")
    ax_acc.bar([i + w / 2 for i in x], [off.loc[s, "retest_a"] for s in seeds], w,
               color=OFF_COLOR, label="consolidation off")
    ax_acc.axhline(0.5, ls="--", lw=1.0, color="#666666", label="chance")
    ax_acc.set_ylim(0, 1.05)
    ax_acc.set_xticks(list(x))
    ax_acc.set_xticklabels([f"seed {s}" for s in seeds])
    ax_acc.set_ylabel("A-retest accuracy")
    ax_acc.set_title("Phase 6: forgetting of task A after training task B")
    ax_acc.legend(loc="lower right", framealpha=0.9)

    # --- band survival (consolidation on) ---------------------------------
    ax_surv.bar([i - w / 2 for i in x], [on.loc[s, "consolidated_survival"] for s in seeds], w,
                color=CONS_COLOR, label="consolidated (previously useful)")
    ax_surv.bar([i + w / 2 for i in x], [on.loc[s, "tentative_survival"] for s in seeds], w,
                color=TENT_COLOR, label="tentative (never consolidated)")
    ax_surv.set_ylim(0, 1.05)
    ax_surv.set_xticks(list(x))
    ax_surv.set_xticklabels([f"seed {s}" for s in seeds])
    ax_surv.set_ylabel("fraction surviving block B")
    ax_surv.set_title("pathway survival by permanence band (consolidation on)")
    ax_surv.legend(loc="center right", framealpha=0.9)

    fig.savefig(args.out, dpi=130)
    print(
        f"wrote {args.out}  "
        f"(retest on={on['retest_a'].mean():.3f} off={off['retest_a'].mean():.3f}; "
        f"survival consolidated={on['consolidated_survival'].mean():.2f} "
        f"tentative={on['tentative_survival'].mean():.2f})"
    )


if __name__ == "__main__":
    main()
