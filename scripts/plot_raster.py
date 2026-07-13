# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "pandas"]
# ///
"""Plot the artefacts of a run: spike raster + population activity over time.

This is the thing you actually look at to decide whether the dynamics are
healthy -- the CSVs are hard to read by eye. Run it after `zig build run`
(or after `brain <config.json>`).

    uv run scripts/plot_raster.py                 # reads ./raster.csv, ./metrics.csv
    uv run scripts/plot_raster.py --dir some/run  # read artefacts from another dir
    uv run scripts/plot_raster.py -o out.png      # choose the output path

Top panel: one dot per spike (t vs neuron id), excitatory and inhibitory in
different colors. Bottom panel: spikes-per-step and the E/I current balance,
the two traces that tell you *why* a run died or exploded.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write a file, never try to open a window
import matplotlib.pyplot as plt
import pandas as pd

EXC_COLOR = "#1f77b4"
INH_COLOR = "#d62728"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("."),
                    help="directory holding raster.csv and metrics.csv (default: .)")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output image path (default: <dir>/raster.png)")
    args = ap.parse_args()

    raster_path = args.dir / "raster.csv"
    metrics_path = args.dir / "metrics.csv"
    for p in (raster_path, metrics_path):
        if not p.exists():
            raise SystemExit(f"error: {p} not found -- run `zig build run` first")

    out_path = args.out or (args.dir / "raster.png")

    raster = pd.read_csv(raster_path)
    metrics = pd.read_csv(metrics_path)

    fig, (ax_r, ax_m) = plt.subplots(
        2, 1, figsize=(11, 7), height_ratios=[3, 1.4], sharex=True,
        constrained_layout=True,
    )

    # --- raster -----------------------------------------------------------
    exc = raster[raster["kind"] == "excitatory"]
    inh = raster[raster["kind"] == "inhibitory"]
    ax_r.scatter(exc["t"], exc["neuron"], s=2, c=EXC_COLOR, label="excitatory", linewidths=0)
    ax_r.scatter(inh["t"], inh["neuron"], s=2, c=INH_COLOR, label="inhibitory", linewidths=0)
    ax_r.set_ylabel("neuron id")
    ax_r.set_title(f"spike raster  ({len(raster)} spikes)")
    ax_r.legend(loc="upper right", markerscale=4, framealpha=0.9)

    # --- population activity ----------------------------------------------
    ax_m.plot(metrics["t"], metrics["spikes"], lw=0.8, color="#333333", label="spikes/step")
    mean_rate = metrics["spikes"].mean()
    ax_m.axhline(mean_rate, ls="--", lw=0.8, color="#888888",
                 label=f"mean {mean_rate:.2f}")
    ax_m.set_ylabel("spikes / step")
    ax_m.set_xlabel("timestep")

    # E/I current balance on a twin axis -- the "why" trace.
    ax_ei = ax_m.twinx()
    ax_ei.plot(metrics["t"], metrics["exc_current"], lw=0.7, color=EXC_COLOR, alpha=0.6)
    ax_ei.plot(metrics["t"], metrics["inh_current"], lw=0.7, color=INH_COLOR, alpha=0.6)
    ax_ei.set_ylabel("E / I current", color="#666666")

    ax_m.legend(loc="upper right", framealpha=0.9)

    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}  ({len(raster)} spikes over {len(metrics)} steps)")


if __name__ == "__main__":
    main()
