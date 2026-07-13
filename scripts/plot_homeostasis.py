# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "pandas"]
# ///
"""Plot the Phase 2 perturbation experiment from perturb.csv.

Run `zig build perturb` first to produce perturb.csv, then:

    uv run scripts/plot_homeostasis.py            # -> perturb.png
    uv run scripts/plot_homeostasis.py -o x.png

Top panel: population firing rate over time, homeostasis ON vs OFF, with the
target band shaded and the perturbation onset marked. The story is one glance:
ON leaves the band when the drive hits and climbs back in; OFF stays pinned
above it. Bottom panel: the mean adaptive threshold -- the controller's output,
rising to reject the perturbation.
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
BAND_COLOR = "#2ca02c"

# Must match src/perturb.zig.
TARGET_RATE = 0.05
BAND_LO = TARGET_RATE * 0.5
BAND_HI = TARGET_RATE * 1.6


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=Path("perturb.csv"))
    ap.add_argument("-o", "--out", type=Path, default=Path("perturb.png"))
    args = ap.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"error: {args.csv} not found -- run `zig build perturb` first")

    df = pd.read_csv(args.csv)
    onset = df.loc[df["phase"] == "perturb", "t"].min()

    fig, (ax_r, ax_t) = plt.subplots(
        2, 1, figsize=(11, 6.5), height_ratios=[2.2, 1], sharex=True,
        constrained_layout=True,
    )

    # --- firing rate ------------------------------------------------------
    ax_r.axhspan(BAND_LO, BAND_HI, color=BAND_COLOR, alpha=0.12, label="target band")
    ax_r.axhline(TARGET_RATE, color=BAND_COLOR, lw=0.8, ls="--", alpha=0.7)
    ax_r.plot(df["t"], df["rate_on"], color=ON_COLOR, lw=1.3, label="homeostasis ON")
    ax_r.plot(df["t"], df["rate_off"], color=OFF_COLOR, lw=1.3, label="homeostasis OFF (control)")
    ax_r.axvline(onset, color="#444444", lw=1.0, ls=":")
    ax_r.annotate("perturbation on", xy=(onset, ax_r.get_ylim()[1]),
                  xytext=(6, -12), textcoords="offset points", fontsize=9, color="#444444")
    ax_r.set_ylabel("firing rate  (rho, spikes/neuron/step)")
    ax_r.set_title("Phase 2 homeostasis: recovery after a sustained perturbation")
    ax_r.legend(loc="center right", framealpha=0.9)

    # --- controller output (mean threshold) -------------------------------
    ax_t.plot(df["t"], df["thresh_on"], color=ON_COLOR, lw=1.3, label="mean threshold (ON)")
    ax_t.plot(df["t"], df["thresh_off"], color=OFF_COLOR, lw=1.0, alpha=0.6,
              label="mean threshold (OFF)")
    ax_t.axvline(onset, color="#444444", lw=1.0, ls=":")
    ax_t.set_ylabel("mean threshold")
    ax_t.set_xlabel("timestep")
    ax_t.legend(loc="center right", framealpha=0.9)

    fig.savefig(args.out, dpi=130)
    on_final = df["rate_on"].iloc[-100:].mean()
    off_final = df["rate_off"].iloc[-100:].mean()
    print(f"wrote {args.out}  (final rate: ON={on_final:.4f}, OFF={off_final:.4f}, "
          f"band=[{BAND_LO:.4f},{BAND_HI:.4f}])")


if __name__ == "__main__":
    main()
