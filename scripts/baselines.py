# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas"]
# ///
"""Stage 1 #4b — conventional external baselines for the two-choice family.

Compares three external models against the SNN's niches (report.md Stage 1 /
final.md Track E):

  1. Tiny Elman RNN trained online with BPTT (full-sequence backprop each episode)
  2. Echo-state network (fixed random reservoir + online linear readout / LMS)
  3. Tabular / feature logistic baseline (one-hot stimulus → action; no temporal
     memory — fails delay by construction)

Protocols mirror the SNN harnesses as closely as a non-spiking model can:

  - immediate association  (train.zig / instrument cost track)
  - delayed association    (delay.zig style: stim → silence → readout)
  - forgetting             (train A, force B, retest A — instrument/continual)
  - distribution shift     (mid-run A↔B mapping flip — instrument)

Metrics (not only final accuracy):
  - final accuracy, episodes-to-criterion
  - online update cost (ops accounting per episode)
  - parameter counts (total vs trained)
  - activity sparsity (mean |h| and active-unit fraction)
  - forgetting (A-retest after B-only disuse)
  - shift recovery (pre / drop / post)

This is an *external* baseline, not part of the Zig SNN. Run:

    uv run scripts/baselines.py
    # -> baseline.csv, baseline_curves.csv, baseline.meta.json

Optional plot:

    uv run scripts/plot_baselines.py
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Protocol constants (aligned with SNN harnesses where practical)
# ---------------------------------------------------------------------------

SEEDS = list(range(1, 9))  # match instrument.zig
STIM_STEPS = 40
READOUT_STEPS = 5  # last few steps carry the decision target
DELAY_STEPS = 20  # Phase-4-style delay probe
TRAIN_EPISODES = 1200
FINAL_WINDOW = 200
CRITERION = 0.70  # rolling-block mean used for episodes-to-criterion
BLOCK = 50

FORGET_A = 800
FORGET_B = 600
FORGET_RETEST = 80

SHIFT_PRE = 1000
SHIFT_POST = 1000

# Model sizes — intentionally tiny vs the 100-neuron SNN, but with full BPTT.
RNN_HIDDEN = 32
ESN_HIDDEN = 64
ESN_SPECTRAL = 0.9
ESN_INPUT_SCALE = 1.0
LR_RNN = 0.05
LR_ESN = 0.05
LR_TABULAR = 0.2

# Soft sanity: baselines should solve immediate association.
PASS_IMMEDIATE = 0.90


# ---------------------------------------------------------------------------
# Task helpers
# ---------------------------------------------------------------------------


def correct_action(choice: int, flipped: bool = False) -> int:
    """choice 0=A, 1=B; fixed map A→0, B→1 unless flipped."""
    base = choice  # A→0, B→1
    return base ^ 1 if flipped else base


def make_sequence(
    choice: int,
    *,
    delay_steps: int = 0,
    stim_steps: int = STIM_STEPS,
) -> np.ndarray:
    """Return (T, 2) input sequence: one-hot stimulus, then optional silence."""
    x = np.zeros((stim_steps + delay_steps, 2), dtype=np.float64)
    x[:stim_steps, choice] = 1.0
    return x


def rng_choice(seed: int, episode: int) -> int:
    """Deterministic A/B draw, independent of model RNG draws."""
    # Mix seed/episode with a simple LCG — no uint overflow warnings.
    z = (seed * 1_000_003 + episode * 97_531 + 0x9E3779B9) & 0xFFFFFFFF
    z = (z ^ (z >> 16)) * 0x7FEB352D & 0xFFFFFFFF
    z = (z ^ (z >> 15)) * 0x846CA68B & 0xFFFFFFFF
    z = z ^ (z >> 16)
    return int(z & 1)


def summarize(xs: list[float] | np.ndarray) -> dict[str, float]:
    a = np.asarray(xs, dtype=np.float64)
    if a.size == 0:
        return {"mean": float("nan"), "ci_half": float("nan"), "n": 0}
    mean = float(a.mean())
    if a.size < 2:
        return {"mean": mean, "ci_half": float("nan"), "n": int(a.size)}
    sd = float(a.std(ddof=1))
    ci = 1.96 * sd / math.sqrt(a.size)
    return {"mean": mean, "ci_half": ci, "n": int(a.size)}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass
class Cost:
    """Per-episode operation accounting (forward + backward/update)."""

    forward: float
    backward: float

    @property
    def total(self) -> float:
        return self.forward + self.backward


class TabularLogistic:
    """Memoryless logistic on the *final* frame only (no temporal integration).

    On a delay trial the last frame is silence, so this baseline has no access
    to the earlier stimulus — the fair no-memory control.  (Scanning for the
    last non-zero frame would cheat the delay protocol.)
    """

    name = "tabular"

    def __init__(self, seed: int):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.1, size=(2, 2))  # input → logits
        self.b = np.zeros(2)
        self.lr = LR_TABULAR
        self.n_params_total = 2 * 2 + 2
        self.n_params_trained = self.n_params_total

    def predict_and_update(
        self,
        x_seq: np.ndarray,
        target: int,
        *,
        train: bool,
    ) -> tuple[int, Cost, dict[str, float]]:
        x = x_seq[-1]
        logits = self.W.T @ x + self.b
        # stable softmax
        z = logits - logits.max()
        exp = np.exp(z)
        p = exp / exp.sum()
        pred = int(np.argmax(p))
        # ops: 2x2 matvec + soft + optional grad
        fwd = 2 * 2 * 2 + 10
        bwd = 0.0
        if train:
            dlogits = p.copy()
            dlogits[target] -= 1.0
            # W grad: outer(x, dlogits)
            self.W -= self.lr * np.outer(x, dlogits)
            self.b -= self.lr * dlogits
            bwd = 2 * 2 * 2 + 2 * 2
        sparsity = {
            "mean_abs_h": float(np.abs(x).mean()),
            "active_frac": float((np.abs(x) > 1e-6).mean()),
        }
        return pred, Cost(fwd, bwd), sparsity


class ESN:
    """Echo-state network: fixed tanh reservoir + online linear readout."""

    name = "esn"

    def __init__(self, seed: int, hidden: int = ESN_HIDDEN):
        rng = np.random.default_rng(seed + 10_000)
        self.H = hidden
        self.W_in = rng.normal(0, ESN_INPUT_SCALE / math.sqrt(2), size=(hidden, 2))
        W = rng.normal(0, 1.0 / math.sqrt(hidden), size=(hidden, hidden))
        # Scale to spectral radius.
        eig = np.linalg.eigvals(W)
        radius = float(np.max(np.abs(eig)))
        self.W = W * (ESN_SPECTRAL / max(radius, 1e-8))
        self.W_out = rng.normal(0, 0.01, size=(2, hidden))
        self.b_out = np.zeros(2)
        self.lr = LR_ESN
        self.n_params_total = hidden * 2 + hidden * hidden + 2 * hidden + 2
        self.n_params_trained = 2 * hidden + 2

    def predict_and_update(
        self,
        x_seq: np.ndarray,
        target: int,
        *,
        train: bool,
    ) -> tuple[int, Cost, dict[str, float]]:
        h = np.zeros(self.H)
        abs_sum = 0.0
        active_sum = 0.0
        T = x_seq.shape[0]
        for t in range(T):
            h = np.tanh(self.W_in @ x_seq[t] + self.W @ h)
            abs_sum += float(np.abs(h).mean())
            active_sum += float((np.abs(h) > 0.05).mean())
        logits = self.W_out @ h + self.b_out
        z = logits - logits.max()
        exp = np.exp(z)
        p = exp / exp.sum()
        pred = int(np.argmax(p))
        # Forward: each step H*2 + H*H + H tanh; final 2*H
        fwd = T * (self.H * 2 + self.H * self.H + self.H) + 2 * self.H + 10
        bwd = 0.0
        if train:
            dlogits = p.copy()
            dlogits[target] -= 1.0
            self.W_out -= self.lr * np.outer(dlogits, h)
            self.b_out -= self.lr * dlogits
            bwd = 2 * self.H * 2 + 2 * 2
        sparsity = {
            "mean_abs_h": abs_sum / T,
            "active_frac": active_sum / T,
        }
        return pred, Cost(fwd, bwd), sparsity


class ElmanBPTT:
    """Tiny Elman RNN with full BPTT and online SGD (cross-entropy at last step)."""

    name = "bptt_rnn"

    def __init__(self, seed: int, hidden: int = RNN_HIDDEN):
        rng = np.random.default_rng(seed + 20_000)
        self.H = hidden
        scale_x = 1.0 / math.sqrt(2)
        scale_h = 1.0 / math.sqrt(hidden)
        self.Wxh = rng.normal(0, scale_x, size=(hidden, 2))
        self.Whh = rng.normal(0, scale_h, size=(hidden, hidden))
        self.bh = np.zeros(hidden)
        self.Why = rng.normal(0, 1.0 / math.sqrt(hidden), size=(2, hidden))
        self.by = np.zeros(2)
        self.lr = LR_RNN
        self.n_params_total = hidden * 2 + hidden * hidden + hidden + 2 * hidden + 2
        self.n_params_trained = self.n_params_total

    def predict_and_update(
        self,
        x_seq: np.ndarray,
        target: int,
        *,
        train: bool,
    ) -> tuple[int, Cost, dict[str, float]]:
        T = x_seq.shape[0]
        H = self.H
        hs = [np.zeros(H)]
        preacts = []
        abs_sum = 0.0
        active_sum = 0.0
        for t in range(T):
            a = self.Wxh @ x_seq[t] + self.Whh @ hs[-1] + self.bh
            h = np.tanh(a)
            preacts.append(a)
            hs.append(h)
            abs_sum += float(np.abs(h).mean())
            active_sum += float((np.abs(h) > 0.05).mean())
        hT = hs[-1]
        logits = self.Why @ hT + self.by
        z = logits - logits.max()
        exp = np.exp(z)
        p = exp / exp.sum()
        pred = int(np.argmax(p))
        # Forward ops roughly.
        fwd = T * (H * 2 + H * H + H) + 2 * H + 10
        bwd = 0.0
        if train:
            dlogits = p.copy()
            dlogits[target] -= 1.0
            dWhy = np.outer(dlogits, hT)
            dby = dlogits
            dh = self.Why.T @ dlogits
            dWxh = np.zeros_like(self.Wxh)
            dWhh = np.zeros_like(self.Whh)
            dbh = np.zeros_like(self.bh)
            for t in reversed(range(T)):
                # dtanh = (1 - h^2) * dh
                h = hs[t + 1]
                h_prev = hs[t]
                da = (1.0 - h * h) * dh
                dWxh += np.outer(da, x_seq[t])
                dWhh += np.outer(da, h_prev)
                dbh += da
                dh = self.Whh.T @ da
                # rough bwd ops per step
                bwd += H * 2 + H * H + H * H + 3 * H
            bwd += 2 * H * 2 + 2
            # clip grads lightly
            for g in (dWxh, dWhh, dbh, dWhy, dby):
                np.clip(g, -5.0, 5.0, out=g)
            self.Wxh -= self.lr * dWxh
            self.Whh -= self.lr * dWhh
            self.bh -= self.lr * dbh
            self.Why -= self.lr * dWhy
            self.by -= self.lr * dby
        sparsity = {
            "mean_abs_h": abs_sum / T,
            "active_frac": active_sum / T,
        }
        return pred, Cost(fwd, bwd), sparsity


ModelFactory = Callable[[int], TabularLogistic | ESN | ElmanBPTT]


MODELS: dict[str, ModelFactory] = {
    "tabular": TabularLogistic,
    "esn": ESN,
    "bptt_rnn": ElmanBPTT,
}


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


def run_immediate(model, seed: int) -> dict:
    correct = []
    costs = []
    spars = []
    episodes_to_crit = None
    ring: list[int] = []
    for ep in range(TRAIN_EPISODES):
        choice = rng_choice(seed, ep)
        target = correct_action(choice, flipped=False)
        x = make_sequence(choice, delay_steps=0)
        pred, cost, sp = model.predict_and_update(x, target, train=True)
        ok = int(pred == target)
        correct.append(ok)
        costs.append(cost.total)
        spars.append(sp)
        ring.append(ok)
        if len(ring) > BLOCK:
            ring.pop(0)
        if episodes_to_crit is None and len(ring) == BLOCK and sum(ring) / BLOCK >= CRITERION:
            episodes_to_crit = ep + 1
    final = float(np.mean(correct[-FINAL_WINDOW:]))
    return {
        "protocol": "immediate",
        "final_accuracy": final,
        "episodes_to_criterion": episodes_to_crit if episodes_to_crit is not None else TRAIN_EPISODES,
        "mean_ops": float(np.mean(costs)),
        "mean_abs_h": float(np.mean([s["mean_abs_h"] for s in spars[-FINAL_WINDOW:]])),
        "active_frac": float(np.mean([s["active_frac"] for s in spars[-FINAL_WINDOW:]])),
        "n_params_total": model.n_params_total,
        "n_params_trained": model.n_params_trained,
    }


def run_delay(model, seed: int, delay_steps: int = DELAY_STEPS) -> dict:
    correct = []
    for ep in range(TRAIN_EPISODES):
        choice = rng_choice(seed, ep)
        target = correct_action(choice, flipped=False)
        x = make_sequence(choice, delay_steps=delay_steps)
        pred, _, _ = model.predict_and_update(x, target, train=True)
        correct.append(int(pred == target))
    return {
        "protocol": f"delay_{delay_steps}",
        "final_accuracy": float(np.mean(correct[-FINAL_WINDOW:])),
        "episodes_to_criterion": float("nan"),
        "mean_ops": float("nan"),
        "mean_abs_h": float("nan"),
        "active_frac": float("nan"),
        "n_params_total": model.n_params_total,
        "n_params_trained": model.n_params_trained,
    }


def run_forgetting(model_factory: ModelFactory, seed: int) -> dict:
    """B-only disuse after full training — orthogonal inputs often retain A.

    This matches the SNN instrument forgetting protocol. Orthogonal A/B channels
    mean BPTT/tabular may *not* forget (honest negative for 'catastrophic
    forgetting under disuse'). Structural pruning is the SNN's extra pressure.
    """
    model = model_factory(seed)
    for ep in range(FORGET_A):
        choice = rng_choice(seed, ep)
        target = correct_action(choice)
        x = make_sequence(choice)
        model.predict_and_update(x, target, train=True)

    def retest_a() -> float:
        ok = 0
        for _ in range(FORGET_RETEST):
            x = make_sequence(0)
            pred, _, _ = model.predict_and_update(x, 0, train=False)
            ok += int(pred == 0)
        return ok / FORGET_RETEST

    retest_start = retest_a()
    for _ in range(FORGET_B):
        x = make_sequence(1)
        model.predict_and_update(x, 1, train=True)
    retest_end = retest_a()
    return {
        "protocol": "forgetting",
        "final_accuracy": retest_end,
        "retest_start": retest_start,
        "retest_end": retest_end,
        "forget_drop": retest_start - retest_end,
        "episodes_to_criterion": float("nan"),
        "mean_ops": float("nan"),
        "mean_abs_h": float("nan"),
        "active_frac": float("nan"),
        "n_params_total": model.n_params_total,
        "n_params_trained": model.n_params_trained,
    }


def run_overwrite(model_factory: ModelFactory, seed: int) -> dict:
    """Same-input label flip (A→0 then A→1): pure catastrophic overwrite probe."""
    model = model_factory(seed)
    for ep in range(FORGET_A):
        x = make_sequence(0)
        model.predict_and_update(x, 0, train=True)

    def retest_old() -> float:
        ok = 0
        for _ in range(FORGET_RETEST):
            pred, _, _ = model.predict_and_update(make_sequence(0), 0, train=False)
            ok += int(pred == 0)
        return ok / FORGET_RETEST

    retest_start = retest_old()
    for _ in range(FORGET_B):
        model.predict_and_update(make_sequence(0), 1, train=True)  # flip label
    retest_end = retest_old()
    return {
        "protocol": "overwrite",
        "final_accuracy": retest_end,
        "retest_start": retest_start,
        "retest_end": retest_end,
        "forget_drop": retest_start - retest_end,
        "episodes_to_criterion": float("nan"),
        "mean_ops": float("nan"),
        "mean_abs_h": float("nan"),
        "active_frac": float("nan"),
        "n_params_total": model.n_params_total,
        "n_params_trained": model.n_params_trained,
    }


def run_shift(model, seed: int, curve_rows: list[dict]) -> dict:
    block_ok = 0
    pre = drop = post = float("nan")
    recover = None
    total = SHIFT_PRE + SHIFT_POST
    for ep in range(total):
        flipped = ep >= SHIFT_PRE
        choice = rng_choice(seed, ep)
        target = correct_action(choice, flipped=flipped)
        x = make_sequence(choice)
        pred, _, _ = model.predict_and_update(x, target, train=True)
        block_ok += int(pred == target)
        if (ep + 1) % BLOCK == 0:
            acc = block_ok / BLOCK
            phase = "post_shift" if flipped else "pre_shift"
            curve_rows.append(
                {
                    "seed": seed,
                    "model": model.name,
                    "protocol": "shift",
                    "episode": ep + 1,
                    "phase": phase,
                    "block_accuracy": acc,
                }
            )
            if not flipped:
                pre = acc
            else:
                post_ep = ep + 1 - SHIFT_PRE
                if post_ep == BLOCK:
                    drop = acc
                post = acc
                if recover is None and acc >= CRITERION:
                    recover = post_ep
            block_ok = 0
    return {
        "protocol": "shift",
        "final_accuracy": post,
        "pre_acc": pre,
        "drop_acc": drop,
        "post_acc": post,
        "episodes_to_recover": recover if recover is not None else SHIFT_POST,
        "episodes_to_criterion": float("nan"),
        "mean_ops": float("nan"),
        "mean_abs_h": float("nan"),
        "active_frac": float("nan"),
        "n_params_total": model.n_params_total,
        "n_params_trained": model.n_params_trained,
    }


# ---------------------------------------------------------------------------
# SNN reference numbers (from Stage 1 instrument / published harnesses)
# ---------------------------------------------------------------------------

# These are the measured SNN figures from Stage 1 instrumentation / train /
# continual — used as a side-by-side reference in the summary, not re-simulated.
SNN_REFERENCE = {
    "immediate_accuracy": 0.980,
    "cost_ratio_local_vs_dense": 0.336,
    "mean_firing_rate": 0.098,
    "forgetting_cons_on": 1.000,
    "forgetting_cons_off": 0.816,
    "shift_pre": 0.930,
    "shift_drop": 0.590,
    "shift_post": 0.935,
    "n_plastic": 256,
    "n_live": 1049,
    "n_neurons": 100,
    "notes": "from instrument.zig / train-style two-choice; delay numbers vary by delay.zig",
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    curve_rows: list[dict] = []
    t0 = time.time()

    print("\n-- Stage 1 external baselines (tabular / ESN / BPTT RNN) --\n")

    for model_name, factory in MODELS.items():
        print(f"  model: {model_name}")
        for seed in args.seeds:
            # Immediate
            m = factory(seed)
            r = run_immediate(m, seed)
            r.update({"seed": seed, "model": model_name})
            rows.append(r)

            # Delay (fresh model)
            m = factory(seed)
            r = run_delay(m, seed, DELAY_STEPS)
            r.update({"seed": seed, "model": model_name})
            rows.append(r)

            # Forgetting / overwrite (fresh models)
            r = run_forgetting(factory, seed)
            r.update({"seed": seed, "model": model_name})
            rows.append(r)
            r = run_overwrite(factory, seed)
            r.update({"seed": seed, "model": model_name})
            rows.append(r)

            # Shift (fresh)
            m = factory(seed)
            r = run_shift(m, seed, curve_rows)
            r.update({"seed": seed, "model": model_name})
            rows.append(r)

            imm = [x for x in rows if x["seed"] == seed and x["model"] == model_name and x["protocol"] == "immediate"][0]
            dly = [x for x in rows if x["seed"] == seed and x["model"] == model_name and x["protocol"].startswith("delay")][0]
            frg = [x for x in rows if x["seed"] == seed and x["model"] == model_name and x["protocol"] == "forgetting"][0]
            ovr = [x for x in rows if x["seed"] == seed and x["model"] == model_name and x["protocol"] == "overwrite"][0]
            sh = [x for x in rows if x["seed"] == seed and x["model"] == model_name and x["protocol"] == "shift"][0]
            print(
                f"    seed {seed}: imm {imm['final_accuracy']:.3f}  "
                f"delay{DELAY_STEPS} {dly['final_accuracy']:.3f}  "
                f"disuse_drop {frg.get('forget_drop', float('nan')):.3f}  "
                f"overwrite_drop {ovr.get('forget_drop', float('nan')):.3f}  "
                f"shift {sh.get('pre_acc', float('nan')):.2f}→{sh.get('drop_acc', float('nan')):.2f}→{sh.get('post_acc', float('nan')):.2f}  "
                f"ops {imm['mean_ops']:.0f}  params {imm['n_params_trained']}/{imm['n_params_total']}"
            )

    df = pd.DataFrame(rows)
    curves = pd.DataFrame(curve_rows)
    csv_path = out / "baseline.csv"
    curves_path = out / "baseline_curves.csv"
    df.to_csv(csv_path, index=False)
    curves.to_csv(curves_path, index=False)

    # Aggregate summary
    print("\n  ========== summary (mean ± 95% CI half) ==========\n")
    summary: dict = {"snn_reference": SNN_REFERENCE, "models": {}}
    all_imm_ok = True
    for model_name in MODELS:
        sub = df[df["model"] == model_name]
        imm = sub[sub["protocol"] == "immediate"]
        dly = sub[sub["protocol"] == f"delay_{DELAY_STEPS}"]
        frg = sub[sub["protocol"] == "forgetting"]
        ovr = sub[sub["protocol"] == "overwrite"]
        sh = sub[sub["protocol"] == "shift"]

        s_imm = summarize(imm["final_accuracy"].tolist())
        s_dly = summarize(dly["final_accuracy"].tolist())
        s_ops = summarize(imm["mean_ops"].tolist())
        s_act = summarize(imm["active_frac"].tolist())
        s_forget = summarize(frg["final_accuracy"].tolist())
        s_drop = summarize(frg["forget_drop"].tolist())
        s_ovr = summarize(ovr["final_accuracy"].tolist())
        s_ovr_drop = summarize(ovr["forget_drop"].tolist())
        s_pre = summarize(sh["pre_acc"].tolist())
        s_d = summarize(sh["drop_acc"].tolist())
        s_post = summarize(sh["post_acc"].tolist())
        params_t = int(imm["n_params_trained"].iloc[0])
        params_tot = int(imm["n_params_total"].iloc[0])

        if s_imm["mean"] < PASS_IMMEDIATE:
            all_imm_ok = False

        summary["models"][model_name] = {
            "immediate": s_imm,
            "delay": s_dly,
            "mean_ops": s_ops,
            "active_frac": s_act,
            "forgetting_retest": s_forget,
            "forget_drop": s_drop,
            "overwrite_retest": s_ovr,
            "overwrite_drop": s_ovr_drop,
            "shift_pre": s_pre,
            "shift_drop": s_d,
            "shift_post": s_post,
            "n_params_trained": params_t,
            "n_params_total": params_tot,
        }

        print(f"  {model_name}")
        print(f"    immediate accuracy     {s_imm['mean']:.3f} ± {s_imm['ci_half']:.3f}")
        print(f"    delay-{DELAY_STEPS} accuracy      {s_dly['mean']:.3f} ± {s_dly['ci_half']:.3f}")
        print(f"    online ops / episode   {s_ops['mean']:.0f} ± {s_ops['ci_half']:.0f}")
        print(f"    active hidden frac     {s_act['mean']:.3f} ± {s_act['ci_half']:.3f}")
        print(
            f"    B-disuse A-retest      {s_forget['mean']:.3f} ± {s_forget['ci_half']:.3f}  (drop {s_drop['mean']:.3f})"
        )
        print(
            f"    overwrite old-retest   {s_ovr['mean']:.3f} ± {s_ovr['ci_half']:.3f}  (drop {s_ovr_drop['mean']:.3f})"
        )
        print(
            f"    shift pre/drop/post    {s_pre['mean']:.3f} / {s_d['mean']:.3f} / {s_post['mean']:.3f}"
        )
        print(f"    params trained/total   {params_t} / {params_tot}")
        print()

    print("  SNN reference (instrument / two-choice, not re-run here)")
    print(f"    immediate accuracy     {SNN_REFERENCE['immediate_accuracy']:.3f}")
    print(f"    mean firing rate       {SNN_REFERENCE['mean_firing_rate']:.3f}")
    print(f"    local/dense cost ratio {SNN_REFERENCE['cost_ratio_local_vs_dense']:.3f}")
    print(
        f"    forgetting on/off      {SNN_REFERENCE['forgetting_cons_on']:.3f} / {SNN_REFERENCE['forgetting_cons_off']:.3f}"
    )
    print(
        f"    shift pre/drop/post    {SNN_REFERENCE['shift_pre']:.3f} / {SNN_REFERENCE['shift_drop']:.3f} / {SNN_REFERENCE['shift_post']:.3f}"
    )
    print(f"    plastic / live         {SNN_REFERENCE['n_plastic']} / {SNN_REFERENCE['n_live']}")
    print()

    # Comparative claims (honest, qualitative gates)
    bptt = summary["models"]["bptt_rnn"]
    esn = summary["models"]["esn"]
    tab = summary["models"]["tabular"]
    # SNN local accounting from instrument (~58k ops/ep at 40 steps, 256 plastic).
    snn_local_ops_ref = 58_496.0
    claims = {
        "bptt_solves_immediate": bptt["immediate"]["mean"] >= PASS_IMMEDIATE,
        "esn_solves_immediate": esn["immediate"]["mean"] >= PASS_IMMEDIATE,
        "tabular_solves_immediate": tab["immediate"]["mean"] >= PASS_IMMEDIATE,
        "tabular_fails_delay": tab["delay"]["mean"] < 0.60,
        "bptt_handles_delay": bptt["delay"]["mean"] >= 0.80,
        "esn_handles_delay": esn["delay"]["mean"] >= 0.70,
        "bptt_overwrites_old_label": bptt["overwrite_drop"]["mean"] > 0.40,
        "bptt_ops_gt_snn_local": bptt["mean_ops"]["mean"] > snn_local_ops_ref,
        "snn_local_ops_ref": snn_local_ops_ref,
    }
    summary["claims"] = claims
    summary["protocol"] = {
        "seeds": list(args.seeds),
        "stim_steps": STIM_STEPS,
        "delay_steps": DELAY_STEPS,
        "train_episodes": TRAIN_EPISODES,
        "forget_a": FORGET_A,
        "forget_b": FORGET_B,
        "shift_pre": SHIFT_PRE,
        "shift_post": SHIFT_POST,
        "rnn_hidden": RNN_HIDDEN,
        "esn_hidden": ESN_HIDDEN,
        "elapsed_sec": time.time() - t0,
    }

    # Status: baselines are valid if BPTT+ESN solve immediate and tabular fails delay.
    valid = (
        claims["bptt_solves_immediate"]
        and claims["esn_solves_immediate"]
        and claims["tabular_solves_immediate"]
        and claims["tabular_fails_delay"]
    )
    status = (
        "COMPLETE — external baselines written; BPTT/ESN solve immediate, "
        "tabular is the no-memory control."
        if valid
        else "COMPLETE WITH WARNINGS — inspect summary claims; CSVs still written."
    )
    summary["status"] = status

    meta_path = out / "baseline.meta.json"
    meta_path.write_text(json.dumps(summary, indent=2, default=float) + "\n")

    print(f"  STATUS: {status}")
    print(f"  wrote {csv_path}, {curves_path}, {meta_path}")
    print(f"  elapsed {time.time() - t0:.1f}s\n")


if __name__ == "__main__":
    main()
