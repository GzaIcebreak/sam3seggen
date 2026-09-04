"""Render a one-page PNG summary of a finetune/train.py run.

Reads the run's own artifacts (log.jsonl, check_step*.json), an optional step-0 baseline
check, and any fidelity reports, then writes a 4-panel figure:
holdout loss per condition kind, training loss, step time, and colour fidelity vs purity.

    finetune\run_ft.bat plot_run.py --run finetune\runs\pv_v1 ^
        --check0 finetune\runs\pv_check0\check_loss.json ^
        --fidelity E:\data\pv\<obj>\variants\sam3_az0\fid_base.json ^
        --out E:\data\pv_v1_summary.png
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_stats import KINDS, read_checks, read_fidelity, read_log, read_run_args, step_times

COLORS = {"clean": "#4c72b0", "corrupt": "#dd8452", "sam3": "#55a868"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--check0", default=None, help="step-0 baseline check_loss.json")
    parser.add_argument("--fidelity", nargs="*", default=None, help="eval_fidelity.py reports (base ckpt)")
    parser.add_argument("--fidelity_after", nargs="*", default=None,
                        help="Same objects evaluated with the finetuned ckpt; draws before/after pairs")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows = read_log(args.run)
    checks = read_checks(args.run, args.check0)
    max_steps = read_run_args(args.run).get("max_steps", 0)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    fig.suptitle(f"{os.path.basename(os.path.normpath(args.run))}: step {rows[-1]['step']}/{max_steps}",
                 fontsize=13, fontweight="bold")

    ax = axes[0][0]
    if checks:
        labels = [("base" if s == 0 else str(s)) for s, _ in checks]
        width = 0.8 / max(1, len(KINDS))
        for i, kind in enumerate(KINDS):
            vals = [summary.get(kind, {}).get("mean") for _, summary in checks]
            xs = [j + (i - 1) * width for j in range(len(checks))]
            bars = ax.bar(xs, [v or 0 for v in vals], width, label=kind, color=COLORS[kind])
            ax.bar_label(bars, fmt="%.4f", fontsize=7, padding=1)
        ax.set_xticks(range(len(checks)), labels)
        ax.legend(title="condition kind", fontsize=8, title_fontsize=8)
    ax.set_title("Holdout loss by condition kind (lower is better)", fontsize=10)
    ax.set_xlabel("checkpoint (step)")
    ax.set_ylabel("v-pred MSE, fixed t and noise")

    ax = axes[0][1]
    ax.plot([r["step"] for r in rows], [r["ema"] for r in rows], color="#4c72b0", lw=1.6)
    ax.set_title("Training loss (EMA 0.98)", fontsize=10)
    ax.set_xlabel("training step")
    ax.set_ylabel("v-pred MSE")
    ax.grid(alpha=0.25)

    ax = axes[1][0]
    steps, secs = step_times(rows)
    ax.plot(steps, secs, color="#dd8452", lw=1.4)
    if secs:
        base = min(secs)
        ax.axhline(base, ls="--", lw=1, color="#4c72b0", label=f"best {base:.1f} s/step")
        ax.legend(fontsize=8)
    ax.set_title("Wall-clock per step (spikes = GPU shared with another process)", fontsize=10)
    ax.set_xlabel("training step")
    ax.set_ylabel("seconds per step")
    ax.grid(alpha=0.25)

    ax = axes[1][1]
    reports = read_fidelity(args.fidelity)
    after = dict(read_fidelity(args.fidelity_after))
    if reports:
        xs = range(len(reports))
        if after:
            width = 0.26
            groups = [
                ("fidelity, base", [s["mean_fidelity"] for _, s in reports], "#c44e52", -1.5),
                ("fidelity, finetuned", [after[n]["mean_fidelity"] if n in after else 0 for n, _ in reports], "#8172b3", -0.5),
                ("purity, base", [s["mean_purity"] for _, s in reports], "#55a868", 0.5),
                ("purity, finetuned", [after[n]["mean_purity"] if n in after else 0 for n, _ in reports], "#4c72b0", 1.5),
            ]
            title = "Colour fidelity vs purity: base vs finetuned ckpt (sam3 conditions)"
        else:
            width = 0.38
            groups = [
                ("fidelity (colour matches 2D map)", [s["mean_fidelity"] for _, s in reports], "#c44e52", -0.5),
                ("purity (one solid colour per part)", [s["mean_purity"] for _, s in reports], "#55a868", 0.5),
            ]
            title = "Colour fidelity vs purity, base ckpt (sam3 conditions)"
        for label, vals, color, offset in groups:
            bars = ax.bar([x + offset * width for x in xs], vals, width, label=label, color=color)
            ax.bar_label(bars, fmt="%.2f", fontsize=7)
        ax.axhline(0.8, ls="--", lw=1, color="#8b8b8b", label="0.8 pass mark")
        ax.set_xticks(list(xs), [n for n, _ in reports])
        ax.set_ylim(0, 1.18)
        ax.legend(fontsize=6.5, loc="lower right", ncols=2)
        ax.set_xlabel("holdout object")
        ax.set_ylabel("share of sampled surface points")
        ax.set_title(title, fontsize=10)
    else:
        ax.set_title("Colour fidelity (no reports passed)", fontsize=10)

    out = args.out or os.path.join(args.run, "summary.png")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
