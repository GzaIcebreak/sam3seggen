"""Upload an already-finished train.py run to Weights & Biases from its log files.

Use this when a run completed without --wandb (or you logged in after the fact).
It posts the same metrics train.py would have logged, at the original step numbers,
so the curves line up with a later --wandb training run.

    finetune\run_ft.bat wandb_backfill.py --run finetune\runs\pv_v1 ^
        --check0 finetune\runs\pv_check0\check_loss.json
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import wandb

from run_stats import read_checks, read_log, read_run_args, step_times


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--check0", default=None)
    parser.add_argument("--project", default="segvigen-finetune")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--id", default=None, help="Defaults to the out_dir basename (same as train.py)")
    parser.add_argument("--mode", default="online", choices=["online", "offline"])
    args = parser.parse_args()

    rows = read_log(args.run)
    if not rows:
        raise SystemExit(f"no log.jsonl rows in {args.run}")
    run_args = read_run_args(args.run)
    checks = read_checks(args.run, args.check0)
    steps, secs = step_times(rows)
    rate = {s: t for s, t in zip(steps, secs)}
    kinds = {}
    for rec in rows:
        for k, v in rec.get("per_kind", {}).items():
            kinds[k] = kinds.get(k, 0) + 1

    run = wandb.init(
        project=args.project, entity=args.entity, mode=args.mode,
        id=args.id or os.path.basename(os.path.normpath(args.run)),
        name=args.name or os.path.basename(os.path.normpath(args.run)),
        resume="allow",
        config={**run_args, "n_log_rows": len(rows), "variants_per_kind_logged": kinds,
                "backfilled": True, "source_run": os.path.abspath(args.run)},
    )
    wandb.define_metric("train/loss", summary="min")
    wandb.define_metric("holdout/*", summary="min")
    for rec in rows:
        payload = {"train/loss": rec["loss"], "train/ema": rec["ema"], "train/lr": rec["lr"],
                   "train/vram_gib": rec.get("vram_gib"),
                   "train/s_per_step": rate.get(rec["step"]),
                   **{f"train/loss_{k}": v for k, v in rec.get("per_kind", {}).items()}}
        run.log({k: v for k, v in payload.items() if v is not None}, step=rec["step"])
    for step, summary in checks:
        run.log({f"holdout/{k}": s["mean"] for k, s in summary.items()}, step=step)
    print(f"wandb: {run.url or args.mode}  ({len(rows)} train points, {len(checks)} holdout checks)")
    run.finish()


if __name__ == "__main__":
    main()
