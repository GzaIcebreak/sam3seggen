"""Reading the artifacts a finetune/train.py run leaves behind.

Shared by plot_run.py (PNG) and report_html.py (HTML) so both report the same numbers.
"""
from __future__ import annotations

import glob
import json
import os
import re

KINDS = ["clean", "corrupt", "sam3"]


def _attempts(rows: list[dict]) -> list[list[dict]]:
    """Split the appended log into one list per train.py attempt.

    A new attempt is where the step number stops increasing: a fresh start resets to 1,
    a --resume_lora start picks up at the step its checkpoint was saved at.
    """
    out, cur = [], []
    for row in rows:
        if cur and row["step"] <= cur[-1]["step"]:
            out.append(cur)
            cur = []
        cur.append(row)
    if cur:
        out.append(cur)
    return out


def read_log(run: str) -> list[dict]:
    """The training history, stitched back together across train_loop restarts.

    Every attempt appends to the same log.jsonl, so a run that died and restarted from
    step 1 leaves several overlapping step ranges in one file. Taking the last attempt
    reports an aborted retry instead of the run that actually trained, so use the attempt
    that got furthest, plus whatever earlier rows it resumed on top of.
    """
    with open(os.path.join(run, "log.jsonl"), "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    attempts = _attempts(rows)
    if not attempts:
        return []
    best = max(attempts, key=lambda a: a[-1]["step"])
    merged = {row["step"]: row for row in best}
    for attempt in attempts:
        if attempt is best:
            break
        for row in attempt:
            if row["step"] < best[0]["step"]:
                merged[row["step"]] = row
    return [merged[step] for step in sorted(merged)]


def read_checks(run: str, check0: str | None = None) -> list[tuple[int, dict]]:
    """Periodic holdout checks, with an optional step-0 baseline prepended."""
    out = []
    if check0 and os.path.exists(check0):
        with open(check0, "r", encoding="utf-8") as f:
            out.append((0, json.load(f)["summary"]))
    for path in glob.glob(os.path.join(run, "check_step*.json")):
        step = int(re.search(r"check_step(\d+)", os.path.basename(path)).group(1))
        with open(path, "r", encoding="utf-8") as f:
            out.append((step, json.load(f)["summary"]))
    return sorted(out)


def step_times(rows: list[dict]) -> tuple[list[int], list[float]]:
    """Seconds per step between consecutive log points, so contention stays visible."""
    steps, secs = [], []
    for prev, cur in zip(rows, rows[1:]):
        ds, dt = cur["step"] - prev["step"], cur["elapsed_s"] - prev["elapsed_s"]
        if ds > 0 and dt > 0:
            steps.append(cur["step"])
            secs.append(dt / ds)
    return steps, secs


def read_run_args(run: str) -> dict:
    path = os.path.join(run, "args.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_fidelity(paths: list[str] | None) -> list[tuple[str, dict]]:
    """(object name, summary) for each eval_fidelity.py report that exists."""
    out = []
    for path in paths or []:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            summary = json.load(f)["summary"]
        out.append((os.path.normpath(path).split(os.sep)[-4][:8], summary))
    return out


def checkpoints(run: str) -> list[tuple[int, str]]:
    out = []
    for path in glob.glob(os.path.join(run, "lora_step*.pt")):
        out.append((int(re.search(r"lora_step(\d+)", os.path.basename(path)).group(1)),
                    os.path.basename(path)))
    return sorted(out)
