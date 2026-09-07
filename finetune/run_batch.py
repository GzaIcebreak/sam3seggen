"""Crash-tolerant driver for make_samples_a / make_samples_b over many objects.

o_voxel's native extension occasionally dies with an access violation after a few dozen
voxelisations, taking the whole batch with it. This driver runs the sample scripts in
chunks of --chunk objects per subprocess; when a chunk fails it retries its objects one
at a time, records the ones that still fail, and moves on. Jobs "a" and "b" alternate
chunk by chunk so they never share the GPU concurrently. Already finished work is skipped
by the scripts themselves, so re-running the driver resumes where it stopped.

    finetune\run_ft.bat run_batch.py --dataset_root E:\data\pv --jobs b a ^
        --objects_b E:\data\pv_list_b.txt --objects_a E:\data\pv_list_a.txt --chunk 8 ^
        --args_b "--azimuths 0,135 --n_corrupt 3" --args_a "--azimuths 0,135"
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common

SCRIPTS = {"a": "make_samples_a.py", "b": "make_samples_b.py"}


def expected_done(obj: common.ObjectDir, job: str, azimuths: list[float], n_corrupt: int) -> bool:
    if not obj.is_prepared():
        return False
    for az in azimuths:
        tag = f"az{az:g}"
        names = [f"sam3_{tag}"] if job == "a" else [f"clean_{tag}"] + [f"corrupt_{tag}_{k}" for k in range(n_corrupt)]
        for n in names:
            if not os.path.exists(os.path.join(obj.variant_dir(n), "meta.json")):
                # path A legitimately skips views where SAM3 bound nothing; treat a written view dir with masks as done
                if job == "a" and os.path.exists(os.path.join(obj.view_dir(tag), "sam3_masks.npz")):
                    continue
                return False
    return True


def run_chunk(job: str, dataset_root: str, objects: list[str], extra: list[str], log) -> int:
    cmd = [sys.executable, os.path.join(common.ROOT, "finetune", SCRIPTS[job]), "--dataset_root", dataset_root,
           "--objects", *objects, *extra]
    log.write(f"\n[{time.strftime('%H:%M:%S')}] {job} chunk of {len(objects)}: {' '.join(objects[:3])}{' ...' if len(objects) > 3 else ''}\n")
    log.flush()
    proc = subprocess.run(cmd, cwd=common.ROOT, stdout=log, stderr=subprocess.STDOUT)
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--jobs", nargs="+", choices=["a", "b"], default=["b", "a"])
    parser.add_argument("--objects_a", default=None)
    parser.add_argument("--objects_b", default=None)
    parser.add_argument("--args_a", default="--azimuths 0,135")
    parser.add_argument("--args_b", default="--azimuths 0,135 --n_corrupt 3")
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--log", default=None)
    parser.add_argument("--state", default=None, help="JSON file tracking failed objects (default <root>/run_batch_state.json)")
    args = parser.parse_args()

    os.chdir(common.ROOT)
    state_path = args.state or os.path.join(args.dataset_root, "run_batch_state.json")
    state = {"failed": {"a": [], "b": []}, "done": {"a": 0, "b": 0}}
    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            state.update(json.load(f))
    log = open(args.log or os.path.join(args.dataset_root, "run_batch.log"), "a", encoding="utf-8")

    def parse(extra: str):
        # posix=False keeps Windows path backslashes (shlex would eat them as escapes)
        toks = [t.strip('"') for t in shlex.split(extra, posix=False)]
        az = [float(a) for a in toks[toks.index("--azimuths") + 1].split(",")] if "--azimuths" in toks else [0.0]
        nc = int(toks[toks.index("--n_corrupt") + 1]) if "--n_corrupt" in toks else 3
        return toks, az, nc

    queues = {}
    for job in args.jobs:
        names = common.object_names(args.dataset_root, None, getattr(args, f"objects_{job}"))
        toks, az, nc = parse(getattr(args, f"args_{job}"))
        failed = set(state["failed"][job])
        pending = [n for n in names if n not in failed
                   and not expected_done(common.ObjectDir(os.path.join(args.dataset_root, n)), job, az, nc)]
        queues[job] = {"pending": pending, "extra": toks, "total": len(names)}
        print(f"job {job}: {len(pending)} pending of {len(names)} ({len(failed)} previously failed)")

    def save_state():
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    t0 = time.time()
    while any(q["pending"] for q in queues.values()):
        for job in args.jobs:
            q = queues[job]
            if not q["pending"]:
                continue
            chunk, q["pending"] = q["pending"][:args.chunk], q["pending"][args.chunk:]
            rc = run_chunk(job, args.dataset_root, chunk, q["extra"], log)
            if rc != 0:
                log.write(f"  chunk failed rc={rc}; retrying objects individually\n")
                for name in chunk:
                    rc1 = run_chunk(job, args.dataset_root, [name], q["extra"], log)
                    if rc1 != 0:
                        state["failed"][job].append(name)
                        log.write(f"  FAILED {job} {name} rc={rc1}\n")
            state["done"][job] += len(chunk)
            save_state()
            done = q["total"] - len(q["pending"])
            print(f"[{(time.time() - t0) / 60:6.1f} min] {job}: {done}/{q['total']} "
                  f"(failed {len(state['failed'][job])})", flush=True)
    print("ALL_DONE", json.dumps({k: len(v) for k, v in state["failed"].items()}))


if __name__ == "__main__":
    main()
