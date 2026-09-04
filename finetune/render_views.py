"""Render textured views + GT part-id rasters for objects, without voxelising anything.

Fills in views/<az>/render.png (bpy, Cycles) and views/<az>/ids.npy (nvdiffrast) for every
object that lacks them. This is the input the SAM3 concept bank (Stage B of the v3 plan) and
an enlarged Path A (Stage C) both need; it deliberately skips common.prepare_object so the
o_voxel crashes never enter the picture.

    finetune\run_ft.bat render_views.py --dataset_root E:\data\pv --azimuths 0,135 --chunk 25

Resumable: existing files are skipped. With --chunk the script drives itself in subprocesses
(bpy keeps some state across scenes; a fresh interpreter every few dozen objects keeps memory
flat and isolates the odd bad glb). Objects that still fail alone are recorded in
<dataset_root>/render_views_state.json and skipped on the next run.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import numpy as np

import common
from common import ObjectDir


def view_tag(azimuth: float) -> str:
    return f"az{azimuth:g}"


def missing_views(obj: ObjectDir, azimuths: list[float]) -> tuple[list[float], list[float]]:
    """(azimuths lacking render.png, azimuths lacking ids.npy)."""
    no_render = [az for az in azimuths if not os.path.exists(os.path.join(obj.view_dir(view_tag(az)), "render.png"))]
    no_ids = [az for az in azimuths if not os.path.exists(os.path.join(obj.view_dir(view_tag(az)), "ids.npy"))]
    return no_render, no_ids


def object_aabb(obj: ObjectDir) -> np.ndarray:
    """Same box prepare_object records; fall back to the input glb for unprepared objects."""
    if os.path.exists(obj.ids_meta):
        with open(obj.ids_meta, "r", encoding="utf-8") as f:
            return np.asarray(json.load(f)["aabb"], dtype=np.float64)
    return common.scene_aabb(obj.input_glb)


def render_object(obj: ObjectDir, azimuths: list[float], engine: str, samples: int) -> dict:
    no_render, no_ids = missing_views(obj, azimuths)
    if not no_render and not no_ids:
        return {"rendered": 0, "ids": 0}
    os.makedirs(obj.views_dir, exist_ok=True)

    if no_render:
        from bpy_render import render_from_transforms
        tmp = os.path.join(obj.views_dir, "render.png")
        written = render_from_transforms(obj.input_glb, common.TRANSFORMS, tmp, engine=engine,
                                         azimuths=no_render, samples=samples)
        for az, path in zip(no_render, written):
            vdir = obj.view_dir(view_tag(az))
            os.makedirs(vdir, exist_ok=True)
            os.replace(path, os.path.join(vdir, "render.png"))

    if no_ids:
        parts = common.load_parts(obj)
        aabb = object_aabb(obj)
        for az in no_ids:
            vdir = obj.view_dir(view_tag(az))
            os.makedirs(vdir, exist_ok=True)
            labels = common.render_part_ids(parts, aabb, az)
            np.save(os.path.join(vdir, "ids.npy"), labels)
            common.paint_map(labels, {p: c for p, c in enumerate(common.id_colors(len(parts)))}).save(
                os.path.join(vdir, "ids_preview.png"))
    return {"rendered": len(no_render), "ids": len(no_ids)}


def run_worker(args, azimuths: list[float]) -> int:
    names = common.object_names(args.dataset_root, args.objects, args.objects_file)
    failed = 0
    for name in names:
        obj = ObjectDir(os.path.join(args.dataset_root, name))
        if not os.path.exists(obj.input_glb) or not os.path.isdir(obj.parts_dir):
            print(f"skip {name}: no input.glb / parts")
            continue
        try:
            t0 = time.time()
            r = render_object(obj, azimuths, args.engine, args.samples)
            if r["rendered"] or r["ids"]:
                print(f"{name}: rendered {r['rendered']} view(s), ids {r['ids']} view(s) in {time.time() - t0:.1f}s", flush=True)
        except Exception:
            traceback.print_exc()
            failed += 1
    return 1 if failed else 0


def run_driver(args, azimuths: list[float]) -> None:
    state_path = os.path.join(args.dataset_root, "render_views_state.json")
    state = {"failed": [], "done": 0}
    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            state.update(json.load(f))
    names = common.object_names(args.dataset_root, args.objects, args.objects_file)
    pending = []
    for name in names:
        obj = ObjectDir(os.path.join(args.dataset_root, name))
        if name in state["failed"] or not os.path.exists(obj.input_glb):
            continue
        no_render, no_ids = missing_views(obj, azimuths)
        if no_render or no_ids:
            pending.append(name)
    total = len(pending)
    print(f"{total} objects pending of {len(names)} ({len(state['failed'])} previously failed)", flush=True)

    log_path = args.log or os.path.join(args.dataset_root, "render_views.log")
    base = [sys.executable, os.path.abspath(__file__), "--dataset_root", args.dataset_root,
            "--azimuths", ",".join(f"{a:g}" for a in azimuths), "--engine", args.engine,
            "--samples", str(args.samples), "--worker", "--objects"]

    def run(objs: list[str], log) -> int:
        log.write(f"\n[{time.strftime('%H:%M:%S')}] chunk of {len(objs)}: {' '.join(objs[:3])}{' ...' if len(objs) > 3 else ''}\n")
        log.flush()
        return subprocess.run(base + objs, cwd=common.ROOT, stdout=log, stderr=subprocess.STDOUT).returncode

    t0 = time.time()
    with open(log_path, "a", encoding="utf-8") as log:
        for k in range(0, total, args.chunk):
            chunk = pending[k:k + args.chunk]
            if run(chunk, log) != 0:
                log.write("  chunk failed; retrying objects individually\n")
                for name in chunk:
                    if run([name], log) != 0:
                        state["failed"].append(name)
                        log.write(f"  FAILED {name}\n")
            state["done"] += len(chunk)
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            done = k + len(chunk)
            rate = (time.time() - t0) / done
            print(f"[{(time.time() - t0) / 60:6.1f} min] {done}/{total} objects, {rate:.1f} s/obj, "
                  f"eta {(total - done) * rate / 60:.0f} min, failed {len(state['failed'])}", flush=True)
    print("ALL_DONE", json.dumps({"failed": len(state["failed"])}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--objects", nargs="*", default=None)
    parser.add_argument("--objects_file", default=None)
    parser.add_argument("--azimuths", default="0,135")
    parser.add_argument("--engine", default="CYCLES")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--chunk", type=int, default=25, help="objects per subprocess (0 = run in-process)")
    parser.add_argument("--log", default=None)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    os.chdir(common.ROOT)
    azimuths = [float(a) for a in args.azimuths.split(",") if a.strip()]
    if args.worker or args.chunk <= 0:
        sys.exit(run_worker(args, azimuths))
    run_driver(args, azimuths)


if __name__ == "__main__":
    main()
