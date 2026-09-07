"""Original P3-SAM auto-mask (Tencent Hunyuan3D-Part) on one or more GLBs, unchanged model and defaults.

Runs inside P3-SAM's own conda env; use finetune\\run_p3sam.bat, which activates `p3sam` and calls this file:

    finetune\\run_p3sam.bat --out datasets\\ext_bench\\p3sam --glb datasets\\ext_bench\\mesh\\dog.glb ...

Per input <stem> it writes --out/<stem>/:
    mesh.glb    the mesh P3-SAM actually labelled (its clean_mesh merges vertices, so this is exported with
                process=False to keep the face order the labels refer to); same world frame as the input
    faces.npy   one int label per face of mesh.glb, -1 = P3-SAM left it unlabelled
    info.json   seconds, face / part counts, settings

P3-SAM is class-agnostic: 400 FPS point prompts -> masks -> NMS-style merge -> face voting, so the labels have
no names. Settings follow demo/auto_mask.py (100k points, 400 prompts, threshold 0.95, clean_mesh); the
post-processing that merges tiny parts is on by default here (the README's auto_mask.py vs
auto_mask_no_postprocess.py split), pass --post_process 0 for the raw merge output.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import trimesh

P3SAM = os.environ.get("P3SAM_ROOT", r"E:\p3part\P3-SAM")
DEMO = os.path.join(P3SAM, "demo")
CKPT_DEFAULT = os.path.join(os.path.expanduser("~"), ".cache", "p3sam", "weights", "p3sam", "p3sam.safetensors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=CKPT_DEFAULT)
    ap.add_argument("--prompt_num", type=int, default=400)
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--post_process", type=int, default=1)
    ap.add_argument("--prompt_bs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # auto_mask.py does sys.path.append('..') and `from model import ...`, so it must be imported from demo/
    sys.path.insert(0, P3SAM)
    sys.path.insert(0, DEMO)
    os.chdir(DEMO)
    from auto_mask import AutoMask, Timer, set_seed
    Timer.STATE = False

    auto = AutoMask(args.ckpt)
    for glb in args.glb:
        glb = os.path.abspath(glb)
        stem = os.path.splitext(os.path.basename(glb))[0]
        od = os.path.join(os.path.abspath(args.out), stem)
        os.makedirs(od, exist_ok=True)
        if os.path.exists(os.path.join(od, "faces.npy")):
            print(f"[p3sam] {stem}: exists, skip", flush=True)
            continue
        mesh = trimesh.load(glb, force="mesh")
        print(f"[p3sam] {stem}: {len(mesh.faces)} faces", flush=True)
        set_seed(args.seed)
        t0 = time.time()
        _, face_ids, out_mesh = auto.predict_aabb(
            mesh, prompt_num=args.prompt_num, threshold=args.threshold, post_process=bool(args.post_process),
            save_mid_res=False, show_info=False, clean_mesh_flag=True, seed=args.seed, is_parallel=False,
            prompt_bs=args.prompt_bs)
        seconds = time.time() - t0
        face_ids = np.asarray(face_ids, dtype=np.int64)
        face_ids[face_ids < 0] = -1
        if len(face_ids) != len(out_mesh.faces):
            raise RuntimeError(f"{stem}: {len(face_ids)} labels for {len(out_mesh.faces)} faces")
        trimesh.Trimesh(out_mesh.vertices, out_mesh.faces, process=False).export(os.path.join(od, "mesh.glb"))
        np.save(os.path.join(od, "faces.npy"), face_ids)
        parts = [int(u) for u in np.unique(face_ids) if u >= 0]
        info = {
            "source": glb, "seconds": seconds, "n_faces_in": int(len(mesh.faces)), "n_faces": int(len(out_mesh.faces)),
            "n_parts": len(parts), "unlabelled_faces": int((face_ids < 0).sum()),
            "settings": {"point_num": 100000, "prompt_num": args.prompt_num, "threshold": args.threshold,
                         "post_process": bool(args.post_process), "clean_mesh": True, "seed": args.seed,
                         "ckpt": args.ckpt},
        }
        with open(os.path.join(od, "info.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=1)
        print(f"[p3sam] {stem}: {len(parts)} parts, {info['unlabelled_faces']} unlabelled faces, {seconds:.0f}s", flush=True)


if __name__ == "__main__":
    main()
