"""P3-SAM (Hunyuan3D-Part) native 3D part segmentation of one mesh -> a part id per face.

Runs in the X-Part venv (it has Sonata, fpsample and numba; SegviGen's does not). The
pipeline calls it through p3sam_units.py. Writes into --out_dir:

    mesh.glb       the mesh P3-SAM labelled: its clean_mesh merges vertices and drops
                   duplicate faces, so the faces are not the source's, but the frame is
    face_ids.npy   int part id per face of mesh.glb (-1 = no part)
    aabb.npy       per-part boxes as P3-SAM reports them
    stats.json     parts, area shares, timing

The demo's --prompt_bs is parsed but never handed on, so its default of 32 prompts per
batch builds a [N=100k, K=32, 1033] float tensor and wants 23 GB; 8 fits in a few GB.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import trimesh

DEFAULT_P3SAM_ROOT = os.environ.get("SEGVIGEN_P3SAM_ROOT", "/root/autodl-tmp/Hunyuan3D-Part/P3-SAM")
DEFAULT_P3SAM_WEIGHTS = os.environ.get(
    "SEGVIGEN_P3SAM_WEIGHTS", "/root/autodl-tmp/Hunyuan3D-Part/weights/p3sam/p3sam.safetensors")


def decimate_with_ids(mesh, face_ids, max_faces):
    """(decimated mesh, its face ids): cumesh on the GPU (as holopart_complete does),
    pymeshlab as the fallback; each new face takes the id of the nearest old centroid."""
    from scipy.spatial import cKDTree

    small = None
    try:
        from holopart_complete import cumesh_simplify

        small = cumesh_simplify(mesh, max_faces)
    except Exception as exc:  # no cumesh / GPU trouble: pymeshlab is in this venv too
        print(f"[p3sam] cumesh simplify unavailable ({type(exc).__name__}: {str(exc)[:80]}); pymeshlab")
        import pymeshlab

        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(np.asarray(mesh.vertices, dtype=np.float64),
                                   np.asarray(mesh.faces, dtype=np.int32)))
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(max_faces),
                                                    preservetopology=True)
        m = ms.current_mesh()
        small = trimesh.Trimesh(m.vertex_matrix(), m.face_matrix(), process=False)
    small = trimesh.Trimesh(np.asarray(small.vertices), np.asarray(small.faces), process=False)
    nearest = cKDTree(np.asarray(mesh.triangles_center)).query(np.asarray(small.triangles_center))[1]
    return small, np.asarray(face_ids)[nearest]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--glb", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--p3sam_root", default=DEFAULT_P3SAM_ROOT)
    parser.add_argument("--weights", default=DEFAULT_P3SAM_WEIGHTS)
    parser.add_argument("--point_num", type=int, default=100000)
    parser.add_argument("--prompt_num", type=int, default=400)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--prompt_bs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_clean", action="store_true",
                        help="skip P3-SAM's clean_mesh (merge vertices, drop duplicate faces)")
    parser.add_argument("--max_faces", type=int, default=150000,
                        help="decimate the labelled mesh to at most this many faces before "
                             "handing it on (the vote and the mask refinement run on it; the "
                             "source cut carries labels over by nearest face anyway)")
    args = parser.parse_args()

    root = os.path.abspath(args.p3sam_root)
    demo = os.path.join(root, "demo")
    for path in (demo, root):
        if path not in sys.path:
            sys.path.insert(0, path)
    os.chdir(demo)   # the demo resolves its siblings relative to the cwd
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    import torch  # noqa: F401  (the venv check)
    from auto_mask import AutoMask, Timer, set_seed

    Timer.STATE = 0
    mesh = trimesh.load(os.path.abspath(args.glb), force="mesh")
    print(f"[p3sam] {len(mesh.faces)} faces; loading {os.path.basename(args.weights)} ...")
    started = time.time()
    auto_mask = AutoMask(os.path.abspath(args.weights))
    loaded = time.time()
    set_seed(args.seed)
    aabb, face_ids, labelled = auto_mask.predict_aabb(
        mesh, save_path=out_dir, point_num=args.point_num, prompt_num=args.prompt_num,
        threshold=args.threshold, post_process=0, save_mid_res=False, show_info=False,
        clean_mesh_flag=not args.no_clean, seed=args.seed, is_parallel=False,
        prompt_bs=args.prompt_bs)
    face_ids = np.asarray(face_ids).astype(np.int64)
    if len(face_ids) != len(labelled.faces):
        raise SystemExit(f"P3-SAM returned {len(face_ids)} ids for {len(labelled.faces)} faces")
    bare = trimesh.Trimesh(np.asarray(labelled.vertices), np.asarray(labelled.faces), process=False)
    full_faces = len(bare.faces)
    if full_faces > args.max_faces:
        bare, face_ids = decimate_with_ids(bare, face_ids, args.max_faces)
        print(f"[p3sam] reference decimated {full_faces} -> {len(bare.faces)} faces for the vote")
    bare.export(os.path.join(out_dir, "mesh.glb"))
    np.save(os.path.join(out_dir, "face_ids.npy"), face_ids)
    np.save(os.path.join(out_dir, "aabb.npy"), np.asarray(aabb, dtype=np.float64).reshape(-1, 2, 3))
    areas = np.asarray(bare.area_faces)
    total = max(float(areas.sum()), 1e-12)
    ids = [int(i) for i in np.unique(face_ids) if i >= 0]
    shares = sorted((float(areas[face_ids == i].sum() / total) for i in ids), reverse=True)
    stats = {
        "faces": int(len(bare.faces)), "labelled_faces": int(full_faces), "source_faces": int(len(mesh.faces)),
        "parts": len(ids), "shares": shares,
        "largest_share": shares[0] if shares else 0.0,
        "unlabelled_share": float(areas[face_ids < 0].sum() / total),
        "point_num": args.point_num, "prompt_num": args.prompt_num, "prompt_bs": args.prompt_bs,
        "threshold": args.threshold, "seed": args.seed,
        "load_seconds": round(loaded - started, 1), "seconds": round(time.time() - loaded, 1),
    }
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    print(f"[p3sam] {stats['parts']} parts in {stats['seconds']} s (model load "
          f"{stats['load_seconds']} s); largest {stats['largest_share']:.1%}, "
          f"unlabelled {stats['unlabelled_share']:.1%} -> {out_dir}")


if __name__ == "__main__":
    main()
