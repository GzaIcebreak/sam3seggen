"""Drive GeoSAM2 inference on SAM3 label maps (one prompt view per run) and record the id -> name map.

    E:\...\.venv_geosam2\Scripts\python finetune\geosam2_run.py --renders E:\...\geosam2\renders
        --out E:\...\geosam2\results --objects_file E:\...\pv_hard.txt [--dataset_root E:\...\pv]
        --view best|match|<int> [--pa 0.02]

Per object and view choice this writes <out>/<obj>/<tag>/
    segmentation_postprocessed_autoViewNN_fromPromptVV_pa0.02.glb / .npy   (GeoSAM2's own export:
        mesh in its normalised frame, one label per face; 0 / 999 = unlabelled)
    ..._filled.npy   same with unlabelled faces filled from the nearest labelled face (fill_unlabelled)
    labels.json   {"prompt_view": V, "ids": {"1": "helmet", ...}, "n_prompted": k, "cmd": [...]}
GeoSAM2 numbers the prompt segments 1..k in ascending order of the label-map value (segments under
64 px are dropped, inference.MASK_MIN_AREA_PX); auto-segmented regions on the opposite view get
ids above k and carry no name.

View choices: "best" = the view where SAM3 found the most prompts (geosam2_masks summary.json);
"match" = the GeoSAM2 view whose silhouette best matches SegviGen's az0 render (same input view for
both methods); an integer = that view.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image

GEOSAM2_ROOT = os.environ.get("GEOSAM2_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "GeoSAM2"))
MIN_AREA_PX = 64  # inference.MASK_MIN_AREA_PX


def silhouette(fg: np.ndarray, size: int = 128) -> np.ndarray:
    ys, xs = np.nonzero(fg)
    if len(ys) == 0:
        return np.zeros((size, size), dtype=bool)
    crop = fg[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    s = size / max(h, w)
    im = Image.fromarray(crop.astype(np.uint8) * 255).resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(im, ((size - im.size[0]) // 2, (size - im.size[1]) // 2))
    return np.asarray(canvas) > 127


def match_view(render_dir: str, segvigen_render: str) -> tuple[int, float]:
    ref = np.asarray(Image.open(segvigen_render).convert("RGB")).max(axis=2) > 8
    ref_s = silhouette(ref)
    best = (0, -1.0)
    for v in range(12):
        fg = np.asarray(Image.open(os.path.join(render_dir, f"color_{v:04d}.webp")).convert("RGBA"))[..., 3] > 0
        s = silhouette(fg)
        iou = float((s & ref_s).sum() / max(1, (s | ref_s).sum()))
        if iou > best[1]:
            best = (v, iou)
    return best


def fill_unlabelled(mesh, labels: np.ndarray, unlabelled: set[int]) -> np.ndarray:
    """Every unlabelled face takes the label of the nearest labelled face centroid. Spatial rather than
    adjacency-based because GLB meshes have split vertices (UV/normal seams), which leaves face_adjacency
    too sparse to reach interior or seam-separated faces. SegviGen never leaves a voxel unpainted, so the
    filled labels are the like-for-like comparison; the raw ones show how much GeoSAM2 left open."""
    from scipy.spatial import cKDTree
    lab = labels.copy()
    for u in unlabelled:
        lab[lab == u] = 0
    have = np.nonzero(lab != 0)[0]
    need = np.nonzero(lab == 0)[0]
    if len(have) == 0 or len(need) == 0:
        return lab
    centroids = mesh.triangles_center
    _, idx = cKDTree(centroids[have]).query(centroids[need])
    lab[need] = lab[have[idx]]
    return lab


def write_filled(glb: str, npy: str) -> str:
    import trimesh
    loaded = trimesh.load(glb, force="scene")
    meshes = [m for m in loaded.dump() if isinstance(m, trimesh.Trimesh)]
    mesh = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
    lab = np.rint(np.load(npy)).astype(np.int64)
    filled = fill_unlabelled(mesh, lab, {0, 999})
    out = npy.replace(".npy", "_filled.npy")
    np.save(out, filled.astype(np.float32))
    return out


def id_names(label_map: np.ndarray, prompts: list[str]) -> dict[str, str]:
    vals, counts = np.unique(label_map, return_counts=True)
    keep = sorted(int(v) for v, c in zip(vals, counts) if v > 0 and c >= MIN_AREA_PX)
    return {str(i + 1): prompts[k - 1] for i, k in enumerate(keep)}


def run_one(render_dir: str, out_dir: str, view: int, pa: float, python: str, log) -> dict:
    sam3_dir = os.path.join(render_dir, "sam3")
    with open(os.path.join(sam3_dir, "summary.json"), "r", encoding="utf-8") as f:
        summary = json.load(f)
    mask_path = os.path.join(sam3_dir, f"view_{view:04d}.npy")
    lab = np.load(mask_path)
    ids = id_names(lab, summary["prompts"])
    os.makedirs(out_dir, exist_ok=True)
    cmd = [python, "inference.py", "--data-root", render_dir.replace("\\", "/"), "--mask-path",
           mask_path.replace("\\", "/"), "--mask-view", str(view), "--opposite-auto-segmentation",
           "--enable-postprocess", "--postprocess-pa", str(pa), "--output-dir", out_dir.replace("\\", "/")]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", OPENCV_IO_ENABLE_OPENEXR="1", PYTHONUTF8="1")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=GEOSAM2_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    outputs = sorted(f for f in os.listdir(out_dir) if f.endswith(".npy") and f.startswith("segmentation")
                     and not f.endswith("_filled.npy"))
    for f in outputs:
        write_filled(os.path.join(out_dir, f.replace(".npy", ".glb")), os.path.join(out_dir, f))
    info = {"prompt_view": view, "ids": ids, "n_prompted": len(ids), "pa": pa, "cmd": cmd,
            "returncode": r.returncode, "seconds": round(time.time() - t0, 1), "outputs": outputs}
    with open(os.path.join(out_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset_root", default=None, help="Needed for --view match (SegviGen az0 render)")
    ap.add_argument("--objects_file", default=None)
    ap.add_argument("--objects", nargs="*", default=None)
    ap.add_argument("--view", default="best", help="best | match | <view id>")
    ap.add_argument("--pa", type=float, default=0.02)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    objects = list(args.objects or [])
    if args.objects_file:
        with open(args.objects_file, "r", encoding="utf-8") as f:
            objects += [l.strip() for l in f if l.strip()]
    tag = args.view if args.view in ("best", "match") else f"view{int(args.view):02d}"
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, f"geosam2_{tag}.log")
    with open(log_path, "a", encoding="utf-8") as log:
        for obj in objects:
            rdir = os.path.join(args.renders, obj)
            out_dir = os.path.join(args.out, obj, tag)
            if os.path.exists(os.path.join(out_dir, "labels.json")) and not args.force:
                with open(os.path.join(out_dir, "labels.json"), "r", encoding="utf-8") as f:
                    if f and json.load(f).get("outputs"):
                        print(f"[skip] {obj} {tag}", flush=True)
                        continue
            if args.view == "best":
                with open(os.path.join(rdir, "sam3", "summary.json"), "r", encoding="utf-8") as f:
                    view = int(json.load(f)["best_view"])
            elif args.view == "match":
                ref = os.path.join(args.dataset_root, obj, "views", "az0", "render.png")
                view, iou = match_view(rdir, ref)
                print(f"[{obj}] az0 matches view {view} (silhouette IoU {iou:.2f})", flush=True)
            else:
                view = int(args.view)
            log.write(f"\n===== {obj} {tag} view {view}\n")
            log.flush()
            info = run_one(rdir, out_dir, view, args.pa, sys.executable, log)
            status = "ok" if info["returncode"] == 0 and info["outputs"] else "FAIL"
            print(f"[{status}] {obj} {tag} view={view} prompted={info['n_prompted']} {info['seconds']}s "
                  f"{info['outputs']}", flush=True)


if __name__ == "__main__":
    main()
