"""GeoSAM2 ablation without training: what do SAM2 propagation and geometry buy over SAM3 alone?

Every row uses the SAME SAM3 (+ concept bank) label maps on GeoSAM2's 12 turntable renders, the SAME
lifting (GeoSAM2's depth-tested back-projection + per-point / per-face mode vote), the SAME post-process
(GeoSAM2 complete_labels: drop components < PA of the largest, adjacency fill, nearest-face fill), with
GeoSAM2's 999 ("visible, no part claimed") folded into 0 before the fill so nothing stays open. Only the
2D masks differ:

    geo_p<k>      k SAM3-prompted views (anchor + 12/k spacing), GeoSAM2's SAM2+LoRA+geometry propagator fills
                  the other views. k=1 is GeoSAM2 as shipped minus its SAM2 auto-seg pass, k=2 is our dual.
    sam3_p12      all 12 views' SAM3 maps lifted directly -- no SAM2 at all (the "SAM3 replaces SAM2" row).
    sam3_p<k>     k views' SAM3 maps lifted directly, no propagation (how much do the 12-k missing views cost).
    lift:<dir>    external per-view label maps (e.g. sam3_track.py's SAM3-tracker propagation) -> same lifting.

    .venv_geosam2\Scripts\python finetune\geosam2_ablate.py --renders E:\...\geosam2\renders --out E:\...\geosam2\results
        --objects <id> ... --view match --ref_render_pattern "E:\...\pv\{obj}\views\az0\render.png"
        --modes geo_p1 geo_p2 geo_p4 sam3_p12 sam3_p1 sam3_p2 sam3_p4 lift:sam3track_p1_match lift:sam3track_p2_match

Writes <out>/<obj>/abl_<mode>_<tag>/
    mesh.glb          the mesh in GeoSAM2's normalised frame (face order = the npy files)
    faces_raw.npy     lifted labels before post-process (0 = no label; GeoSAM2's 999 folded into 0)
    faces.npy         after complete_labels (the row that is scored)
    faces_inst.npy    complete_labels' third output (its per-component renumbering never fires in this torch
                      version -- 0-d tensors compare by identity -- so it equals faces.npy; kept for parity)
    labels.json       {"ids": {"1": "roof", ...}, "inst_ids": {...}, "views": [...], "seconds": ...}

Two deviations from GeoSAM2 as shipped: views without a label map are given an all-false alpha so they
cast no 999 votes (otherwise k<12 rows lose most labels), and the mesh is scaled before it is translated
(prepare_mesh) to match geosam2_render.py -- inference.py had the order reversed, which misaligns any
asset whose bbox centre is not at the origin.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

GEOSAM2_ROOT = os.environ.get("GEOSAM2_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "GeoSAM2"))
sys.path.insert(0, GEOSAM2_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.chdir(GEOSAM2_ROOT)

import torch  # noqa: E402

import inference as gi  # noqa: E402  GeoSAM2's inference.py
from utils.inference_utils import complete_labels, lift_2dmask_3d, shrink_mask  # noqa: E402
from geosam2_run import MIN_AREA_PX, match_view  # noqa: E402

NUM_VIEWS = 12


def prompt_views(anchor: int, n: int) -> list[int]:
    step = NUM_VIEWS // n
    return [(anchor + i * step) % NUM_VIEWS for i in range(n)]


def resolve_anchor(rdir: str, view: str, ref_pattern: str | None, obj: str) -> int:
    if view == "best":
        with open(os.path.join(rdir, "sam3", "summary.json"), "r", encoding="utf-8") as f:
            return int(json.load(f)["best_view"])
    if view == "match":
        if not ref_pattern:
            raise SystemExit("--view match needs --ref_render_pattern")
        return match_view(rdir, ref_pattern.format(obj=obj))[0]
    return int(view)


def prepare_mesh(mesh, scale: float, translation: np.ndarray):
    """GeoSAM2's prepare_mesh_and_point_cloud with the renderer's transform order. geosam2_render.py's
    normalize_scene scales the object first and then centres it (world = scale * local + offset), but
    inference.py translates first and scales second, so any asset whose bbox centre is not at the origin
    (dog: offset -0.42, scale 0.94) lands 5-10 % of its size away from the depth maps and fails the 1e-3
    depth test on ~90 % of its surface. Their demo meshes are pre-centred (offset 0), which hides this."""
    from utils.inference_utils import sample_points_on_faces_parallel
    rot = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float)
    mesh.apply_transform(rot)
    mesh.apply_scale(scale)
    mesh.apply_translation(translation)
    pts = sample_points_on_faces_parallel(mesh.vertices[mesh.faces], num_points=gi.SAMPLE_NUM)
    return mesh, torch.from_numpy(pts).float().reshape(-1, 3)


def load_maps(folder: str, views=range(NUM_VIEWS)) -> dict[int, np.ndarray]:
    return {v: np.rint(np.load(os.path.join(folder, f"view_{v:04d}.npy"))).astype(np.int32) for v in views}


def parts_in(label_map: np.ndarray) -> dict[int, np.ndarray]:
    return {int(k): label_map == k for k in np.unique(label_map) if k > 0 and (label_map == k).sum() >= MIN_AREA_PX}


def paint(masks: dict[int, np.ndarray], shape) -> np.ndarray:
    """Label map from overlapping part masks: larger parts first, so small parts stay on top (GeoSAM2's rule)."""
    lab = np.zeros(shape, np.int32)
    for k in sorted(masks, key=lambda k: -int(masks[k].sum())):
        lab[masks[k]] = k
    return lab


def ang(a: int, b: int) -> int:
    d = abs(a - b) % NUM_VIEWS
    return min(d, NUM_VIEWS - d)


# ----------------------------------------------------------------------------- GeoSAM2 propagation
def propagate_geosam2(predictor, state, maps: dict[int, np.ndarray], views: list[int]) -> dict[int, np.ndarray]:
    """One SAM2+geometry pass per prompted view; per frame the passes vote, ties go to the nearest prompt view."""
    shape = next(iter(maps.values())).shape
    per_pass: dict[int, dict[int, np.ndarray]] = {}
    for s in views:
        parts = parts_in(maps[s])
        if not parts:
            continue
        predictor.reset_state(state)
        for k, m in parts.items():
            predictor.add_new_mask(inference_state=state, frame_idx=s, obj_id=k, mask=m)
        segs: dict[int, dict[int, np.ndarray]] = {}
        for f, obj_ids, logits in predictor.propagate_in_video_v2(state, start_frame_idx=s):
            segs[f] = {int(o): logits[i].cpu().numpy() for i, o in enumerate(obj_ids)}
        segs = shrink_mask(segs)
        per_pass[s] = {f: paint({k: (m > 0).squeeze() for k, m in d.items()}, shape) for f, d in segs.items()}
        per_pass[s][s] = paint(parts, shape)  # the prompted view keeps its SAM3 map verbatim
    if not per_pass:
        return {}
    out = {}
    for f in range(NUM_VIEWS):
        stack = np.stack([per_pass[s][f] for s in per_pass])
        order = sorted(range(len(per_pass)), key=lambda i: ang(list(per_pass)[i], f))
        if len(per_pass) == 1:
            out[f] = stack[0]
            continue
        # per-pixel majority over passes (0 = "no part" is a vote too); ties -> nearest prompt view
        vals = np.unique(stack)
        counts = np.stack([(stack == v).sum(0) for v in vals])  # [n_vals, H, W]
        best = counts.max(0)
        lab = np.zeros(shape, np.int32)
        decided = np.zeros(shape, bool)
        for i in order:
            cand = stack[i]
            ok = ~decided & (counts[np.searchsorted(vals, cand), np.arange(shape[0])[:, None], np.arange(shape[1])[None]] == best)
            lab[ok] = cand[ok]
            decided |= ok
        out[f] = lab
    return out


# ----------------------------------------------------------------------------- lifting (shared)
def lift(data, maps: dict[int, np.ndarray], mesh, point_cloud, pa: float):
    ids = sorted({int(k) for m in maps.values() for k in np.unique(m) if k > 0})
    if not ids:
        return None
    shape = next(iter(maps.values())).shape
    # GeoSAM2 keeps its masks as [1, H, W] (SAM2 logits layout); mask_aggregation indexes with that shape
    segs = {f: {k: (maps[f] == k)[None] if f in maps else np.zeros((1, *shape), bool) for k in ids}
            for f in range(NUM_VIEWS)}
    # a view without a label map must not vote: GeoSAM2's per-point mode ignores invisible (0) but counts
    # "visible, no part claimed" (999), so views we have no masks for are made invisible
    alpha = [m if f in maps else np.zeros_like(m) for f, m in enumerate(data["img_masks"])]
    face = lift_2dmask_3d(
        torch.stack([torch.from_numpy(m) for m in alpha], dim=0),
        torch.stack([torch.from_numpy(d).unsqueeze(-1) for d in data["depth_maps"]], dim=0),
        torch.from_numpy(np.stack(data["norm_maps"], axis=0)),
        torch.stack(data["c2ws"], dim=0), data["fovy_deg"], point_cloud, list(range(NUM_VIEWS)), segs, mesh,
        sample_num_per_face=gi.SAMPLE_NUM, view_id="x", uuid="x", prior_keys=set(ids), export_mesh=False)
    raw = face.clone()
    raw[(raw == 999) | (raw < 0)] = 0  # 999 = pixel no part claimed, 0 = point seen by no view
    if (raw != 0).sum() == 0:
        return None
    _, filled, inst = complete_labels(raw.clone(), data["mesh_vanilla"], smooth_type="adjacent", PA=pa)
    return raw.numpy().astype(np.int32), filled.numpy().astype(np.int32), inst.numpy().astype(np.int32), ids


def run_mode(predictor, state, data, mesh, pc, rdir: str, out_dir: str, mode: str, anchor: int, pa: float, prompts):
    t0 = time.time()
    if mode.startswith("lift:"):
        folder = os.path.join(os.path.dirname(out_dir), mode.split(":", 1)[1], "masks")
        info_path = os.path.join(os.path.dirname(folder), "info.json")
        if not os.path.exists(folder):
            return {"error": f"no masks at {folder}"}
        maps, views = load_maps(folder), None
        if os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as f:
                views = json.load(f).get("prompt_views")
    elif mode.startswith("sam3_p"):
        n = int(mode[6:])
        views = prompt_views(anchor, n)
        maps = load_maps(os.path.join(rdir, "sam3"), views)
    elif mode.startswith("geo_p"):
        n = int(mode[5:])
        views = prompt_views(anchor, n)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            maps = propagate_geosam2(predictor, state, load_maps(os.path.join(rdir, "sam3"), views), views)
        if not maps:
            return {"error": "no SAM3 masks in any prompted view"}
    else:
        raise ValueError(mode)
    res = lift(data, maps, mesh, pc, pa)
    if res is None:
        return {"error": "nothing lifted"}
    raw, filled, inst, ids = res
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "faces_raw.npy"), raw.astype(np.float32))
    np.save(os.path.join(out_dir, "faces.npy"), filled.astype(np.float32))
    np.save(os.path.join(out_dir, "faces_inst.npy"), inst.astype(np.float32))
    mesh.export(os.path.join(out_dir, "mesh.glb"))
    names = {str(k): prompts[k - 1] for k in ids if 0 < k <= len(prompts)}
    inst_ids = {}
    for u in np.unique(inst):
        if u <= 0:
            continue
        src = filled[inst == u]
        base = int(np.bincount(src[src > 0]).argmax()) if (src > 0).any() else 0
        if str(base) in names:
            inst_ids[str(int(u))] = names[str(base)]
    info = {"mode": mode, "anchor": anchor, "views": views, "ids": names, "inst_ids": inst_ids, "pa": pa,
            "seconds": round(time.time() - t0, 1), "n_faces": int(len(filled)),
            "unlabelled_faces_raw": int((raw == 0).sum()), "n_segments_inst": int(len(inst_ids)),
            "labelled_share_2d": [round(float((maps[v] > 0).mean()), 4) if v in maps else None for v in range(NUM_VIEWS)]}
    with open(os.path.join(out_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--objects", nargs="*", default=None)
    ap.add_argument("--objects_file", default=None)
    ap.add_argument("--view", default="match")
    ap.add_argument("--ref_render_pattern", default=None)
    ap.add_argument("--modes", nargs="+", default=["geo_p1", "geo_p2", "geo_p4", "sam3_p12", "sam3_p1", "sam3_p2", "sam3_p4"])
    ap.add_argument("--pa", type=float, default=0.02)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    objects = list(args.objects or [])
    if args.objects_file:
        with open(args.objects_file, "r", encoding="utf-8") as f:
            objects += [l.strip() for l in f if l.strip()]
    tag = args.view if args.view in ("best", "match") else f"view{int(args.view):02d}"
    need_geo = any(m.startswith("geo_") for m in args.modes)
    predictor = None
    if need_geo:
        from sam2.build_sam import build_sam2_video_predictor_geosam2
        device = gi.init_env()
        predictor = build_sam2_video_predictor_geosam2("configs/geosam2.yaml", "ckpt/geosam2.pt", device=device)

    for obj in objects:
        rdir = os.path.join(args.renders, obj)
        todo = [m for m in args.modes if args.force or not os.path.exists(
            os.path.join(args.out, obj, f"abl_{m.replace(':', '_')}_{tag}", "labels.json"))]
        if not todo:
            print(f"[skip] {obj}", flush=True)
            continue
        with open(os.path.join(rdir, "sam3", "summary.json"), "r", encoding="utf-8") as f:
            prompts = json.load(f)["prompts"]
        anchor = resolve_anchor(rdir, args.view, args.ref_render_pattern, obj)
        data = gi.read_data(rdir.replace("\\", "/"))
        mesh, pc = prepare_mesh(data["mesh"], float(data["scaling_factor"]), data["translation"].numpy())
        state = None
        if predictor is not None and any(m.startswith("geo_") for m in todo):
            state = predictor.init_state(video_path=rdir.replace("\\", "/"), video_id_list=list(range(NUM_VIEWS)))
        for m in todo:
            out_dir = os.path.join(args.out, obj, f"abl_{m.replace(':', '_')}_{tag}")
            try:
                info = run_mode(predictor, state, data, mesh, pc, rdir, out_dir, m, anchor, args.pa, prompts)
            except Exception as e:
                info = {"error": f"{type(e).__name__}: {e}"}
            if "error" in info:
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, "labels.json"), "w", encoding="utf-8") as f:
                    json.dump({"mode": m, **info}, f, indent=2)
                print(f"[FAIL] {obj} {m}: {info['error']}", flush=True)
            else:
                print(f"[ok] {obj} {m} anchor={anchor} views={info['views']} ids={len(info['ids'])} "
                      f"segs={info['n_segments_inst']} unlab_raw={info['unlabelled_faces_raw']}/{info['n_faces']} "
                      f"{info['seconds']}s", flush=True)
        if state is not None:
            predictor.reset_state(state)
            del state
            torch.cuda.empty_cache()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
