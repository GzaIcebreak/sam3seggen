"""Side-by-side benchmark on the user's own assets (datasets/ext_parts, the 3D拆件.zip set):

    SegviGen native      full_seg.ckpt on the front render (no SAM3, SegviGen's own part colouring)
    SegviGen base+SAM3   full_seg_w_2d_map.ckpt on the SAM3 (+concept bank v3) 2D map of the same view
    ours (v6+SAM3)       full_seg_v6.ckpt (Path A trajectory LoRA merged) on the same map
    GeoSAM2 single       SAM3 label map of the matching view as mask prompt + auto-seg on the opposite view
    GeoSAM2 dual         SAM3 label maps of the matching view and its opposite, no auto-seg, unlabelled filled
    P3-SAM               original Hunyuan3D-Part auto-mask on the same mesh (class-agnostic, no SAM3, no names)

No GT exists for these assets, so `score` reports label-free statistics on a shared surface sample of the
original mesh (segments, fragments per segment, unlabelled share, boundary density) plus pairwise
agreement between methods (class-agnostic matched mIoU, per-name IoU where both carry names).

Stages (all resumable, outputs under datasets/ext_bench/<key>/):
    finetune\run_ft.bat ext_bench.py front      8-azimuth turntable -> front view (front.json, render.png, render_back.png)
    finetune\run_ft.bat ext_bench.py mesh       ASCII-named copy, decimated to --max_faces when larger (GeoSAM2 input)
    finetune\run_ft.bat ext_bench.py sam3       SAM3 + concept bank map of the front view (map.png, legend.json)
    finetune\run_ft.bat ext_bench.py segvigen   the three SegviGen runs + upright re-export in the asset frame
    finetune\run_ft.bat ext_bench.py geo_prep   12-view render + SAM3 label maps only (light; can overlap the SegviGen stage)
    finetune\run_ft.bat ext_bench.py geosam2    geo_prep, then single/dual x match/best (~10 GB GPU), parts GLBs
    finetune\run_ft.bat ext_bench.py p3sam      original P3-SAM auto-mask in its conda env (run_p3sam.bat), parts GLBs
    finetune\run_ft.bat ext_bench.py score      metrics -> ext_bench/scores.json
    finetune\run_ft.bat ext_bench.py montage    front/back renders of every output -> compare_front.png, compare_back.png
    finetune\run_ft.bat ext_bench.py report     scores.json + montages -> ext_bench/REPORT.md
    finetune\run_ft.bat ext_bench.py all        everything, in order
Add --assets key1 key2 to restrict.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DS = r"E:\AI_New\ModelGen\datasets"
SRC = os.path.join(DS, "ext_parts")
OUT = os.path.join(DS, "ext_bench")
BANK = os.path.join(DS, "concept_bank_v3", "bank.pt")
PY = r"E:\AI_New\ModelGen\.venv\Scripts\python.exe"
PY_SAM3 = r"E:\AI_New\ModelGen\.venv_holo\Scripts\python.exe"
PY_GEO = r"E:\AI_New\ModelGen\.venv_geosam2\Scripts\python.exe"
BAT = os.path.join(ROOT, "finetune", "run_ft.bat")
TRANSFORMS = os.path.join(ROOT, "data_toolkit", "transforms.json")
CKPT = {
    "native": os.path.join(ROOT, "ckpt", "full_seg.ckpt"),
    "base": os.path.join(ROOT, "ckpt", "full_seg_w_2d_map.ckpt"),
    "v6": os.path.join(ROOT, "ckpt", "full_seg_v6.ckpt"),
}
GEO_RENDERS = os.path.join(OUT, "geosam2", "renders")
GEO_RESULTS = os.path.join(OUT, "geosam2", "results")
P3SAM_OUT = os.path.join(OUT, "p3sam")
P3SAM_BAT = os.path.join(ROOT, "finetune", "run_p3sam.bat")
AZIMUTHS = (0, 45, 90, 135, 180, 225, 270, 315)

# key -> (file stem in ext_parts, part prompts). Prompts are the names a user would type for the asset.
ASSETS: dict[str, tuple[str, list[str]]] = {
    "human": ("人体", ["head", "torso", "arm", "hand", "leg", "foot"]),
    "dog": ("小狗", ["head", "ear", "body", "leg", "tail"]),
    "robot": ("机器人", ["head", "torso", "arm", "hand", "leg", "foot"]),
    "chair": ("椅子", ["seat", "backrest", "leg", "armrest"]),
    "mickey": ("米老鼠", ["head", "ear", "arm", "hand", "leg", "foot", "tail", "overalls", "base"]),
    "shelf": ("置物架", ["shelf board", "frame"]),
    "pineapple": ("菠萝", ["leaves", "fruit"]),
    "car": ("跑车", ["body", "wheel", "window", "spoiler", "headlight", "mirror"]),
    "sword": ("长剑", ["blade", "guard", "grip", "pommel"]),
    "plane": ("飞机", ["fuselage", "wing", "tail fin", "propeller", "engine", "float", "seat"]),
}
# The silhouette metric cannot tell front from back; these were picked by eye from cand/.
FRONT_OVERRIDE = {"human": 0.0, "mickey": 0.0, "plane": 45.0}

ENV = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", OPENCV_IO_ENABLE_OPENEXR="1",
           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd, log_path: str, cwd: str = ROOT) -> int:
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(str(c) for c in cmd)}\n")
        log.flush()
        t0 = time.time()
        r = subprocess.run([str(c) for c in cmd], cwd=cwd, stdout=log, stderr=subprocess.STDOUT, env=ENV)
        log.write(f"[exit {r.returncode}, {time.time() - t0:.0f}s]\n")
    return r.returncode


def wd(key: str) -> str:
    d = os.path.join(OUT, key)
    os.makedirs(d, exist_ok=True)
    return d


def src_glb(key: str) -> str:
    return os.path.join(SRC, ASSETS[key][0] + ".glb")


def front_json(key: str) -> dict:
    with open(os.path.join(wd(key), "front.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


# ----------------------------------------------------------------------------- front
def stage_front(keys):
    from data_toolkit.front_view import rank_views
    for k in keys:
        d = wd(k)
        fj = os.path.join(d, "front.json")
        paths = [os.path.join(d, "cand", f"c_{a:g}.png") for a in AZIMUTHS]
        if not all(os.path.exists(p) for p in paths):
            say(f"front {k}")
            run([PY, os.path.join(ROOT, "data_toolkit", "render_cond_view.py"), "--glb", src_glb(k), "--transforms",
                 TRANSFORMS, "--out", os.path.join(d, "cand", "c.png"), "--azimuths",
                 ",".join(f"{a:g}" for a in AZIMUTHS)], os.path.join(d, "log.txt"))
        ranked = rank_views(paths)
        az = FRONT_OVERRIDE.get(k, float(AZIMUTHS[paths.index(ranked[0][0])]))
        if os.path.exists(fj) and read_json(fj)["azimuth"] == az:
            continue
        shutil.copyfile(os.path.join(d, "cand", f"c_{az:g}.png"), os.path.join(d, "render.png"))
        shutil.copyfile(os.path.join(d, "cand", f"c_{(az + 180) % 360:g}.png"), os.path.join(d, "render_back.png"))
        write_json(fj, {"azimuth": az, "override": k in FRONT_OVERRIDE,
                        "ranking": [(os.path.basename(p), m) for p, m in ranked]})
        say(f"  {k}: azimuth {az:g}")


# ----------------------------------------------------------------------------- mesh
def stage_mesh(keys, max_faces: int):
    import trimesh
    for k in keys:
        out = os.path.join(OUT, "mesh", f"{k}.glb")
        if os.path.exists(out):
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        s = trimesh.load(src_glb(k), force="scene")
        n = sum(len(g.faces) for g in s.dump() if isinstance(g, trimesh.Trimesh))
        if n <= max_faces:
            shutil.copyfile(src_glb(k), out)
            say(f"mesh {k}: {n} faces, copied")
            continue
        say(f"mesh {k}: {n} faces -> decimate to {max_faces}")
        # the bpy module often dies with 0xC0000005 while tearing down after a successful export
        run([BAT, "decimate_glb.py", "--glb", src_glb(k), "--out", out, "--max_faces", str(max_faces)],
            os.path.join(wd(k), "log.txt"))
        if not os.path.exists(out):
            say(f"  {k}: decimation FAILED")


# ----------------------------------------------------------------------------- sam3
def stage_sam3(keys):
    for k in keys:
        d = wd(k)
        if os.path.exists(os.path.join(d, "legend.json")):
            continue
        say(f"sam3 {k}")
        run([PY_SAM3, os.path.join(ROOT, "sam3_to_2dmap.py"), "--image", os.path.join(d, "render.png"),
             "--out", os.path.join(d, "map.png"), "--legend", os.path.join(d, "legend.json"), "--allow_missing",
             "--threshold", "0.4", "--concept_bank", BANK, "--prompts", *ASSETS[k][1]], os.path.join(d, "log.txt"))
        if os.path.exists(os.path.join(d, "legend.json")):
            leg = read_json(os.path.join(d, "legend.json"))
            say("  " + ", ".join(f"{p['prompt']}:{'-' if p['score'] is None else round(float(p['score']), 2)}"
                                 for p in leg))


# ----------------------------------------------------------------------------- segvigen
def upright(seg_glb: str, ref_glb: str, out_glb: str) -> tuple[str, float]:
    """Re-export a SegviGen output in the source asset's frame (they come out turned about X)."""
    import trimesh
    from eval_fidelity import align_output, load_output
    ref = trimesh.load(ref_glb, force="scene")
    ref_mesh = trimesh.util.concatenate([g for g in ref.dump() if isinstance(g, trimesh.Trimesh)])
    seg = load_output(seg_glb)
    c, e = ref_mesh.bounds.mean(0), ref_mesh.extents.max()
    ref_pts = (trimesh.sample.sample_surface(ref_mesh, 2000)[0] - c) / e
    seg_n = seg.copy()
    seg_n.apply_translation(-seg_n.bounds.mean(0))
    seg_n.apply_scale(1.0 / seg_n.extents.max())
    aligned, frame, cov = align_output(seg_n, ref_pts)
    aligned.apply_scale(e)
    aligned.apply_translation(c)
    aligned.export(out_glb)
    return frame, cov


def stage_segvigen(keys):
    for k in keys:
        d = wd(k)
        az = front_json(k)["azimuth"]
        glb = src_glb(k)
        jobs = {
            "native": [CKPT["native"], "--img", os.path.join(d, "native_render.png"), "--transforms", TRANSFORMS,
                       "--azimuth", f"{az:g}"],
            "base": [CKPT["base"], "--img", os.path.join(d, "map.png"), "--two_d_map"],
            "v6": [CKPT["v6"], "--img", os.path.join(d, "map.png"), "--two_d_map"],
        }
        for tag, (ckpt, *extra) in jobs.items():
            seg = os.path.join(d, f"seg_{tag}.glb")
            if not os.path.exists(seg):
                if tag != "native" and not os.path.exists(os.path.join(d, "map.png")):
                    say(f"segvigen {k} {tag}: no map.png, skipping")
                    continue
                say(f"segvigen {k} {tag}")
                t0 = time.time()
                rc = run([BAT, r"..\inference_full.py", "--ckpt_path", ckpt, "--glb", glb, "--input_vxz",
                          os.path.join(d, "input.vxz"), "--export_glb", seg, *extra], os.path.join(d, "log.txt"))
                times = read_json(os.path.join(d, "times.json")) if os.path.exists(os.path.join(d, "times.json")) else {}
                times[f"segvigen_{tag}"] = round(time.time() - t0, 1)
                write_json(os.path.join(d, "times.json"), times)
                if rc != 0 or not os.path.exists(seg):
                    say(f"  {k} {tag}: FAILED (exit {rc})")
                    continue
            up = os.path.join(d, f"seg_{tag}_upright.glb")
            if not os.path.exists(up):
                frame, cov = upright(seg, glb, up)
                say(f"  {k} {tag}: upright frame={frame} coverage={cov:.3f}")


# ----------------------------------------------------------------------------- geosam2
def stage_geo_prep(keys):
    """GeoSAM2 inputs: 12-view renders of the (decimated) mesh and SAM3 label maps on them. Light on the GPU."""
    os.makedirs(GEO_RENDERS, exist_ok=True)
    glog = os.path.join(OUT, "geosam2", "log.txt")
    todo = [k for k in keys if not os.path.exists(os.path.join(GEO_RENDERS, k, "meta.json"))]
    if todo:
        say(f"geosam2 render {todo}")
        run([BAT, "geosam2_render.py", "--out", GEO_RENDERS, "--glb", *[os.path.join(OUT, "mesh", f"{k}.glb") for k in todo],
             "--name", *todo], glog)
    todo = [k for k in keys if not os.path.exists(os.path.join(GEO_RENDERS, k, "sam3", "summary.json"))]
    if todo:
        prompts_json = os.path.join(OUT, "geosam2", "prompts.json")
        write_json(prompts_json, {k: ASSETS[k][1] for k in keys})
        say(f"geosam2 sam3 masks {todo}")
        run([PY_SAM3, os.path.join(ROOT, "finetune", "geosam2_masks.py"), "--renders", GEO_RENDERS, "--objects", *todo,
             "--prompts_json", prompts_json, "--concept_bank", BANK], glog)
    for k in keys:
        sj = os.path.join(GEO_RENDERS, k, "sam3", "summary.json")
        if os.path.exists(sj):
            s = read_json(sj)
            best = next(v for v in s["views"] if v["view"] == s["best_view"])
            say(f"  {k}: best view {s['best_view']} found {best['n_found']}/{len(s['prompts'])} prompts, "
                f"labelled {best['labelled_share']:.2f}")


def stage_geosam2(keys):
    from geosam2_run import match_view
    stage_geo_prep(keys)
    os.makedirs(GEO_RESULTS, exist_ok=True)
    glog = os.path.join(OUT, "geosam2", "log.txt")
    for k in keys:
        rdir = os.path.join(GEO_RENDERS, k)
        if not os.path.exists(os.path.join(rdir, "sam3", "summary.json")):
            say(f"geosam2 {k}: no SAM3 masks, skipping")
            continue
        summary = read_json(os.path.join(rdir, "sam3", "summary.json"))
        view, iou = match_view(rdir, os.path.join(wd(k), "render.png"))
        views = {"match": view, "best": int(summary["best_view"])}
        write_json(os.path.join(wd(k), "geosam2_views.json"), {**views, "match_iou": iou})
        for name, v in views.items():
            single = os.path.join(GEO_RESULTS, k, f"view{v:02d}")
            if not os.path.exists(os.path.join(single, "labels.json")):
                say(f"geosam2 single {k} {name}=view{v:02d}")
                run([PY_GEO, os.path.join(ROOT, "finetune", "geosam2_run.py"), "--renders", GEO_RENDERS, "--out", GEO_RESULTS,
                     "--objects", k, "--view", str(v)], glog)
            dual = os.path.join(GEO_RESULTS, k, f"dual_view{v:02d}")
            if not os.path.exists(os.path.join(dual, "labels.json")):
                say(f"geosam2 dual {k} {name}=view{v:02d}")
                run([PY_GEO, os.path.join(ROOT, "finetune", "geosam2_dual.py"), "--renders", GEO_RENDERS, "--out", GEO_RESULTS,
                     "--objects", k, "--view", str(v)], glog)
        for cfg, (mesh, npy, lj) in geosam2_outputs(k).items():
            parts = os.path.join(wd(k), f"geosam2_{cfg}_parts.glb")
            if not os.path.exists(parts):
                run([BAT, "geosam2_to_glb.py", "--mesh", mesh, "--labels", npy, "--labels_json", lj, "--ref", src_glb(k),
                     "--out", parts], glog)


def geosam2_outputs(k: str) -> dict[str, tuple[str, str, str]]:
    """cfg -> (mesh glb, face-label npy, labels.json) for every finished GeoSAM2 run of this asset."""
    out = {}
    vj = os.path.join(wd(k), "geosam2_views.json")
    if not os.path.exists(vj):
        return out
    views = read_json(vj)
    for name in ("match", "best"):
        v = views[name]
        d = os.path.join(GEO_RESULTS, k, f"view{v:02d}")
        lj = os.path.join(d, "labels.json")
        if os.path.exists(lj):
            info = read_json(lj)
            if info.get("outputs"):
                npy = os.path.join(d, info["outputs"][0])
                out[f"single_{name}"] = (npy.replace(".npy", ".glb"), npy, lj)
                if os.path.exists(npy.replace(".npy", "_filled.npy")):
                    out[f"single_{name}_filled"] = (npy.replace(".npy", ".glb"), npy.replace(".npy", "_filled.npy"), lj)
        d = os.path.join(GEO_RESULTS, k, f"dual_view{v:02d}")
        lj = os.path.join(d, "labels.json")
        if os.path.exists(lj) and "error" not in read_json(lj) and os.path.exists(os.path.join(d, "dual_filled.npy")):
            out[f"dual_{name}"] = (os.path.join(d, "dual.glb"), os.path.join(d, "dual.npy"), lj)
            out[f"dual_{name}_filled"] = (os.path.join(d, "dual.glb"), os.path.join(d, "dual_filled.npy"), lj)
    return out


# ----------------------------------------------------------------------------- p3sam
def p3sam_dir(k: str) -> str:
    return os.path.join(P3SAM_OUT, k)


def stage_p3sam(keys):
    """Original P3-SAM auto-mask on the same (decimated) mesh GeoSAM2 gets. It is the one method here that never
    sees SAM3 or a name: pure learned 3D part prior with point prompts, so its column shows what the geometry
    alone suggests as parts. Runs in its own conda env through run_p3sam.bat; trimesh reads mesh/<k>.glb in the
    asset frame (checked: extents / centres identical to the source), so no re-alignment is needed."""
    todo = [k for k in keys if os.path.exists(os.path.join(OUT, "mesh", f"{k}.glb"))
            and not os.path.exists(os.path.join(p3sam_dir(k), "faces.npy"))]
    if todo:
        say(f"p3sam: {', '.join(todo)}")
        run([P3SAM_BAT, "--out", P3SAM_OUT, "--glb", *[os.path.join(OUT, "mesh", f"{k}.glb") for k in todo]],
            os.path.join(OUT, "p3sam_log.txt"))
    for k in keys:
        d = p3sam_dir(k)
        glb = os.path.join(wd(k), "p3sam_parts.glb")
        if os.path.exists(os.path.join(d, "faces.npy")) and not os.path.exists(glb):
            p3sam_parts_glb(d, glb)
            info = read_json(os.path.join(d, "info.json"))
            say(f"  {k}: {info['n_parts']} parts, {info['seconds']:.0f}s")
        elif not os.path.exists(os.path.join(d, "faces.npy")):
            say(f"  {k}: no P3-SAM result")


def p3sam_parts_glb(d: str, out: str):
    """faces.npy -> one flat-coloured sub-mesh per P3-SAM part, largest part first in the palette; -1 grey."""
    import trimesh
    from ablation_score import PALETTE
    from common import GREY
    from geosam2_to_glb import load_mesh
    mesh = load_mesh(os.path.join(d, "mesh.glb"))
    labels = np.rint(np.load(os.path.join(d, "faces.npy"))).astype(np.int64)
    if len(labels) != len(mesh.faces):
        raise ValueError(f"{d}: {len(labels)} labels for {len(mesh.faces)} faces")
    areas = mesh.area_faces
    parts = sorted([int(u) for u in np.unique(labels) if u >= 0], key=lambda u: -float(areas[labels == u].sum()))
    scene = trimesh.Scene()
    for rank, k in enumerate(parts + ([-1] if (labels < 0).any() else [])):
        sub = mesh.submesh([np.nonzero(labels == k)[0]], append=True)
        name, color = ("unlabelled", GREY) if k < 0 else (f"part_{k}", PALETTE[rank % len(PALETTE)])
        mat = trimesh.visual.material.PBRMaterial(baseColorFactor=[*color, 255], metallicFactor=0.0, roughnessFactor=0.9)
        sub.visual = trimesh.visual.TextureVisuals(material=mat)
        scene.add_geometry(sub, node_name=name, geom_name=name)
    scene.export(out)


# ----------------------------------------------------------------------------- score
UNCOVERED, UNLABELLED = -1, -2
NAMED_SEGVIGEN = ("base", "v6")


def surface_samples(ref_glb: str, n: int = 60000):
    """Uniform samples on the original asset, in its own unit-cube frame (bbox centre, max extent = 1)."""
    import trimesh
    ref = trimesh.load(ref_glb, force="scene")
    mesh = to_unit(trimesh.util.concatenate([g for g in ref.dump() if isinstance(g, trimesh.Trimesh)]))
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    spacing = float(np.sqrt(mesh.area / n))
    return pts, spacing


def to_unit(mesh):
    m = mesh.copy()
    m.apply_translation(-m.bounds.mean(0))
    m.apply_scale(1.0 / m.extents.max())
    return m


def labels_on_samples(pred, pts, max_dist: float = 0.02):
    """Every output is self-normalised to the unit cube and the x-axis quarter turn that lands on the asset
    is picked by surface coverage, so SegviGen's Y-up remesh, GeoSAM2's Z-up mesh and the asset agree
    regardless of how their exporters interpret the source scene graph."""
    import trimesh
    from eval_fidelity import align_output
    rng = np.random.default_rng(0)
    mesh, frame, cov = align_output(to_unit(pred.mesh), pts[rng.choice(len(pts), 2000, replace=False)])
    closest, dist, face_idx = trimesh.proximity.ProximityQuery(mesh).on_surface(pts)
    lab = pred.labels_at(mesh, face_idx, closest)
    return np.where(dist <= max_dist, lab, UNCOVERED), frame, cov


def load_preds(k: str) -> dict:
    """method -> prediction object (TexturePred / FaceLabelPred with .mesh, .labels_at, .name_of)."""
    from common import GREY
    from eval_parts import FaceLabelPred, TexturePred, legend_palette
    d = wd(k)
    preds = {}
    leg = os.path.join(d, "legend.json")
    for tag in ("native", "base", "v6"):
        glb = os.path.join(d, f"seg_{tag}_upright.glb")
        if not os.path.exists(glb):
            continue
        if tag in NAMED_SEGVIGEN and os.path.exists(leg):
            palette, names = legend_palette(leg)
        else:
            palette, names = None, {}
        preds[f"segvigen_{tag}"] = TexturePred(glb, palette or [GREY], names) if palette else NativePred(glb)
    for cfg, (mesh, npy, lj) in geosam2_outputs(k).items():
        names = {int(i): n for i, n in read_json(lj)["ids"].items()}
        preds[f"geosam2_{cfg}"] = FaceLabelPred(mesh, npy, names, {0, 999})
    if os.path.exists(os.path.join(p3sam_dir(k), "faces.npy")):
        preds["p3sam"] = FaceLabelPred(os.path.join(p3sam_dir(k), "mesh.glb"), os.path.join(p3sam_dir(k), "faces.npy"),
                                       {}, {-1})
    return preds


class NativePred:
    """SegviGen native output: no palette is known, so colours are quantised (k-means over the samples)."""

    def __init__(self, glb: str):
        from eval_fidelity import load_output
        self.mesh = load_output(glb)
        self.centers = None

    def labels_at(self, mesh, face_idx, closest):
        from eval_fidelity import texture_lookup
        from scipy.cluster.vq import kmeans2
        colors = texture_lookup(mesh, face_idx, closest)
        # SegviGen paints each part in a flat colour; cluster at fixed spacing (max_dist 40 in RGB) by
        # greedy merging of the k-means centres of an over-clustered solution.
        rng = np.random.default_rng(0)
        sub = colors[rng.choice(len(colors), min(len(colors), 20000), replace=False)]
        centers, _ = kmeans2(sub.astype(np.float64), 24, minit="++", seed=0)
        keep = []
        for cc in centers:
            if all(np.linalg.norm(cc - kk) > 40 for kk in keep):
                keep.append(cc)
        self.centers = np.asarray(keep)
        lab = np.linalg.norm(colors[:, None, :] - self.centers[None], axis=-1).argmin(1)
        counts = np.bincount(lab, minlength=len(self.centers))
        # colours under 0.5% of the surface are anti-aliasing / seams, merge into nearest big colour
        big = np.nonzero(counts >= 0.005 * len(lab))[0]
        if len(big) and len(big) < len(self.centers):
            remap = big[np.linalg.norm(self.centers[:, None] - self.centers[big][None], axis=-1).argmin(1)]
            lab = remap[lab]
        return lab

    def name_of(self, label: int):
        return None


def fragments(pts, lab, spacing):
    from eval_fidelity import components_per_label
    segs = [s for s in np.unique(lab) if s >= 0]
    if not segs:
        return {}
    idx = {s: i for i, s in enumerate(segs)}
    sel = lab >= 0
    dense = np.array([idx[s] for s in lab[sel]])
    comps = components_per_label(pts[sel], dense, len(segs), spacing)
    return {int(s): comps[idx[s]] for s in segs}


def boundary_share(pts, lab, spacing):
    from scipy.spatial import cKDTree
    pairs = cKDTree(pts).query_pairs(1.5 * spacing, output_type="ndarray")
    if len(pairs) == 0:
        return None
    valid = (lab[pairs[:, 0]] != UNCOVERED) & (lab[pairs[:, 1]] != UNCOVERED)
    pairs = pairs[valid]
    diff = lab[pairs[:, 0]] != lab[pairs[:, 1]]
    return float(len(np.unique(pairs[diff].ravel())) / len(pts))


def matched_miou(a, b):
    """Class-agnostic agreement between two labelings (unlabelled counts as a segment, uncovered ignored)."""
    from scipy.optimize import linear_sum_assignment
    ok = (a != UNCOVERED) & (b != UNCOVERED)
    a, b = a[ok], b[ok]
    ua, ub = np.unique(a), np.unique(b)
    ia = {s: i for i, s in enumerate(ua)}
    ib = {s: i for i, s in enumerate(ub)}
    table = np.zeros((len(ua), len(ub)), dtype=np.int64)
    np.add.at(table, ([ia[s] for s in a], [ib[s] for s in b]), 1)
    union = table.sum(1)[:, None] + table.sum(0)[None] - table
    iou = np.where(union > 0, table / np.maximum(union, 1), 0)
    r, c = linear_sum_assignment(-iou)
    w = table.sum(1)[r] / max(1, table.sum())
    return float((iou[r, c] * w).sum()), float(iou[r, c].mean())


def name_iou(pa, la, pb, lb):
    """Per shared name: IoU of the two methods' regions carrying that name."""
    def regions(p, l):
        out = {}
        for s in np.unique(l):
            if s >= 0 and p.name_of(int(s)):
                for n in p.name_of(int(s)).split("+"):
                    out.setdefault(n.strip().lower(), np.zeros(len(l), bool))
                    out[n.strip().lower()] |= l == s
        return out
    ra, rb = regions(pa, la), regions(pb, lb)
    res = {}
    for n in sorted(set(ra) | set(rb)):
        a = ra.get(n, np.zeros(len(la), bool))
        b = rb.get(n, np.zeros(len(lb), bool))
        u = (a | b).sum()
        res[n] = float((a & b).sum() / u) if u else None
    return res


def stage_score(keys):
    all_scores = read_json(os.path.join(OUT, "scores.json")) if os.path.exists(os.path.join(OUT, "scores.json")) else {}
    for k in keys:
        say(f"score {k}")
        preds = load_preds(k)
        if not preds:
            continue
        pts, spacing = surface_samples(src_glb(k))
        labs, frames = {}, {}
        for m, p in preds.items():
            lab, frame, cov = labels_on_samples(p, pts)
            labs[m], frames[m] = lab, (frame, cov)
        rows = {}
        for m, lab in labs.items():
            fr = fragments(pts, lab, spacing)
            segs = [s for s in fr if s >= 0]
            named = [s for s in segs if preds[m].name_of(s)]
            rows[m] = {
                "frame": frames[m][0], "surface_coverage": frames[m][1],
                "n_segments": len(segs),
                "n_named_segments": len(named),
                "names": sorted({preds[m].name_of(s) for s in named}),
                "fragments_total": int(sum(fr[s] for s in segs)),
                "fragments_per_segment": float(np.mean([fr[s] for s in segs])) if segs else None,
                "segments_fragmented": int(sum(fr[s] >= 2 for s in segs)),
                "unlabelled_share": float((lab == UNLABELLED).mean()),
                "uncovered_share": float((lab == UNCOVERED).mean()),
                "boundary_share": boundary_share(pts, lab, spacing),
                "segment_shares": {str(s): float((lab == s).mean()) for s in segs},
            }
        agree = {}
        ms = list(labs)
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                wm, mm = matched_miou(labs[ms[i]], labs[ms[j]])
                entry = {"miou_weighted": wm, "miou_mean": mm}
                if ms[i].split("_")[-1] in NAMED_SEGVIGEN or ms[i].startswith("geosam2"):
                    if ms[j].split("_")[-1] in NAMED_SEGVIGEN or ms[j].startswith("geosam2"):
                        ni = name_iou(preds[ms[i]], labs[ms[i]], preds[ms[j]], labs[ms[j]])
                        vals = [v for v in ni.values() if v is not None]
                        entry["name_iou"] = ni
                        entry["name_miou"] = float(np.mean(vals)) if vals else None
                agree[f"{ms[i]}|{ms[j]}"] = entry
        legend = read_json(os.path.join(wd(k), "legend.json")) if os.path.exists(os.path.join(wd(k), "legend.json")) else []
        times = read_json(os.path.join(wd(k), "times.json")) if os.path.exists(os.path.join(wd(k), "times.json")) else {}
        for cfg, (_, _, lj) in geosam2_outputs(k).items():
            if not cfg.endswith("_filled"):
                times[f"geosam2_{cfg}"] = read_json(lj).get("seconds")
        if os.path.exists(os.path.join(p3sam_dir(k), "info.json")):
            times["p3sam"] = read_json(os.path.join(p3sam_dir(k), "info.json")).get("seconds")
        all_scores[k] = {
            "prompts": ASSETS[k][1],
            "sam3_front": {p["prompt"]: (None if p["score"] is None else round(float(p["score"]), 3)) for p in legend
                           if p.get("prompt") != "<unassigned>"},
            "methods": rows, "agreement": agree, "seconds": times, "n_samples": int(len(pts)),
        }
        write_json(os.path.join(OUT, "scores.json"), all_scores)
        for m, r in rows.items():
            say(f"  {m:28s} segs={r['n_segments']:2d} frag/seg={r['fragments_per_segment'] or 0:.2f} "
                f"unlab={r['unlabelled_share']:.2f} uncov={r['uncovered_share']:.2f} bnd={r['boundary_share'] or 0:.3f}")


# ----------------------------------------------------------------------------- montage
COLUMNS = [
    ("input", "Input"),
    ("map", "SAM3 map (bank)"),
    ("segvigen_native", "SegviGen native"),
    # original P3-SAM auto-mask (finetune/p3sam_run.py in its conda env): class-agnostic, never sees SAM3
    ("p3sam", "P3-SAM (auto, no SAM3)"),
    ("segvigen_base", "SegviGen base + SAM3"),
    ("segvigen_v6", "Ours: v6 + SAM3"),
    ("geosam2_single_match", "GeoSAM2 single (raw)"),
    ("geosam2_dual_match_filled", "GeoSAM2 dual (filled)"),
    # best rows of the geosam2_ablate.py study (GLBs written by ablation_score.py ext); the dual column above
    # predates the transform-order fix, geo_p2 is the corrected two-view GeoSAM2 propagation
    ("abl_geo_p2", "GeoSAM2 p2 (fixed)"),
    ("abl_lift_sam3track_p2_match", "SAM3 tracker p2 (best)"),
]


def render_view(glb: str, out_png: str, az: float, log: str) -> str | None:
    """One orbit view, self-normalised like the input renders (a --ref_glb would be read through Blender's view
    of the source scene graph, which differs from trimesh's by a uniform factor on some of these assets).
    A single azimuth is written to --out itself, so the suffix goes in here."""
    target = out_png.replace(".png", f"_{az:g}.png")
    if not os.path.exists(target):
        run([PY, os.path.join(ROOT, "data_toolkit", "render_cond_view.py"), "--glb", glb,
             "--transforms", TRANSFORMS, "--out", target, "--azimuths", f"{az:g}"], log)
    return target if os.path.exists(target) else None


def stage_montage(keys):
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 16)
    except OSError:
        font = ImageFont.load_default()
    def tiles_for(k, side):
        d = wd(k)
        az = front_json(k)["azimuth"]
        if side == "back":
            az = (az + 180) % 360
        tiles = []
        for col, _ in COLUMNS:
            if col == "input":
                tiles.append(os.path.join(d, "render.png" if side == "front" else "render_back.png"))
            elif col == "map":
                tiles.append(os.path.join(d, "map.png") if side == "front" else None)
            else:
                glb = os.path.join(d, f"seg_{col.split('_')[1]}_upright.glb" if col.startswith("segvigen")
                                   else f"{col}_parts.glb")
                tiles.append(render_view(glb, os.path.join(d, "vis", f"{col}.png"), az, os.path.join(d, "log.txt"))
                             if os.path.exists(glb) else None)
        return tiles

    def grid(rows, S, out, row_label):
        W, H = len(COLUMNS) * S, len(rows) * (S + 22) + 22
        canvas = Image.new("RGB", (W, H), "white")
        draw = ImageDraw.Draw(canvas)
        for c, (_, label) in enumerate(COLUMNS):
            draw.text((c * S + 6, 3), label, fill="black", font=font)
        for r, (name, tiles) in enumerate(rows):
            y = 22 + r * (S + 22)
            draw.text((6, y + 3), row_label(name), fill="black", font=font)
            for c, t in enumerate(tiles):
                if t and os.path.exists(t):
                    im = Image.open(t).convert("RGBA")
                    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
                    im = Image.alpha_composite(bg, im).convert("RGB").resize((S, S))
                    canvas.paste(im, (c * S, y + 22))
                else:
                    draw.text((c * S + S // 2 - 20, y + 22 + S // 2), "n/a", fill="grey", font=font)
        canvas.save(out)
        say(f"saved {out}")

    # overview: one row per asset, front and back sheets
    for side in ("front", "back"):
        grid([(k, tiles_for(k, side)) for k in keys], 300, os.path.join(OUT, f"compare_{side}.png"),
             lambda k: f"{k} ({ASSETS[k][0]})  az={front_json(k)['azimuth']:g}")
    # detail: one sheet per asset at render resolution, front row + back row
    os.makedirs(os.path.join(OUT, "detail"), exist_ok=True)
    for k in keys:
        grid([("front", tiles_for(k, "front")), ("back", tiles_for(k, "back"))], 512,
             os.path.join(OUT, "detail", f"{k}.png"), lambda side: f"{k} ({ASSETS[k][0]}) {side}")


# ----------------------------------------------------------------------------- report
METHOD_LABEL = {
    "segvigen_native": "SegviGen 原生 (full_seg)",
    "p3sam": "P3-SAM 原版 (auto-mask, 无 SAM3)",
    "segvigen_base": "SegviGen base + SAM3",
    "segvigen_v6": "我们的方案 (v6 + SAM3)",
    "geosam2_single_match": "GeoSAM2 single (同视角, raw)",
    "geosam2_single_match_filled": "GeoSAM2 single (同视角, 填充)",
    "geosam2_single_best": "GeoSAM2 single (best 视角, raw)",
    "geosam2_single_best_filled": "GeoSAM2 single (best 视角, 填充)",
    "geosam2_dual_match": "GeoSAM2 dual (同视角, raw)",
    "geosam2_dual_match_filled": "GeoSAM2 dual (同视角, 填充)",
    "geosam2_dual_best": "GeoSAM2 dual (best 视角, raw)",
    "geosam2_dual_best_filled": "GeoSAM2 dual (best 视角, 填充)",
}
MAIN_METHODS = ["segvigen_native", "p3sam", "segvigen_base", "segvigen_v6", "geosam2_single_match", "geosam2_dual_match_filled"]


def fmt(v, nd=2):
    return "–" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def stage_report(keys):
    scores = read_json(os.path.join(OUT, "scores.json"))
    keys = [k for k in keys if k in scores]
    methods = [m for m in METHOD_LABEL if any(m in scores[k]["methods"] for k in keys)]
    L = []
    L.append(f"# 外部资产对照：SegviGen 原生 / 我们的方案 / GeoSAM2（{time.strftime('%Y-%m-%d')}）\n")
    L.append("资产：`datasets/3D拆件.zip` 的 10 个模型（`datasets/ext_parts/`）。没有部件 GT，所以这里的数字都是**结构统计与方法间一致性**，"
             "不是准确率；实际效果看 `compare_front.png` / `compare_back.png`（条件视角与其背面，同一机位）。\n")
    L.append("![front](compare_front.png)\n\n![back](compare_back.png)\n")
    L.append("## 1. 设置\n")
    L.append("| 资产 | 提示词 | 正面机位 | 前视图 SAM3 命中（分数） | GeoSAM2 best 视角命中 |")
    L.append("|---|---|---|---|---|")
    for k in keys:
        s = scores[k]
        sam = ", ".join(f"{p}:{fmt(v)}" for p, v in s["sam3_front"].items())
        gj = os.path.join(GEO_RENDERS, k, "sam3", "summary.json")
        geo = "–"
        if os.path.exists(gj):
            g = read_json(gj)
            b = next(v for v in g["views"] if v["view"] == g["best_view"])
            geo = f"view {g['best_view']}: {b['n_found']}/{len(g['prompts'])}, 前景覆盖 {b['labelled_share']:.2f}"
        L.append(f"| {k} ({ASSETS[k][0]}) | {', '.join(s['prompts'])} | {front_json(k)['azimuth']:g} | {sam} | {geo} |")
    L.append("")
    L.append("三路输入相同：SegviGen base/v6 用前视图上 SAM3+概念库 v3 的 2D 图；GeoSAM2 用同一套 SAM3+概念库在其 12 视角渲染上的标签图，"
             "`match` = 与前视图轮廓 IoU 最高的视角（同一输入视角），`best` = SAM3 命中最多的视角。SegviGen 原生 = `full_seg.ckpt` 直接看前视图渲染，无 SAM3。"
             "P3-SAM 原版 = Hunyuan3D-Part 的 `auto_mask.py`（400 个 FPS 点提示、阈值 0.95、含默认后处理）直接跑在 GeoSAM2 用的同一网格上，无文本、无名字。"
             "GeoSAM2 / P3-SAM 输入网格 >200k 面时减面到 200k（人体/机器人/椅子/跑车/长剑/飞机）。\n")
    L.append("## 2. 结构统计（原网格表面均匀采样 6 万点，在各方法输出最近点读标签）\n")
    L.append("- `段数`：不同标签数（灰/未标不计）；`碎片/段`：每段的连通块数均值（≥ 该段 1% 的块才算），1.0 = 每个部件是一整块；`碎段`：≥2 块的段数。"
             "同名的重复部件（两条腿、七层搁板）天然 ≥2 块，所以碎片数只能在同一资产上横向比较方法，不能跨资产比；\n"
             "- `留白`：灰色 / 未标注面的比例；`边界密度`：有异标签邻点的采样点比例（越高边界越长/越碎）；\n"
             "- `秒`：SegviGen 为整个进程（含加载 TRELLIS/DINOv3 约 30 s + 体素化 + 采样），GeoSAM2 single 为整个 `inference.py` 进程，"
             "dual 为加载后的推理时间；都不含渲染与 SAM3。\n")
    for k in keys:
        s = scores[k]
        L.append(f"### {k}（{ASSETS[k][0]}）\n")
        L.append(f"![{k}](detail/{k}.png)\n")
        L.append("| 方法 | 段数 | 命名段 | 碎片/段 | 碎段 | 留白 | 边界密度 | 秒 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for m in methods:
            r = s["methods"].get(m)
            if not r:
                continue
            sec = s["seconds"].get(m) or s["seconds"].get(m.replace("_filled", ""))
            L.append(f"| {METHOD_LABEL[m]} | {r['n_segments']} | {r['n_named_segments']} | {fmt(r['fragments_per_segment'])} | "
                     f"{r['segments_fragmented']} | {fmt(r['unlabelled_share'])} | {fmt(r['boundary_share'], 3)} | {fmt(sec, 0)} |")
        L.append("")
    L.append("### 均值（所有资产）\n")
    L.append("| 方法 | 段数 | 碎片/段 | 碎段 | 留白 | 边界密度 | 秒 | 资产数 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for m in methods:
        rows = [scores[k]["methods"][m] for k in keys if m in scores[k]["methods"]]
        if not rows:
            continue
        secs = [scores[k]["seconds"].get(m) or scores[k]["seconds"].get(m.replace("_filled", "")) for k in keys if m in scores[k]["methods"]]
        secs = [x for x in secs if x is not None]
        mean = lambda f: float(np.mean([f(r) for r in rows if f(r) is not None])) if any(f(r) is not None for r in rows) else None
        L.append(f"| {METHOD_LABEL[m]} | {fmt(mean(lambda r: r['n_segments']), 1)} | {fmt(mean(lambda r: r['fragments_per_segment']))} | "
                 f"{fmt(mean(lambda r: r['segments_fragmented']), 1)} | {fmt(mean(lambda r: r['unlabelled_share']))} | "
                 f"{fmt(mean(lambda r: r['boundary_share']), 3)} | {fmt(float(np.mean(secs)) if secs else None, 0)} | {len(rows)} |")
    L.append("")
    L.append("## 3. 方法间一致性（同一采样点；类无关 = 匈牙利匹配后按段大小加权的 IoU；按名 = 同名区域 IoU 均值）\n")
    pairs = [("segvigen_v6", "segvigen_base"), ("segvigen_v6", "segvigen_native"), ("segvigen_v6", "geosam2_dual_match_filled"),
             ("segvigen_v6", "geosam2_single_match"), ("segvigen_base", "geosam2_dual_match_filled"),
             ("segvigen_native", "geosam2_dual_match_filled"), ("segvigen_v6", "p3sam"), ("segvigen_native", "p3sam")]
    L.append("| 资产 | " + " | ".join(f"{METHOD_LABEL[a]} vs {METHOD_LABEL[b]}" for a, b in pairs) + " |")
    L.append("|---|" + "---|" * len(pairs))
    agg = {p: [] for p in pairs}
    for k in keys:
        cells = []
        for a, b in pairs:
            e = scores[k]["agreement"].get(f"{a}|{b}") or scores[k]["agreement"].get(f"{b}|{a}")
            if not e:
                cells.append("–")
                continue
            agg[(a, b)].append(e)
            cells.append(f"{e['miou_weighted']:.2f}" + (f" / 名 {e['name_miou']:.2f}" if e.get("name_miou") is not None else ""))
        L.append(f"| {k} | " + " | ".join(cells) + " |")
    cells = []
    for p in pairs:
        es = agg[p]
        if not es:
            cells.append("–")
            continue
        w = float(np.mean([e["miou_weighted"] for e in es]))
        nm = [e["name_miou"] for e in es if e.get("name_miou") is not None]
        cells.append(f"**{w:.2f}**" + (f" / 名 **{float(np.mean(nm)):.2f}**" if nm else ""))
    L.append("| **均值** | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## 4. 逐资产各方法的段与名字\n")
    for k in keys:
        L.append(f"- **{k}**：" + "；".join(
            f"{METHOD_LABEL[m]}: {scores[k]['methods'][m]['n_segments']} 段" +
            (f" [{', '.join(scores[k]['methods'][m]['names'])}]" if scores[k]["methods"][m]["names"] else "")
            for m in MAIN_METHODS if m in scores[k]["methods"]))
    L.append("")
    notes = os.path.join(OUT, "notes.md")
    if os.path.exists(notes):
        with open(notes, "r", encoding="utf-8") as f:
            L.append(f.read())
    with open(os.path.join(OUT, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    say(f"saved {os.path.join(OUT, 'REPORT.md')}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="+",
                    choices=["front", "mesh", "sam3", "segvigen", "geo_prep", "geosam2", "p3sam", "score", "montage",
                             "report", "all"])
    ap.add_argument("--assets", nargs="*", default=None)
    ap.add_argument("--max_faces", type=int, default=200000)
    args = ap.parse_args()
    keys = args.assets or list(ASSETS)
    stages = (["front", "mesh", "sam3", "segvigen", "geosam2", "p3sam", "score", "montage", "report"] if "all" in args.stages
              else args.stages)
    os.makedirs(OUT, exist_ok=True)
    for st in stages:
        say(f"=== stage {st}")
        {"front": stage_front, "mesh": lambda ks: stage_mesh(ks, args.max_faces), "sam3": stage_sam3,
         "segvigen": stage_segvigen, "geo_prep": stage_geo_prep, "geosam2": stage_geosam2, "p3sam": stage_p3sam,
         "score": stage_score, "montage": stage_montage, "report": stage_report}[st](keys)
    say("EXT BENCH DONE")


if __name__ == "__main__":
    main()
