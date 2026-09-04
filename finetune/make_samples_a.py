"""Path A: real SAM3 conditions bound to ground-truth parts.

Per object and view:
  1. bpy renders the textured input from transforms.json (+ azimuth)      [SegviGen venv]
  2. SAM3 segments the render with the part names as prompts              [.venv_holo, subprocess]
  3. each prompt mask is bound to the GT parts it covers (>= --cover of the part's visible
     pixels), several parts under one prompt become one colour group, parts no prompt
     claimed are GREY in 2D and 3D, partially covered parts keep their full colour in 3D
  4. the SAM3-coloured map, its DINOv3 cond and the recoloured target latent are written

    python finetune/make_samples_a.py --dataset_root <root> --azimuths 0,135
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np

import common
from common import ObjectDir, LABEL_BG, LABEL_GREY
from corrupt import visible_parts
from make_samples_b import view_tag, ensure_ids_raster

DEFAULT_PY_SAM3 = os.environ.get("SEGVIGEN_PY_SAM3", r"E:\AI_New\ModelGen\.venv_holo\Scripts\python.exe")


def unique_prompts(names: list[str]) -> list[str]:
    seen = []
    for n in names:
        n = n.strip()
        if n and n not in seen:
            seen.append(n)
    return seen


def render_views(obj: ObjectDir, azimuths: list[float], engine: str, samples: int, force: bool) -> list[str]:
    from bpy_render import render_from_transforms
    pending = [az for az in azimuths if force or not os.path.exists(os.path.join(obj.view_dir(view_tag(az)), "render.png"))]
    if pending:
        tmp = os.path.join(obj.views_dir, "render.png")
        os.makedirs(obj.views_dir, exist_ok=True)
        written = render_from_transforms(obj.input_glb, common.TRANSFORMS, tmp, engine=engine, azimuths=pending, samples=samples)
        for az, path in zip(pending, written):
            vdir = obj.view_dir(view_tag(az))
            os.makedirs(vdir, exist_ok=True)
            os.replace(path, os.path.join(vdir, "render.png"))
    return [obj.view_dir(view_tag(az)) for az in azimuths]


def bind_masks(labels: np.ndarray, masks: np.ndarray, prompts: list[str], scores: np.ndarray,
               cover: float, partial_precision: float, partial_cover: float):
    """Assign prompt masks to GT parts; returns (paint_labels, groups, grey_parts, binding report)."""
    fg = labels != LABEL_BG
    visible = visible_parts(labels)
    part_px = {p: (labels == p) for p in visible}
    part_area = {p: float(m.sum()) for p, m in part_px.items()}

    paint = np.where(fg, LABEL_GREY, LABEL_BG).astype(np.int16)
    taken: set[int] = set()
    groups: list[list[int]] = []
    report = []
    order = sorted(range(len(prompts)), key=lambda k: int(masks[k].sum()))
    for k in order:
        m = masks[k] & fg
        area = float(m.sum())
        entry = {"prompt": prompts[k], "score": float(scores[k]), "pixels": int(area), "parts": [], "mode": "unbound"}
        if area == 0:
            report.append(entry)
            continue
        cov = {p: float((m & part_px[p]).sum()) / part_area[p] for p in visible}
        prec = {p: float((m & part_px[p]).sum()) / area for p in visible}
        bound = [p for p in visible if cov[p] >= cover and p not in taken]
        mode = "full"
        if not bound:
            best = max(visible, key=lambda p: prec[p])
            if prec[best] >= partial_precision and cov[best] >= partial_cover and best not in taken:
                bound = [best]
                mode = "partial"
        if not bound:
            entry["coverage"] = {str(p): round(cov[p], 3) for p in visible if cov[p] > 0.05}
            report.append(entry)
            continue
        taken.update(bound)
        groups.append(sorted(bound))
        paint[m & (paint == LABEL_GREY)] = bound[0]
        entry.update({"parts": sorted(bound), "mode": mode,
                      "coverage": {str(p): round(cov[p], 3) for p in bound}})
        report.append(entry)
    grey_parts = [p for p in visible if p not in taken]
    return paint, groups, grey_parts, report


def process_object(obj: ObjectDir, encoders, cond_models, azimuths, rng, args, prompts_override) -> dict:
    meta = common.prepare_object(obj, encoders, force=args.force)
    parts = common.load_parts(obj)
    aabb = np.asarray(meta["aabb"])
    n_parts = meta["n_parts"]
    names = obj.names() or [f"part {i}" for i in range(n_parts)]
    prompts = unique_prompts(prompts_override or names)

    view_dirs = render_views(obj, azimuths, args.engine, args.samples, args.force)
    for az, vdir in zip(azimuths, view_dirs):
        ensure_ids_raster(obj, parts, aabb, az, force=args.force)
        with open(os.path.join(vdir, "prompts.json"), "w", encoding="utf-8") as f:
            json.dump(prompts, f, ensure_ascii=False, indent=2)

    pending = [v for v in view_dirs if args.force or not os.path.exists(os.path.join(v, "sam3_masks.npz"))]
    if pending:
        cmd = [args.py_sam3, os.path.join(common.ROOT, "finetune", "sam3_masks.py"), "--views", *pending,
               "--threshold", str(args.threshold)]
        if args.concept_bank:
            cmd += ["--concept_bank", args.concept_bank]
        if args.force:
            cmd.append("--force")
        subprocess.run(cmd, check=True, cwd=common.ROOT)

    per_view = []
    for az, vdir in zip(azimuths, view_dirs):
        tag = view_tag(az)
        labels = np.load(os.path.join(vdir, "ids.npy"))
        data = np.load(os.path.join(vdir, "sam3_masks.npz"), allow_pickle=True)
        masks, scores = data["masks"], data["scores"]
        view_prompts = [str(p) for p in data["prompts"]]
        paint, groups, grey, report = bind_masks(labels, masks, view_prompts, scores,
                                                 args.cover, args.partial_precision, args.partial_cover)
        hidden = [p for p in range(n_parts) if p not in visible_parts(labels)]
        per_view.append({"tag": tag, "paint": paint, "groups": groups, "grey": grey, "report": report, "hidden": hidden})

    source = "v2" if os.path.exists(os.path.join(obj.path, "names_v1.json")) else "v1"
    written = []
    if args.pair and len(per_view) > 1:
        groups, grey, colors = joint_groups(per_view, n_parts, args.hidden_policy, rng)
        if not groups:
            print("  SAM3 bound nothing in any view, skipping object")
            return {"object": obj.path, "n_parts": n_parts, "variants": []}
        pair_tags = [v["tag"] for v in per_view if v["groups"]]
        first = None
        for v in per_view:
            if not v["groups"]:
                print(f"  {v['tag']}: SAM3 bound nothing, skipping view")
                continue
            extra = {"prompts": v["report"], "hidden_parts": v["hidden"], "threshold": args.threshold,
                     "part_names": names, "concept_bank": args.concept_bank, "names_source": source,
                     "pair": "sam3", "pair_views": pair_tags}
            vname = f"sam3_{v['tag']}"
            written.append(common.write_variant(obj, vname, v["paint"], groups, grey, colors, "sam3", v["tag"],
                                                encoders, cond_models, extra=extra, force=args.force,
                                                target_from=first))
            first = first or vname
            bound_prompts = sum(1 for r in v["report"] if r["parts"])
            print(f"  {v['tag']}: {bound_prompts}/{len(v['report'])} prompts bound (joint groups={groups}, grey={grey})")
        return {"object": obj.path, "n_parts": n_parts, "variants": written}

    for v in per_view:
        groups, grey, hidden, tag = v["groups"], v["grey"], v["hidden"], v["tag"]
        if args.hidden_policy == "grey":
            grey = sorted(set(grey) | set(hidden))
        else:
            # Invisible parts keep their own colour (paper behaviour): give each its own group.
            groups = groups + [[p] for p in hidden]
        if not groups:
            print(f"  {tag}: SAM3 bound nothing, skipping view")
            continue
        colors = common.random_palette(rng, len(groups))
        extra = {"prompts": v["report"], "hidden_parts": hidden, "threshold": args.threshold,
                 "part_names": names, "concept_bank": args.concept_bank, "names_source": source}
        written.append(common.write_variant(obj, f"sam3_{tag}", v["paint"], groups, grey, colors, "sam3", tag,
                                            encoders, cond_models, extra=extra, force=args.force))
        bound_prompts = sum(1 for r in v["report"] if r["parts"])
        print(f"  {tag}: {bound_prompts}/{len(v['report'])} prompts bound, groups={groups}, grey={grey}")
    return {"object": obj.path, "n_parts": n_parts, "variants": written}


def joint_groups(per_view: list[dict], n_parts: int, hidden_policy: str, rng):
    """One colour assignment shared by all views of an object.

    Parts a prompt tied together in any view share a colour (union of the per-view groups);
    a part bound in at least one view is coloured in 3D and simply stays grey in the 2D map of
    the views where SAM3 missed it. Parts visible somewhere but bound nowhere are grey; parts
    hidden in every view follow --hidden_policy.
    """
    parent = list(range(n_parts))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    bound: set[int] = set()
    for v in per_view:
        for g in v["groups"]:
            bound.update(g)
            for p in g[1:]:
                parent[find(p)] = find(g[0])
    comps: dict[int, list[int]] = {}
    for p in sorted(bound):
        comps.setdefault(find(p), []).append(p)
    groups = list(comps.values())
    visible_any = set().union(*(set(range(n_parts)) - set(v["hidden"]) for v in per_view))
    grey = sorted(p for p in visible_any if p not in bound)
    hidden_all = sorted(p for p in range(n_parts) if p not in visible_any)
    if hidden_policy == "grey":
        grey = sorted(set(grey) | set(hidden_all))
    else:
        groups = groups + [[p] for p in hidden_all]
    colors = common.random_palette(rng, len(groups)) if groups else []
    return groups, grey, colors


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--objects", nargs="*", default=None)
    parser.add_argument("--objects_file", default=None, help="Text file with one object dir name per line")
    parser.add_argument("--azimuths", default="0")
    parser.add_argument("--prompts", nargs="*", default=None, help="Override names.json prompts (all objects)")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--cover", type=float, default=0.5, help="Mask must cover this fraction of a part to bind it")
    parser.add_argument("--partial_precision", type=float, default=0.5)
    parser.add_argument("--partial_cover", type=float, default=0.15)
    parser.add_argument("--engine", default="CYCLES", help="bpy render engine for the textured view")
    parser.add_argument("--samples", type=int, default=32, help="Cycles samples (inference renders use 128; 32 + denoise is visually the same)")
    parser.add_argument("--py_sam3", default=DEFAULT_PY_SAM3)
    parser.add_argument("--concept_bank", default=None, help="Stage-B bank.pt forwarded to sam3_masks.py")
    parser.add_argument("--hidden_policy", choices=["gt", "grey"], default="gt")
    parser.add_argument("--no_pair", dest="pair", action="store_false",
                        help="Colour every view independently (v1/v2 behaviour) instead of one joint "
                             "assignment shared by all views of the object")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max_parts", type=int, default=64)
    args = parser.parse_args()

    os.chdir(common.ROOT)
    names = common.object_names(args.dataset_root, args.objects, args.objects_file)
    azimuths = [float(a) for a in args.azimuths.split(",") if a.strip()]
    rng = np.random.default_rng(args.seed)
    encoders = common.load_encoders()
    cond_models = common.load_cond_models()

    report = []
    for name in names:
        obj = ObjectDir(os.path.join(args.dataset_root, name))
        if not os.path.exists(obj.input_glb) or not os.path.isdir(obj.parts_dir):
            continue
        n_parts = len(obj.part_files())
        if n_parts > args.max_parts or n_parts < 2:
            print(f"skip {name}: {n_parts} parts")
            continue
        try:
            print(f"== {name}")
            r = process_object(obj, encoders, cond_models, azimuths, rng, args, args.prompts)
            report.append(r)
        except Exception as exc:
            traceback.print_exc()
            report.append({"object": obj.path, "error": repr(exc)})
    with open(os.path.join(args.dataset_root, "make_samples_a_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
