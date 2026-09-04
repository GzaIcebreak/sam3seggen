"""Path B: clean + synthetically corrupted 2D-map samples for every object in a dataset root.

Run with the SegviGen venv from the SegviGen root:
    python finetune/make_samples_b.py --dataset_root <root> --azimuths 0,135 --n_corrupt 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np

import common
from common import ObjectDir, LABEL_BG
from corrupt import corrupt, visible_parts


def view_tag(azimuth: float) -> str:
    return f"az{azimuth:g}"


def ensure_ids_raster(obj: ObjectDir, parts, aabb, azimuth: float, force: bool = False) -> np.ndarray:
    vdir = obj.view_dir(view_tag(azimuth))
    os.makedirs(vdir, exist_ok=True)
    path = os.path.join(vdir, "ids.npy")
    if os.path.exists(path) and not force:
        return np.load(path)
    labels = common.render_part_ids(parts, aabb, azimuth)
    np.save(path, labels)
    common.paint_map(labels, {p: c for p, c in enumerate(common.id_colors(len(parts)))}).save(
        os.path.join(vdir, "ids_preview.png"))
    return labels


def process_object(obj: ObjectDir, encoders, cond_models, azimuths, n_corrupt: int, rng, force: bool,
                   hidden_policy: str) -> dict:
    meta = common.prepare_object(obj, encoders, force=force)
    parts = common.load_parts(obj)
    aabb = np.asarray(meta["aabb"])
    n_parts = meta["n_parts"]
    written = []
    # One palette per object: the clean views of an object form a pair with a shared 3D target
    # (hidden_policy "gt" keeps every part coloured, so the target is view-independent).
    shared_colors = common.random_palette(rng, n_parts)
    tags = [view_tag(az) for az in azimuths]
    first_clean = None
    for az in azimuths:
        labels = ensure_ids_raster(obj, parts, aabb, az, force=force)
        tag = view_tag(az)
        visible = visible_parts(labels)
        if len(visible) < 1:
            continue
        hidden = [p for p in range(n_parts) if p not in visible]

        colors = shared_colors
        groups = [[p] for p in range(n_parts)]
        grey = hidden if hidden_policy == "grey" else []
        groups_c = [g for g in groups if g[0] not in grey]
        colors_c = [colors[g[0]] for g in groups_c]
        paired = hidden_policy == "gt" and len(azimuths) > 1
        extra = {"ops": [], "hidden_parts": hidden}
        if paired:
            extra.update({"pair": "clean", "pair_views": tags})
        vname = f"clean_{tag}"
        written.append(common.write_variant(
            obj, vname, labels, groups_c, grey, colors_c, "clean", tag,
            encoders, cond_models, extra=extra, force=force, target_from=first_clean if paired else None))
        first_clean = first_clean or vname

        for k in range(n_corrupt):
            c_labels, c_groups, c_grey, ops = corrupt(labels, n_parts, rng)
            if hidden_policy == "grey":
                c_grey = sorted(set(c_grey) | set(hidden))
                c_groups = [[p for p in g if p not in hidden] for g in c_groups]
                c_groups = [g for g in c_groups if g]
            colors = common.random_palette(rng, len(c_groups))
            written.append(common.write_variant(
                obj, f"corrupt_{tag}_{k}", c_labels, c_groups, c_grey, colors, "corrupt", tag,
                encoders, cond_models, extra={"ops": ops, "hidden_parts": hidden}, force=force))
    return {"object": obj.path, "n_parts": n_parts, "variants": written}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--objects", nargs="*", default=None, help="Subset of object dir names")
    parser.add_argument("--objects_file", default=None, help="Text file with one object dir name per line")
    parser.add_argument("--azimuths", default="0", help="Comma-separated orbit angles around the calibrated camera")
    parser.add_argument("--n_corrupt", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--hidden_policy", choices=["gt", "grey"], default="gt",
                        help="3D colour of parts invisible from the view: their own colour (paper behaviour) or GREY")
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
            print(f"skip {name}: not an object dir")
            continue
        n_parts = len(obj.part_files())
        if n_parts > args.max_parts or n_parts < 2:
            print(f"skip {name}: {n_parts} parts")
            continue
        try:
            r = process_object(obj, encoders, cond_models, azimuths, args.n_corrupt, rng, args.force, args.hidden_policy)
            print(f"ok {name}: {r['n_parts']} parts, {len(r['variants'])} variants")
            report.append(r)
        except Exception as exc:  # keep the batch going; the report says what failed
            traceback.print_exc()
            report.append({"object": obj.path, "error": repr(exc)})
    with open(os.path.join(args.dataset_root, "make_samples_b_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
