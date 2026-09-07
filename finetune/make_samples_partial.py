"""v4 "partial" variants: the clean 2D map with 1-3 visible parts painted GREY, the clean 3D target
and the full legend. The greyed parts' colour is knowable from the legend only, so the colour loss
on their cells can be lowered by reading the legend and by nothing else.

Reuses the clean variant's output_tex_slat.pth (target_from), so only the map + DINO cond are new.
    finetune\run_ft.bat make_samples_partial.py --dataset_root E:\AI_New\ModelGen\datasets\pv --per_view 2
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


def clean_variants(obj: ObjectDir) -> list[tuple[str, dict]]:
    out = []
    if not os.path.isdir(obj.variants_dir):
        return out
    for vname in sorted(os.listdir(obj.variants_dir)):
        meta_path = os.path.join(obj.variant_dir(vname), "meta.json")
        if not vname.startswith("clean_") or not os.path.exists(meta_path):
            continue
        if not os.path.exists(os.path.join(obj.variant_dir(vname), "output_tex_slat.pth")):
            continue
        with open(meta_path, encoding="utf-8") as f:
            out.append((vname, json.load(f)))
    return out


def process_object(obj: ObjectDir, cond_models, per_view: int, rng: np.random.Generator, force: bool,
                   min_share: float, max_mask: int) -> list[str]:
    written = []
    for vname, meta in clean_variants(obj):
        tag = meta["view"]
        ids_path = os.path.join(obj.view_dir(tag), "ids.npy")
        if not os.path.exists(ids_path) or meta.get("grey_parts"):
            continue
        labels = np.load(ids_path)
        fg = int((labels != LABEL_BG).sum())
        vals, counts = np.unique(labels[labels >= 0], return_counts=True)
        # only parts big enough in this view for greying them to change the map
        candidates = [int(v) for v, c in zip(vals, counts) if c >= min_share * fg]
        if len(candidates) < 2:
            continue
        used: set[tuple[int, ...]] = set()
        for k in range(per_view):
            n_mask = int(rng.integers(1, min(max_mask, len(candidates) - 1) + 1))
            for _ in range(20):
                masked = tuple(sorted(int(p) for p in rng.choice(candidates, size=n_mask, replace=False)))
                if masked not in used:
                    break
            used.add(masked)
            colors = [tuple(c) for c in meta["colors"]]
            written.append(common.write_variant(
                obj, f"partial_{tag}_{k}", labels, meta["groups"], [], colors, "partial", tag,
                None, cond_models, extra={"hidden_parts": meta.get("hidden_parts", []), "source": vname},
                force=force, target_from=vname, mask_2d=list(masked)))
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--objects_file", default=None)
    parser.add_argument("--per_view", type=int, default=2)
    parser.add_argument("--max_mask", type=int, default=3)
    parser.add_argument("--min_share", type=float, default=0.005, help="Min share of foreground pixels for a maskable part")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    os.chdir(common.ROOT)
    names = common.object_names(args.dataset_root, None, args.objects_file)
    rng = np.random.default_rng(args.seed)
    cond_models = common.load_cond_models()
    n_ok = n_var = n_fail = 0
    for i, name in enumerate(names):
        obj = ObjectDir(os.path.join(args.dataset_root, name))
        if not obj.is_prepared():
            continue
        try:
            written = process_object(obj, cond_models, args.per_view, rng, args.force, args.min_share, args.max_mask)
        except Exception:
            traceback.print_exc()
            n_fail += 1
            continue
        n_ok += 1
        n_var += len(written)
        if n_ok % 50 == 0:
            print(f"[{i + 1}/{len(names)}] objects={n_ok} variants={n_var} failed={n_fail}", flush=True)
    print(f"finished: objects={n_ok} variants={n_var} failed={n_fail}")


if __name__ == "__main__":
    main()
