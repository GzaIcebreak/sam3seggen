"""SAM3 (+ concept bank) on GeoSAM2-convention renders -> per-view label maps GeoSAM2 can take as prompts.

    .venv_holo\Scripts\python finetune\geosam2_masks.py --renders E:\...\geosam2\renders --dataset_root E:\...\pv
        --objects_file E:\...\pv_hard.txt --concept_bank E:\...\concept_bank_v3\bank.pt
    .venv_holo\Scripts\python finetune\geosam2_masks.py --renders ... --objects dog --prompts head ear body leg tail

For every render dir <renders>/<obj>/ this writes sam3/
    view_XXXX.npy      int32 HxW label map: 0 = background/unlabelled, k>0 = prompts[k-1]
    view_XXXX.npz      raw union masks, scores per prompt (same layout as sam3_masks.npz)
    summary.json       per view: prompts found, labelled foreground share; the "best" view
Label maps follow Path A's rule (make_samples_a.bind_masks): masks are painted smallest first and a
pixel keeps the first mask that claimed it, so small parts are not swallowed by big ones. The colour
render is composited on black like SegviGen's renders the concept bank was trained on.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
from PIL import Image

from sam3_to_2dmap import load_concept_bank, load_sam3, segment_prompts

N_VIEWS = 12


def unique_prompts(names: list[str]) -> list[str]:
    seen: list[str] = []
    for n in names:
        n = n.strip()
        if n and n not in seen:
            seen.append(n)
    return seen


def load_view(path: str) -> tuple[Image.Image, np.ndarray]:
    im = Image.open(path).convert("RGBA")
    fg = np.asarray(im)[..., 3] > 0
    black = Image.new("RGBA", im.size, (0, 0, 0, 255))
    return Image.alpha_composite(black, im).convert("RGB"), fg


def label_map(masks: np.ndarray, fg: np.ndarray) -> np.ndarray:
    lab = np.zeros(fg.shape, dtype=np.int32)
    order = sorted(range(len(masks)), key=lambda k: int(masks[k].sum()))
    for k in order:
        m = masks[k] & fg & (lab == 0)
        lab[m] = k + 1
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", required=True)
    ap.add_argument("--dataset_root", default=None, help="Read prompts from <root>/<obj>/names.json")
    ap.add_argument("--objects_file", default=None)
    ap.add_argument("--objects", nargs="*", default=None)
    ap.add_argument("--prompts", nargs="*", default=None, help="Prompts for objects without names.json")
    ap.add_argument("--prompts_json", default=None, help='{"<obj>": ["head", ...], ...} per-object prompts')
    ap.add_argument("--model", default="facebook/sam3")
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--concept_bank", default=None)
    ap.add_argument("--no_e0", action="store_true")
    ap.add_argument("--views", type=int, nargs="*", default=None, help="Subset of view ids (default all 12)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    objects = list(args.objects or [])
    if args.objects_file:
        with open(args.objects_file, "r", encoding="utf-8") as f:
            objects += [l.strip() for l in f if l.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = load_sam3(args.model, device)
    bank = load_concept_bank(args.concept_bank, device)
    views = args.views if args.views else list(range(N_VIEWS))
    per_object = {}
    if args.prompts_json:
        with open(args.prompts_json, "r", encoding="utf-8") as f:
            per_object = json.load(f)

    for obj in objects:
        rdir = os.path.join(args.renders, obj)
        out_dir = os.path.join(rdir, "sam3")
        summary_path = os.path.join(out_dir, "summary.json")
        if os.path.exists(summary_path) and not args.force:
            print(f"[skip] {obj}")
            continue
        names_path = os.path.join(args.dataset_root, obj, "names.json") if args.dataset_root else None
        if names_path and os.path.exists(names_path):
            with open(names_path, "r", encoding="utf-8") as f:
                prompts = unique_prompts(json.load(f))
        elif obj in per_object:
            prompts = unique_prompts(per_object[obj])
        elif args.prompts:
            prompts = unique_prompts(args.prompts)
        else:
            print(f"[{obj}] no prompts, skipping")
            continue
        os.makedirs(out_dir, exist_ok=True)
        per_view = []
        for v in views:
            image, fg = load_view(os.path.join(rdir, f"color_{v:04d}.webp"))
            print(f"[{obj}] view {v}: {len(prompts)} prompts", flush=True)
            found = segment_prompts(processor, model, image, prompts, args.threshold, device,
                                    bank=bank, use_e0=not args.no_e0)
            by_prompt = {p["prompt"]: p for p in found}
            h, w = fg.shape
            masks = np.zeros((len(prompts), h, w), dtype=bool)
            scores = np.zeros(len(prompts), dtype=np.float32)
            for i, p in enumerate(prompts):
                if p in by_prompt:
                    masks[i] = by_prompt[p]["mask"]
                    scores[i] = by_prompt[p]["score"]
            lab = label_map(masks, fg)
            np.save(os.path.join(out_dir, f"view_{v:04d}.npy"), lab)
            np.savez_compressed(os.path.join(out_dir, f"view_{v:04d}.npz"), masks=masks, scores=scores,
                                prompts=np.array(prompts, dtype=object))
            present = [prompts[k - 1] for k in np.unique(lab) if k > 0]
            per_view.append({"view": v, "found": present, "n_found": len(present),
                             "labelled_share": float((lab > 0).sum() / max(1, fg.sum())),
                             "fg_pixels": int(fg.sum())})
        # best view: most prompts found, ties by labelled foreground share
        best = max(per_view, key=lambda r: (r["n_found"], r["labelled_share"]))
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"prompts": prompts, "threshold": args.threshold, "concept_bank": args.concept_bank,
                       "views": per_view, "best_view": best["view"]}, f, ensure_ascii=False, indent=2)
        print(f"[{obj}] best view {best['view']}: {best['n_found']}/{len(prompts)} prompts, "
              f"{best['labelled_share']:.2f} of foreground labelled", flush=True)


if __name__ == "__main__":
    main()
