"""v5: per-DINO-patch part names for every variant -> <variant>/tokens.npz.

The DiT conditions on 1029 DINOv3 tokens (CLS, 4 registers, 32x32 patches of the 512x512 crop
that data_toolkit/img_to_cond.py cuts out of map.png). v5 adds the text vector of the part under
each patch to that patch's token, so the name is delivered where the colour is (or, for partial
variants, where the colour is missing). This script reproduces the crop geometry and votes a
name per patch:

  * clean / corrupt / partial: the exact GT raster views/<view>/ids.npy -> names.json
  * sam3: the SAM3 union mask of the prompt that covered the pixel (highest score wins), i.e.
    exactly what a deployment has, since there the names come from SAM3 as well

tokens.npz = {"name_idx": int16[1024] (-1 = no name), "names": str[K], "bbox": int[4]}.

    finetune\run_ft.bat token_labels.py --dataset_root E:\...\pv [--force]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image

import common

GRID = 32          # 512 / patch 16
N_PREFIX = 5       # CLS + 4 register tokens precede the patches in the DINOv3 output
MIN_FG = 0.2       # a patch with less foreground than this carries no name


def crop_box(alpha_mask: np.ndarray) -> tuple[int, int, int, int]:
    """Same arithmetic as img_to_cond.preprocess_image (bbox of alpha > 0.8, square around the
    centre, PIL's round-to-int crop). For a synthetic map the rembg alpha is the non-white mask."""
    ys, xs = np.where(alpha_mask)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    size = int(max(x1 - x0, y1 - y0))
    box = (cx - size // 2, cy - size // 2, cx + size // 2, cy + size // 2)
    return tuple(int(round(v)) for v in box)  # type: ignore[return-value]


def name_raster(obj: common.ObjectDir, meta: dict, ids: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """int16 raster of indices into the returned name list (-1 = none)."""
    names: list[str] = []
    index: dict[str, int] = {}

    def idx(name: str | None) -> int:
        name = (name or "").strip()
        if not name:
            return -1
        if name not in index:
            index[name] = len(names)
            names.append(name)
        return index[name]

    out = np.full(ids.shape, -1, dtype=np.int16)
    if meta["kind"] == "sam3":
        npz = np.load(os.path.join(obj.view_dir(meta["view"]), "sam3_masks.npz"), allow_pickle=True)
        prompts = [str(p) for p in npz["prompts"]]
        bound = {p["prompt"] for p in meta.get("prompts", []) if p.get("mode") != "unbound" and p.get("pixels", 0) > 0}
        order = np.argsort(npz["scores"])  # ascending, so the highest score paints last
        for k in order:
            if prompts[k] not in bound:
                continue
            m = npz["masks"][k] & (ids >= 0)
            out[m] = idx(prompts[k])
    else:
        part_names = obj.names() or []
        lut = np.array([idx(part_names[p] if p < len(part_names) else None) for p in range(max(len(part_names), int(ids.max()) + 1))],
                       dtype=np.int16) if ids.max() >= 0 else np.zeros(0, dtype=np.int16)
        fg = ids >= 0
        out[fg] = lut[ids[fg]]
    return out, names


def patch_vote(raster: np.ndarray, box: tuple[int, int, int, int], n_names: int) -> np.ndarray:
    x0, y0, x1, y1 = box
    cw, ch = x1 - x0, y1 - y0
    H, W = raster.shape
    # crop with -1 padding where the square box leaves the image
    crop = np.full((ch, cw), -1, dtype=np.int16)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = raster[sy0:sy1, sx0:sx1]
    yy, xx = np.mgrid[0:ch, 0:cw]
    patch = (yy * GRID // ch) * GRID + (xx * GRID // cw)
    cell_px = np.bincount(patch.ravel(), minlength=GRID * GRID).astype(np.float64)
    named = crop >= 0
    counts = np.zeros((GRID * GRID, max(n_names, 1)), dtype=np.int64)
    np.add.at(counts, (patch[named], crop[named].astype(np.int64)), 1)
    fg_frac = counts.sum(1) / np.maximum(cell_px, 1)
    out = counts.argmax(1).astype(np.int16)
    out[fg_frac < MIN_FG] = -1
    return out


def process(obj: common.ObjectDir, vname: str, meta: dict, force: bool) -> bool:
    vdir = obj.variant_dir(vname)
    dst = os.path.join(vdir, "tokens.npz")
    if os.path.exists(dst) and not force:
        return False
    ids = np.load(os.path.join(obj.view_dir(meta["view"]), "ids.npy"))
    img = np.asarray(Image.open(os.path.join(vdir, "map.png")).convert("RGB"))
    if img.shape[:2] != ids.shape:
        raise RuntimeError(f"{vdir}: map {img.shape[:2]} vs ids {ids.shape}")
    box = crop_box(np.any(img != 255, axis=2))
    raster, names = name_raster(obj, meta, ids)
    name_idx = patch_vote(raster, box, len(names))
    np.savez(dst, name_idx=name_idx, names=np.array(names, dtype=object), bbox=np.array(box, dtype=np.int32))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--objects_file", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    keep = set(common.object_names(args.dataset_root, None, args.objects_file)) if args.objects_file else None
    t0 = time.time()
    done = skipped = failed = 0
    named_patches = total_patches = 0
    for i, (obj, vname, meta) in enumerate(common.list_variants(args.dataset_root)):
        if keep is not None and os.path.basename(obj.path) not in keep:
            continue
        try:
            if process(obj, vname, meta, args.force):
                done += 1
                blob = np.load(os.path.join(obj.variant_dir(vname), "tokens.npz"), allow_pickle=True)
                named_patches += int((blob["name_idx"] >= 0).sum())
                total_patches += blob["name_idx"].size
            else:
                skipped += 1
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAILED {obj.path}/{vname}: {e}", flush=True)
        if (i + 1) % 1000 == 0:
            print(f"{i + 1} variants, {time.time() - t0:.0f}s", flush=True)
    print(f"done={done} skipped={skipped} failed={failed} in {time.time() - t0:.0f}s; "
          f"named patches {named_patches}/{total_patches} ({100 * named_patches / max(1, total_patches):.1f}%)")


if __name__ == "__main__":
    main()
