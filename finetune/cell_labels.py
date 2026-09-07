"""Per-object latent-cell labels for the explicit colour loss (v4).

Every texture-latent cell covers a 16^3 block of ids.vxz voxels. This writes
    <obj>/cell_part.npz   coords [N,3] int32  (same order as shape_slat.pth)
                          part   [N]   int16  majority part index inside the block (-1 = none)
                          purity [N]   float16 share of the block's voxels carrying that part
so train.py can turn any variant's (groups, grey_parts, colors) into a per-cell colour class
without touching o_voxel at training time.

    finetune\run_ft.bat cell_labels.py --dataset_root E:\AI_New\ModelGen\datasets\pv
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import common
from common import ObjectDir

FACTOR = 16  # tex_enc_next_dc_f16c32: 16 fine voxels per latent cell


def _key(c: np.ndarray) -> np.ndarray:
    c = c.astype(np.int64)
    return (c[:, 0] << 40) | (c[:, 1] << 20) | c[:, 2]


def cell_labels(fine_coords: np.ndarray, voxel_part: np.ndarray, lat_coords: np.ndarray):
    """Majority part + purity per latent cell; -1 where the block holds no decodable voxel."""
    n_parts = int(voxel_part.max()) + 1 if voxel_part.size else 0
    key_f = _key(fine_coords // FACTOR)
    key_l = _key(lat_coords)
    order = np.argsort(key_f, kind="stable")
    key_f, vp = key_f[order], voxel_part[order]
    starts = np.searchsorted(key_f, key_l, "left")
    ends = np.searchsorted(key_f, key_l, "right")
    part = np.full(len(key_l), -1, dtype=np.int16)
    purity = np.zeros(len(key_l), dtype=np.float16)
    for i, (s, e) in enumerate(zip(starts, ends)):
        if e <= s:
            continue
        block = vp[s:e]
        block = block[block >= 0]
        if block.size == 0:
            continue
        h = np.bincount(block, minlength=n_parts)
        part[i] = h.argmax()
        purity[i] = h.max() / (e - s)
    return part, purity


def process(obj: ObjectDir, force: bool = False) -> dict:
    out = os.path.join(obj.path, "cell_part.npz")
    if os.path.exists(out) and not force:
        return {"object": obj.path, "skipped": True}
    fine, _ = common.read_vxz(obj.ids_vxz)
    fine = np.asarray(fine)
    vp = np.load(obj.voxel_part)
    lat = common.load_slat(obj.shape_slat)["coords"][:, 1:].numpy()
    if fine.max() // FACTOR > lat.max() + 1:
        raise ValueError(f"{obj.path}: fine grid {fine.max() + 1} vs latent grid {lat.max() + 1}: not a factor-{FACTOR} pair")
    part, purity = cell_labels(fine, vp, lat)
    np.savez(out, coords=lat.astype(np.int32), part=part, purity=purity)
    return {"object": obj.path, "n_cells": int(len(part)), "labelled": float((part >= 0).mean()),
            "pure": float(((part >= 0) & (purity >= 0.9)).mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--objects_file", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    names = common.object_names(args.dataset_root, None, args.objects_file)
    done = skipped = failed = 0
    pure = []
    for i, name in enumerate(names):
        obj = ObjectDir(os.path.join(args.dataset_root, name))
        if not (os.path.exists(obj.voxel_part) and os.path.exists(obj.ids_vxz) and os.path.exists(obj.shape_slat)):
            continue
        try:
            r = process(obj, args.force)
        except Exception:
            traceback.print_exc()
            failed += 1
            continue
        if r.get("skipped"):
            skipped += 1
        else:
            done += 1
            pure.append(r["pure"])
        if (done + skipped) % 100 == 0:
            print(f"[{i + 1}/{len(names)}] done={done} skipped={skipped} failed={failed} "
                  f"mean pure-cell share={np.mean(pure) if pure else 0:.3f}", flush=True)
    print(f"finished: done={done} skipped={skipped} failed={failed} mean pure-cell share={np.mean(pure) if pure else 0:.3f}")


if __name__ == "__main__":
    main()
