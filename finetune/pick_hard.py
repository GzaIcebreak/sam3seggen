"""Pick the hard-sample evaluation objects (Stage E): >= --min_parts parts and at least one thin
part attached to a big part, measured on voxel_part.npy (6-neighbour contacts between labels).

    finetune\run_ft.bat pick_hard.py --dataset_root E:\data\pv --n 20 --exclude E:\data\pv_holdout_mix.txt \
        --out E:\data\pv_hard.txt

"thin": bbox aspect (longest / shortest extent) >= --aspect or <= --small share of the object's
voxels; "big": >= --big share. Score = contact voxels / thin-part voxels (how much of the thin
part is glued to something large), summed over such pairs; ties favour more parts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import common


def contacts(coords: np.ndarray, labels: np.ndarray, n_parts: int) -> np.ndarray:
    """[n_parts, n_parts] count of 6-neighbour voxel pairs with different labels."""
    key = (coords[:, 0].astype(np.int64) * 512 + coords[:, 1]) * 512 + coords[:, 2]
    order = np.argsort(key)
    skey, slab = key[order], labels[order]
    C = np.zeros((n_parts, n_parts), dtype=np.int64)
    for d in (1, 512, 512 * 512):
        nk = skey + d
        pos = np.searchsorted(skey, nk)
        pos = np.clip(pos, 0, len(skey) - 1)
        hit = skey[pos] == nk
        a, b = slab[hit], slab[pos[hit]]
        diff = a != b
        np.add.at(C, (a[diff], b[diff]), 1)
    return C + C.T


def score_object(obj: common.ObjectDir, aspect: float, small: float, big: float) -> dict | None:
    if not os.path.exists(obj.voxel_part) or not os.path.exists(obj.ids_vxz):
        return None
    labels = np.load(obj.voxel_part).astype(np.int64)
    coords, _ = common.read_vxz(obj.ids_vxz)
    coords = coords.numpy() if torch.is_tensor(coords) else np.asarray(coords)
    valid = labels >= 0                      # negative = voxel not assigned to any part
    labels, coords = labels[valid], coords[valid]
    if len(labels) == 0:
        return None
    n_parts = int(labels.max()) + 1
    total = len(labels)
    counts = np.bincount(labels, minlength=n_parts)
    ext = np.zeros((n_parts, 3))
    for p in range(n_parts):
        sel = labels == p
        if sel.any():
            c = coords[sel]
            ext[p] = c.max(0) - c.min(0) + 1
    asp = ext.max(1) / np.maximum(ext.min(1), 1)
    thin = [p for p in range(n_parts) if counts[p] > 0 and (asp[p] >= aspect or counts[p] / total <= small)]
    bigs = [p for p in range(n_parts) if counts[p] / total >= big]
    if not thin or not bigs:
        return {"score": 0.0, "pairs": [], "n_parts": n_parts}
    C = contacts(coords, labels, n_parts)
    pairs, score = [], 0.0
    for p in thin:
        for q in bigs:
            if p == q or C[p, q] < 30:
                continue
            s = float(C[p, q] / counts[p])
            pairs.append({"thin": p, "big": q, "contact": int(C[p, q]), "thin_share": float(counts[p] / total),
                          "aspect": float(asp[p]), "glue": s})
            score += s
    return {"score": score, "pairs": pairs, "n_parts": n_parts}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--min_parts", type=int, default=6)
    ap.add_argument("--aspect", type=float, default=4.0)
    ap.add_argument("--small", type=float, default=0.03)
    ap.add_argument("--big", type=float, default=0.2)
    ap.add_argument("--exclude", nargs="*", default=[], help="object list files to skip (e.g. the holdout)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    skip = set()
    for f in args.exclude:
        with open(f, encoding="utf-8") as fh:
            skip |= {l.strip() for l in fh if l.strip() and not l.startswith("#")}
    names = [d for d in sorted(os.listdir(args.dataset_root))
             if d not in skip and common.ObjectDir(os.path.join(args.dataset_root, d)).is_prepared()]
    rows = []
    for i, name in enumerate(names):
        obj = common.ObjectDir(os.path.join(args.dataset_root, name))
        if len(obj.part_files()) < args.min_parts:
            continue
        r = score_object(obj, args.aspect, args.small, args.big)
        if r and r["score"] > 0:
            r["object"] = name
            r["names"] = obj.names()
            rows.append(r)
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(names)} scanned, {len(rows)} candidates", flush=True)
    rows.sort(key=lambda r: (-r["score"], -r["n_parts"]))
    pick = rows[:args.n]
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(r["object"] for r in pick) + "\n")
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(pick, f, ensure_ascii=False, indent=1)
    for r in pick:
        top = max(r["pairs"], key=lambda p: p["glue"])
        nm = r["names"] or []
        print(f"{r['object'][:8]} parts={r['n_parts']:2d} score={r['score']:.2f} "
              f"thin '{nm[top['thin']] if top['thin'] < len(nm) else top['thin']}' on "
              f"'{nm[top['big']] if top['big'] < len(nm) else top['big']}' contact={top['contact']} aspect={top['aspect']:.1f}")


if __name__ == "__main__":
    main()
