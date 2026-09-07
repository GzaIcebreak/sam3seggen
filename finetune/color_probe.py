"""Frozen linear probe tex-latent (normalised, 32-d) -> base colour (RGB in 0..1), for the v4 colour loss.

Fit by ridge regression on label-pure latent cells of the training objects (holdout excluded),
reported on held-out objects, then refit on everything and saved:
    finetune\run_ft.bat color_probe.py --dataset_root E:\AI_New\ModelGen\datasets\pv ^
        --holdout_file E:\AI_New\ModelGen\datasets\pv_holdout_v3.txt
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import common
from cells import variant_cell_targets

DEFAULT_OUT = os.path.join(common.ROOT, "finetune", "color_probe.pt")


def fit_ridge(X: np.ndarray, Y: np.ndarray, lam: float = 1e-2):
    Xb = np.concatenate([X, np.ones((len(X), 1), dtype=X.dtype)], 1)
    W = np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ Y)
    return W[:-1], W[-1]


def evaluate(W, b, per_object):
    r2_num = r2_den = 0.0
    accs = []
    for X, Y, palette in per_object:
        P = X @ W + b
        r2_num += ((P - Y) ** 2).sum()
        r2_den += ((Y - Y.mean(0)) ** 2).sum()
        d = ((P[:, None] - palette[None]) ** 2).sum(-1)
        accs.append((palette[d.argmin(1)] == Y).all(1).mean())
    return {"r2": float(1 - r2_num / r2_den), "palette_acc": float(np.mean(accs))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_root", nargs="+", required=True)
    parser.add_argument("--holdout_file", default=None)
    parser.add_argument("--n_objects", type=int, default=400)
    parser.add_argument("--min_purity", type=float, default=0.9)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    holdout = set()
    if args.holdout_file:
        with open(args.holdout_file, encoding="utf-8") as f:
            holdout = {l.strip() for l in f if l.strip()}
    norm = common.load_normalization()
    by_obj: dict[str, tuple] = {}
    for root in args.dataset_root:
        for obj, vname, meta in common.list_variants(root):
            name = os.path.basename(obj.path)
            if name in holdout or name in by_obj or not os.path.exists(os.path.join(obj.path, "cell_part.npz")):
                continue
            by_obj[name] = (obj, vname, meta)
    names = sorted(by_obj)
    random.Random(args.seed).shuffle(names)
    names = names[:args.n_objects]
    print(f"{len(names)} objects (holdout excluded: {len(holdout)})")

    per_object = []
    for name in names:
        obj, vname, meta = by_obj[name]
        cls, palette = variant_cell_targets(obj, meta, args.min_purity)
        keep = cls >= 0
        if keep.sum() < 50:
            continue
        slat = common.load_slat(os.path.join(obj.variant_dir(vname), "output_tex_slat.pth"))
        feats = ((slat["feats"] - norm["tex_mean"]) / norm["tex_std"]).numpy().astype(np.float64)
        pal = palette.numpy().astype(np.float64)
        per_object.append((feats[keep], pal[cls[keep]], pal))
    n_test = max(1, len(per_object) // 5)
    train, test = per_object[n_test:], per_object[:n_test]
    W, b = fit_ridge(np.concatenate([x for x, _, _ in train]), np.concatenate([y for _, y, _ in train]))
    held = evaluate(W, b, test)
    print(f"held-out {len(test)} objects: R2={held['r2']:.3f} palette_acc={held['palette_acc']:.3f}")
    W, b = fit_ridge(np.concatenate([x for x, _, _ in per_object]), np.concatenate([y for _, y, _ in per_object]))
    fit = evaluate(W, b, per_object)
    print(f"refit on all {len(per_object)}: R2={fit['r2']:.3f} palette_acc={fit['palette_acc']:.3f}")
    torch.save({"weight": torch.tensor(W, dtype=torch.float32), "bias": torch.tensor(b, dtype=torch.float32),
                "input": "tex latent normalised by pipeline.json tex_slat_normalization", "output": "rgb 0..1",
                "held_out": held, "fit": fit, "n_objects": len(per_object), "min_purity": args.min_purity},
               args.out)
    print(f"saved {args.out}")
    print(json.dumps({"held_out": held, "fit": fit}))


if __name__ == "__main__":
    main()
