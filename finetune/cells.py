"""Latent-cell colour targets shared by the v4 colour loss, the probe fit and the evaluation.

A variant's meta (groups, grey_parts, colors) plus the object's cell_part.npz give, per latent
cell, the class it should decode to: index g for colour group g, index G for GREY, -1 to ignore
(no decodable voxel, impure block, or a part that is neither coloured nor grey).
"""
from __future__ import annotations

import os

import numpy as np
import torch

import common

GREY_RGB = torch.tensor(common.GREY, dtype=torch.float32) / 255.0


def load_cells(obj: common.ObjectDir):
    z = np.load(os.path.join(obj.path, "cell_part.npz"))
    return z["coords"], z["part"].astype(np.int64), z["purity"].astype(np.float32)


def part_classes(meta: dict, n_parts: int) -> np.ndarray:
    """part index -> colour class (group index, G for grey, -1 unknown)."""
    G = len(meta["groups"])
    cls = np.full(n_parts, -1, dtype=np.int64)
    for g, members in enumerate(meta["groups"]):
        for p in members:
            if p < n_parts:
                cls[p] = g
    for p in meta.get("grey_parts", []):
        if p < n_parts:
            cls[p] = G
    return cls


def palette_of(meta: dict) -> torch.Tensor:
    """[G+1, 3] colours in 0..1, last row GREY."""
    cols = torch.tensor(meta["colors"], dtype=torch.float32) / 255.0
    return torch.cat([cols, GREY_RGB[None]], 0)


class ColorProbe:
    """Frozen linear map normalised tex latent [N, 32] -> RGB 0..1 [N, 3] (color_probe.py)."""

    def __init__(self, path: str, device: str = "cuda"):
        blob = torch.load(path, map_location="cpu")
        self.weight = blob["weight"].to(device)
        self.bias = blob["bias"].to(device)
        self.info = {k: blob[k] for k in ("held_out", "fit", "n_objects") if k in blob}

    def __call__(self, latent: torch.Tensor) -> torch.Tensor:
        return latent.float() @ self.weight + self.bias


def palette_ce(rgb: torch.Tensor, cls: torch.Tensor, palette: torch.Tensor, tau: float):
    """Class-balanced cross-entropy of nearest-palette assignment for ONE sample.

    logits_j = -|rgb - palette_j|^2 / tau over the palette rows (colour groups + GREY); every colour
    class present contributes equally, so thin parts weigh as much as the body they are attached to.
    Returns (loss, accuracy, n_cells) or None when the sample has no labelled cell.
    """
    m = cls >= 0
    if not bool(m.any()):
        return None
    r, c = rgb[m], cls[m]
    logits = -torch.cdist(r, palette) ** 2 / tau
    ce = torch.nn.functional.cross_entropy(logits, c, reduction="none")
    counts = torch.bincount(c, minlength=palette.shape[0]).float()
    w = 1.0 / counts[c]
    w = w / w.sum()
    acc = (logits.argmax(1) == c).float().mean()
    return (w * ce).sum(), acc, int(m.sum())


@torch.no_grad()
def nearest_accuracy(rgb: torch.Tensor, cls: torch.Tensor, palette: torch.Tensor, subset: torch.Tensor):
    """Nearest-palette accuracy on `subset` cells with a label, or None if there are none."""
    m = subset & (cls >= 0)
    if not bool(m.any()):
        return None
    return (torch.cdist(rgb[m], palette).argmin(1) == cls[m]).float().mean()


def variant_cell_targets(obj: common.ObjectDir, meta: dict, min_purity: float = 0.9):
    """(class per cell [N] int64 with -1 ignore, palette [G+1, 3]) in cell_part.npz order."""
    _, part, purity = load_cells(obj)
    n_parts = int(part.max()) + 1 if part.size else 0
    cls_of_part = part_classes(meta, max(n_parts, 1))
    cls = np.where(part >= 0, cls_of_part[np.clip(part, 0, None)], -1)
    cls = np.where(purity >= min_purity, cls, -1)
    return cls, palette_of(meta)
