"""Synthetic SAM3-style corruption of a clean part-index raster (path B).

Every op works on an int16 label raster: part index >= 0, LABEL_BG, LABEL_GREY.
Ops that also change the 3D target return that information explicitly
(dropped parts -> grey in 3D, merged parts -> one colour in 3D); the rest are 2D only,
which is exactly what teaches the model to snap to geometry and fill in what SAM3 missed.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from common import LABEL_BG, LABEL_GREY


def visible_parts(labels: np.ndarray) -> list[int]:
    vals = np.unique(labels)
    return [int(v) for v in vals if v >= 0]


def _disk(r: int) -> np.ndarray:
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    return (yy * yy + xx * xx) <= r * r


def _smooth_noise(rng: np.random.Generator, shape, sigma: float) -> np.ndarray:
    n = ndimage.gaussian_filter(rng.random(shape), sigma)
    n -= n.min()
    return n / (n.max() + 1e-9)


def jitter_boundaries(labels: np.ndarray, rng: np.random.Generator, max_radius: int = 6, frac: float = 0.6) -> np.ndarray:
    """Let a random subset of parts bleed raggedly into their neighbours (SAM3 edge slop)."""
    out = labels.copy()
    parts = visible_parts(labels)
    rng.shuffle(parts)
    for p in parts[:max(1, int(round(len(parts) * frac)))]:
        r = int(rng.integers(1, max_radius + 1))
        grown = ndimage.binary_dilation(labels == p, structure=_disk(r))
        take = grown & (out >= 0) & (out != p)
        noise = _smooth_noise(rng, labels.shape, sigma=float(rng.uniform(2, 6)))
        take &= noise > rng.uniform(0.3, 0.6)
        out[take] = p
    return out


def drop_parts(labels: np.ndarray, rng: np.random.Generator, p_drop: float = 0.25,
               small_frac: float = 0.02, keep_min: int = 1) -> tuple[np.ndarray, list[int]]:
    """Whole parts SAM3 never detected. Small parts are twice as likely to vanish."""
    parts = visible_parts(labels)
    fg = float((labels >= 0).sum())
    drop = []
    for p in parts:
        prob = p_drop * (2.0 if (labels == p).sum() < small_frac * fg else 1.0)
        if rng.random() < prob:
            drop.append(p)
    if len(parts) - len(drop) < keep_min:
        drop = drop[:max(0, len(parts) - keep_min)]
    out = labels.copy()
    for p in drop:
        out[labels == p] = LABEL_GREY
    return out, drop


def grey_holes(labels: np.ndarray, rng: np.random.Generator, n_range=(1, 4), radius=(6, 28)) -> np.ndarray:
    out = labels.copy()
    fg = np.argwhere(out >= 0)
    if len(fg) == 0:
        return out
    h, w = labels.shape
    yy, xx = np.ogrid[:h, :w]
    for _ in range(int(rng.integers(n_range[0], n_range[1] + 1))):
        cy, cx = fg[rng.integers(len(fg))]
        ry, rx = rng.integers(radius[0], radius[1] + 1, size=2)
        ell = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
        out[ell & (out >= 0)] = LABEL_GREY
    return out


def partial_erase(labels: np.ndarray, rng: np.random.Generator, min_pixels: int = 400) -> tuple[np.ndarray, int | None]:
    """SAM3 saw only one side of a part: erase one half-plane of it (2D only)."""
    candidates = [p for p in visible_parts(labels) if (labels == p).sum() >= min_pixels]
    if not candidates:
        return labels.copy(), None
    p = int(rng.choice(candidates))
    ys, xs = np.nonzero(labels == p)
    cy, cx = ys.mean(), xs.mean()
    theta = rng.uniform(0, np.pi)
    side = (ys - cy) * np.cos(theta) + (xs - cx) * np.sin(theta) > rng.uniform(-0.2, 0.2) * max(np.ptp(ys), np.ptp(xs))
    out = labels.copy()
    out[ys[side], xs[side]] = LABEL_GREY
    return out, p


def speckle(labels: np.ndarray, rng: np.random.Generator, n_range=(3, 12), radius=(2, 6)) -> np.ndarray:
    """Small wrong-label blobs along boundaries."""
    out = labels.copy()
    fg = out >= 0
    edge = fg & (ndimage.maximum_filter(out, size=3) != ndimage.minimum_filter(out, size=3))
    pts = np.argwhere(edge)
    if len(pts) == 0:
        return out
    h, w = labels.shape
    yy, xx = np.ogrid[:h, :w]
    for _ in range(int(rng.integers(n_range[0], n_range[1] + 1))):
        cy, cx = pts[rng.integers(len(pts))]
        r = int(rng.integers(radius[0], radius[1] + 1))
        region = ((yy - cy) ** 2 + (xx - cx) ** 2 <= r * r) & (out >= 0)
        neighbours = [int(v) for v in np.unique(out[region]) if v != out[cy, cx]]
        if not neighbours:
            continue
        out[region] = int(rng.choice(neighbours))
    return out


def adjacent_pairs(labels: np.ndarray) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for axis_shift in ((0, 1), (1, 0)):
        a = labels[:labels.shape[0] - axis_shift[0], :labels.shape[1] - axis_shift[1]]
        b = labels[axis_shift[0]:, axis_shift[1]:]
        touch = (a >= 0) & (b >= 0) & (a != b)
        for u, v in zip(a[touch].tolist(), b[touch].tolist()):
            pairs.add((min(u, v), max(u, v)))
    return pairs


def merge_neighbours(labels: np.ndarray, n_parts: int, rng: np.random.Generator, n_merges: int = 1) -> list[list[int]]:
    """Union adjacent parts into colour groups (changes 2D and 3D alike: 'one prompt, two parts')."""
    parent = list(range(n_parts))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = list(adjacent_pairs(labels))
    rng.shuffle(pairs)
    merged = 0
    for u, v in pairs:
        if merged >= n_merges:
            break
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
            merged += 1
    groups: dict[int, list[int]] = {}
    for p in range(n_parts):
        groups.setdefault(find(p), []).append(p)
    return [sorted(g) for g in groups.values()]


def corrupt(labels: np.ndarray, n_parts: int, rng: np.random.Generator,
            probs: dict | None = None) -> tuple[np.ndarray, list[list[int]], list[int], list[str]]:
    """Compose a random subset of ops. Returns (labels, groups, grey_parts, applied ops)."""
    probs = {
        "jitter": 0.9, "drop": 0.5, "holes": 0.5, "merge": 0.35, "partial": 0.4, "speckle": 0.6,
        **(probs or {}),
    }
    ops: list[str] = []
    out = labels.copy()
    groups = [[p] for p in range(n_parts)]
    grey_parts: list[int] = []

    if rng.random() < probs["merge"] and len(visible_parts(out)) >= 2:
        groups = merge_neighbours(out, n_parts, rng, n_merges=int(rng.integers(1, 3)))
        if any(len(g) > 1 for g in groups):
            ops.append("merge")
    if rng.random() < probs["jitter"]:
        out = jitter_boundaries(out, rng)
        ops.append("jitter")
    if rng.random() < probs["speckle"]:
        out = speckle(out, rng)
        ops.append("speckle")
    if rng.random() < probs["drop"]:
        out, dropped = drop_parts(out, rng)
        if dropped:
            grey_parts = dropped
            ops.append("drop")
    if rng.random() < probs["partial"]:
        out, p = partial_erase(out, rng)
        if p is not None:
            ops.append("partial")
    if rng.random() < probs["holes"]:
        out = grey_holes(out, rng)
        ops.append("holes")

    # A greyed part must not keep a colour through its merge group.
    if grey_parts:
        grey_set = set(grey_parts)
        groups = [[p for p in g if p not in grey_set] for g in groups]
        groups = [g for g in groups if g]
    return out, groups, grey_parts, ops
