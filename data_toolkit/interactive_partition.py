"""Turn SegviGen's overlapping interactive masks into a partition of the original mesh.

Interactive mode answers "which part contains this point", one point at a time, and the
answers do not add up to a segmentation: on monk, twelve queries came back claiming
between 4% and 59% of the voxels each, 85% of claimed voxels were claimed by more than one
mask, and the confidence is all but binary (its median is 0.001 and its upper deciles are
1.0), so there is nothing to arbitrate with. Deciding a contested voxel by argmax is
therefore close to arbitrary, and that -- not any weakness in the boundaries themselves --
is what puts salt and pepper along every contact.

P3-SAM drives its own promptable model the same way and gets a clean partition out of it,
because it wraps the masks in a selection stage. This is that missing stage:

  * a mask is kept only if it is stable, i.e. barely changes when the threshold moves,
    which is what separates a real part from a boundary that the sampler was unsure of;
  * duplicates are collapsed, because seeds landing in the same part return the very same
    mask (five of the twelve did, at IoU 1.00);
  * the survivors are laid down smallest first, so a fine part nested inside a coarse one
    keeps its own label instead of being swallowed by it.

Labels then move to the original mesh by looking up the voxel under each face centroid,
which is what keeps the source topology, UVs and albedo -- SegviGen's own remesh is never
used.
"""
import argparse
import json
import os
import sys

import numpy as np
import trimesh

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_toolkit.interactive_mask import tidy_labels
from data_toolkit.part_seed_points import GRID, normalize_like_pipeline


def stability(confidence, low, high):
    """How much a mask survives moving the threshold, as in SAM's stability score.

    A mask whose extent swings with the threshold is one the sampler was unsure of, and
    those are exactly the ones that overlap their neighbours; a real part is insensitive.
    """
    wide, narrow = confidence > low, confidence > high
    inter = (wide & narrow).sum(axis=0).astype(np.float64)
    union = (wide | narrow).sum(axis=0).astype(np.float64)
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def collapse_duplicates(masks, scores, iou_threshold):
    """Keep one representative per group of near-identical masks, the most stable one."""
    sizes = masks.sum(axis=0).astype(np.float64)
    order = np.argsort(-scores)
    kept, dropped = [], {}
    for index in order:
        for other in kept:
            inter = float((masks[:, index] & masks[:, other]).sum())
            union = sizes[index] + sizes[other] - inter
            if union > 0 and inter / union > iou_threshold:
                dropped.setdefault(other, []).append(index)
                break
        else:
            kept.append(index)
    return sorted(kept), dropped


def partition(masks, chosen, min_voxels):
    """Lay the masks down smallest first; a voxel belongs to the finest mask claiming it."""
    labels = np.full(masks.shape[0], -1, dtype=np.int64)
    order = sorted(chosen, key=lambda i: int(masks[:, i].sum()))
    assigned = []
    for index in order:
        free = masks[:, index] & (labels < 0)
        if int(free.sum()) < min_voxels:
            continue
        labels[free] = len(assigned)
        assigned.append(index)
    return labels, assigned


def voxel_labels_to_faces(mesh, coords, voxel_labels, grid=GRID):
    """Per-face labels by looking up the voxel under each face centroid.

    The volume is indexed in the input glb's own normalised coordinates, so no
    correspondence with SegviGen's remesh is needed and the source topology survives.
    """
    keys = (coords[:, 0].astype(np.int64) * grid + coords[:, 1]) * grid + coords[:, 2]
    order = np.argsort(keys)
    keys, voxel_labels = keys[order], voxel_labels[order]

    vertices = normalize_like_pipeline(np.asarray(mesh.vertices, dtype=np.float64))
    centroids = vertices[np.asarray(mesh.faces)].mean(axis=1)
    voxels = np.clip(np.floor((centroids + 0.5) * grid).astype(np.int64), 0, grid - 1)
    probe = (voxels[:, 0] * grid + voxels[:, 1]) * grid + voxels[:, 2]

    position = np.clip(np.searchsorted(keys, probe), 0, len(keys) - 1)
    found = keys[position] == probe
    labels = np.full(len(centroids), -1, dtype=np.int64)
    labels[found] = voxel_labels[position[found]]
    return labels, found


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--glb", required=True, help="The original model, whose topology is kept")
    parser.add_argument("--confidence", required=True,
                        help="*_confidence.npz written by inference_interactive.py")
    parser.add_argument("--names", help="*_parts.json naming the masks, for the report")
    parser.add_argument("--out", required=True, help="npy of one label per face of --glb")
    parser.add_argument("--low", type=float, default=0.4)
    parser.add_argument("--high", type=float, default=0.6,
                        help="The two thresholds a mask is scored across; a stable mask is "
                             "nearly the same at both.")
    parser.add_argument("--min_stability", type=float, default=0.9,
                        help="Masks below this are the sampler's own uncertainty, not parts.")
    parser.add_argument("--dedupe_iou", type=float, default=0.9,
                        help="Masks above this IoU are the same part reached from two seeds.")
    parser.add_argument("--min_voxel_share", type=float, default=0.005,
                        help="A mask keeping less than this share of the volume after the "
                             "smaller ones have taken their voxels is not a part of its own.")
    parser.add_argument("--smooth_iterations", type=int, default=2)
    parser.add_argument("--min_patch_ratio", type=float, default=0.001,
                        help="Absorb connected patches below this share of the surface.")
    args = parser.parse_args()

    stored = np.load(os.path.abspath(args.confidence))
    coords, confidence = stored["coords"], stored["confidence"]
    names = (json.load(open(os.path.abspath(args.names), encoding="utf-8"))
             if args.names else [f"mask_{i:02d}" for i in range(confidence.shape[1])])

    masks = confidence > (args.low + args.high) / 2.0
    scores = stability(confidence, args.low, args.high)
    shares = masks.mean(axis=0)
    print(f"{coords.shape[0]} voxels, {confidence.shape[1]} masks")
    print(f"  contested voxels {(masks.sum(axis=1) > 1).mean():.1%}, "
          f"mean masks per claimed voxel {masks.sum(axis=1)[masks.any(axis=1)].mean():.2f}")

    stable = np.flatnonzero(scores >= args.min_stability)
    print(f"  stable (>= {args.min_stability}): {len(stable)} of {len(scores)}")
    for i in np.argsort(-shares):
        mark = "" if i in stable else "  dropped: unstable"
        print(f"    {names[i]:<12} {shares[i]:>6.1%} stability {scores[i]:.3f}{mark}")

    kept, dropped = collapse_duplicates(masks, scores, args.dedupe_iou)
    kept = [i for i in kept if i in set(stable.tolist())]
    for rep, others in dropped.items():
        print(f"  {names[rep]} absorbed duplicates: {[names[o] for o in others]}")
    print(f"  {len(kept)} distinct masks after collapsing duplicates")

    min_voxels = int(args.min_voxel_share * coords.shape[0])
    voxel_labels, assigned = partition(masks, kept, min_voxels)
    print(f"  {len(assigned)} masks survived the smallest-first layout "
          f"(each needed {min_voxels} free voxels)")
    print(f"  volume covered {float((voxel_labels >= 0).mean()):.1%}")

    mesh = trimesh.load(os.path.abspath(args.glb), force="mesh")
    labels, found = voxel_labels_to_faces(mesh, coords, voxel_labels)
    print(f"  {len(mesh.faces)} faces, {found.mean():.1%} landed on an occupied voxel, "
          f"{float((labels >= 0).mean()):.1%} got a label")

    labels = tidy_labels(mesh, labels, len(assigned),
                         smooth_iterations=args.smooth_iterations,
                         min_patch_ratio=args.min_patch_ratio)
    out_names = [names[i] for i in assigned]
    areas = np.asarray(mesh.area_faces)
    print("\nfinal parts:")
    for k, name in enumerate(out_names):
        member = labels == k
        print(f"  {name:<12} {int(member.sum()):>8} faces  {areas[member].sum() / areas.sum():>6.1%} area")
    if (labels < 0).any():
        print(f"  {'<none>':<12} {int((labels < 0).sum()):>8} faces  unlabelled")

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.save(out, labels)
    with open(os.path.splitext(out)[0] + "_names.json", "w", encoding="utf-8") as handle:
        json.dump(out_names, handle, ensure_ascii=False, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
