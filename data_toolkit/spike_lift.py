"""Lift multi-view SAM3 masks onto a mesh and measure how faithfully they survived.

The metric is deliberately the round trip: 3D labels are re-projected through the
same cameras and compared against the SAM3 masks they came from. "Keep the SAM3
result" then has a number attached to it, on any model, with no reference to which
part is called what.
"""
import argparse
import json
import os
import sys

import numpy as np
import trimesh
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_toolkit.lift_sam3 import lift, load_cameras, load_masks
from data_toolkit.multiview import GLTF_TO_BLENDER

# seg.glb already carries to_glb's baked axis swap, so on top of the glTF->Blender
# conversion it lands a half turn about X. Measured, not derived: see spike_align_check.py.
ROTATIONS = {
    "seg_glb": np.diag([1.0, -1.0, -1.0]),
    "source_gltf": GLTF_TO_BLENDER,
}

PART_COLORS = np.array(
    [
        [220, 40, 40], [40, 90, 230], [30, 180, 70], [240, 210, 30],
        [40, 200, 210], [230, 70, 180], [140, 50, 200], [240, 130, 30],
    ],
    dtype=np.uint8,
)


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else float("nan")


def reference_part_map(mask_set, view, part_names):
    """One part per pixel, highest-scoring concept winning, plus the contested share.

    SAM3's masks overlap heavily -- on the monk figure 'armor' shares roughly 40% of
    its pixels with a body concept -- while a face can only carry one label. Scoring
    against the raw unions would therefore measure an unreachable target, so overlaps
    are resolved here the same way the vote resolves them: by confidence.
    """
    height, width = mask_set.foreground[view].shape
    best = np.full((height, width), -1, dtype=np.int32)
    strength = np.zeros((height, width), dtype=np.float64)
    claimed = np.zeros((height, width), dtype=bool)
    overlapped = np.zeros((height, width), dtype=bool)
    for concept, owner in enumerate(mask_set.owners):
        score = mask_set.scores[view, concept]
        if score <= 0:
            continue
        mask = mask_set.masks[view, concept]
        overlapped |= mask & claimed
        claimed |= mask
        take = mask & (score > strength)
        best[take] = part_names.index(owner)
        strength[take] = score
    share = float(overlapped.sum() / claimed.sum()) if claimed.any() else 0.0
    return best, share


def main():
    parser = argparse.ArgumentParser(description="Lift SAM3 masks onto a mesh and score the round trip.")
    parser.add_argument("--target_glb", required=True)
    parser.add_argument("--views_dir", required=True)
    parser.add_argument("--masks", default=None, help="default: <views_dir>/sam3_masks.npz")
    parser.add_argument("--rotation", choices=sorted(ROTATIONS), default="seg_glb")
    parser.add_argument("--smooth_iterations", type=int, default=2)
    parser.add_argument("--supersample", type=int, default=4,
                        help="Rasterise face ids this many times finer than the masks.")
    parser.add_argument("--min_patch_ratio", type=float, default=0.001,
                        help="Absorb connected patches below this share of total surface area.")
    parser.add_argument("--smoothness", type=float, default=0.0,
                        help="Charge for seam length. 0 follows the masks exactly, teeth included; "
                             "raising it buys intact boundaries at some cost in semantic fidelity.")
    parser.add_argument("--crease_gain", type=float, default=2.0,
                        help="How strongly seams are discounted along the model's own folds.")
    parser.add_argument("--confidence", default=None,
                        help="Confidence volume from inference_interactive.py. SegviGen's masks are "
                             "smooth but cannot express a grouped part, so they only make a seam "
                             "cheap where they change; the masks still decide which part wins.")
    parser.add_argument("--confidence_gain", type=float, default=4.0,
                        help="How strongly seams are discounted where SegviGen's masks change.")
    parser.add_argument("--confidence_hops", type=int, default=4,
                        help="How many faces wide the discounted channel around SegviGen's "
                             "boundary is; its masks change over about one triangle on a dense mesh.")
    parser.add_argument("--out_dir", default=None, help="Where to write label projections")
    args = parser.parse_args()

    views_dir = os.path.abspath(args.views_dir)
    manifest, cameras = load_cameras(views_dir)
    mask_set = load_masks(os.path.abspath(args.masks or os.path.join(views_dir, "sam3_masks.npz")))
    mesh = trimesh.load(os.path.abspath(args.target_glb), force="mesh")
    print(f"mesh {len(mesh.faces)} faces | {len(cameras)} views | concepts {mask_set.concepts}")

    part_labels, part_names, face_ids, report = lift(
        mesh,
        ROTATIONS[args.rotation],
        mask_set,
        cameras,
        manifest["camera_angle_x"],
        manifest["resolution"],
        smooth_iterations=args.smooth_iterations,
        supersample=args.supersample,
        min_patch_ratio=args.min_patch_ratio,
        smoothness=args.smoothness,
        crease_gain=args.crease_gain,
        confidence_path=os.path.abspath(args.confidence) if args.confidence else None,
        confidence_gain=args.confidence_gain,
        confidence_hops=args.confidence_hops,
    )
    # Score at the masks' own resolution by point-sampling the finer id buffer.
    face_ids = face_ids[:, :: args.supersample, :: args.supersample]
    print("\ncoverage:")
    for key, value in report.items():
        share = f"  ({100.0 * value / report['faces']:.1f}%)" if key != "faces" else ""
        print(f"  {key:>28}: {value}{share}")

    areas = mesh.area_faces
    print("\nparts:")
    for index, name in enumerate(part_names):
        selection = part_labels == index
        print(f"  {name:>10}: {int(selection.sum()):>7} faces  {100.0 * areas[selection].sum() / areas.sum():5.1f}% area")

    per_part = {name: [] for name in part_names}
    per_view = {name: [] for name in part_names}
    contested = []
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for view in range(len(cameras)):
        ids = face_ids[view]
        pixel_part = np.where(ids > 0, part_labels[np.maximum(ids - 1, 0)], -1)
        reference, overlap = reference_part_map(mask_set, view, part_names)
        contested.append(overlap)
        # Pixels no concept claimed carry no SAM3 opinion, so they are not evidence
        # either way; scoring them would just penalise whatever the mesh had to guess.
        judged = reference >= 0
        for index, name in enumerate(part_names):
            if not (reference == index).any():
                continue
            per_part[name].append(iou((pixel_part == index) & judged, reference == index))
            per_view[name].append(mask_set.views[view])
        if out_dir:
            def paint(per_pixel):
                canvas = np.full((*ids.shape, 3), 255, dtype=np.uint8)
                for index in range(len(part_names)):
                    canvas[per_pixel == index] = PART_COLORS[index % len(PART_COLORS)]
                return canvas

            Image.fromarray(paint(pixel_part)).save(
                os.path.join(out_dir, f"labels_{mask_set.views[view]}.png"))
            # The SAM3 side of the same comparison, so a ragged 3D boundary can be told
            # apart from a ragged 2D one instead of being blamed on the lift by default.
            Image.fromarray(paint(reference)).save(
                os.path.join(out_dir, f"sam3_{mask_set.views[view]}.png"))
            Image.fromarray(np.concatenate([paint(reference), paint(pixel_part)], axis=1)).save(
                os.path.join(out_dir, f"compare_{mask_set.views[view]}.png"))

    print(f"\nSAM3 masks overlapping another concept: {100.0 * float(np.mean(contested)):.1f}% of claimed pixels")
    print("round-trip IoU against the SAM3 part map the labels came from:")
    summary = {}
    for name in part_names:
        scores = [s for s in per_part[name] if not np.isnan(s)]
        summary[name] = {
            "mean_iou": float(np.mean(scores)) if scores else None,
            "min_iou": float(np.min(scores)) if scores else None,
            "views": len(scores),
        }
        if scores:
            order = np.argsort(per_part[name])
            worst = ", ".join(
                f"{per_view[name][i]} {per_part[name][i]:.3f}" for i in order[:3]
            )
            summary[name]["worst_views"] = worst
            print(f"  {name:>10}: mean {np.mean(scores):.3f}  over {len(scores)} views  | worst: {worst}")

    if out_dir:
        with open(os.path.join(out_dir, "lift_report.json"), "w", encoding="utf-8") as handle:
            json.dump({"coverage": report, "parts": summary}, handle, indent=2)
        np.save(os.path.join(out_dir, "part_labels.npy"), part_labels)
        with open(os.path.join(out_dir, "part_names.json"), "w", encoding="utf-8") as handle:
            json.dump(part_names, handle, ensure_ascii=False, indent=2)
        print(f"\nwrote label projections and report to {out_dir}")


if __name__ == "__main__":
    main()
