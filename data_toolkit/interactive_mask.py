"""Turn interactive SegviGen runs into per-face part labels.

Interactive mode answers one question at a time: it paints the part the seed points
picked out white and everything else black. Reading that back per face gives a
confidence, and because every run decodes the same shape latent into the same mesh, the
face indices line up across runs -- so N single-part runs can be combined by taking the
strongest claim on each face, with unclaimed faces falling to a catch-all label.
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


def face_confidence(glb_path):
    """Per-face mask value in [0, 1], sampled at each face's UV centroid."""
    mesh = trimesh.load(os.path.abspath(glb_path), force="mesh")
    texture = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB"))
    height, width, _ = texture.shape
    uv = np.asarray(mesh.visual.uv)[np.asarray(mesh.faces)].mean(axis=1)
    columns = np.clip((uv[:, 0] * (width - 1)).astype(int), 0, width - 1)
    # glTF UVs run top-down, image rows run the other way.
    rows = np.clip(((1.0 - uv[:, 1]) * (height - 1)).astype(int), 0, height - 1)
    return mesh, texture[rows, columns].mean(axis=1) / 255.0


def labels_from_palette(glb_path, palette):
    """Per-face labels for a glb painted one flat palette colour per part.

    The combined output already resolved the overlaps between parts at the voxel level, so
    each face just needs matching to the nearest palette entry -- exact up to the texture
    filtering and decimation that ran afterwards.
    """
    mesh = trimesh.load(os.path.abspath(glb_path), force="mesh")
    texture = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB")).astype(np.float64)
    height, width, _ = texture.shape
    uv = np.asarray(mesh.visual.uv)[np.asarray(mesh.faces)].mean(axis=1)
    columns = np.clip((uv[:, 0] * (width - 1)).astype(int), 0, width - 1)
    rows = np.clip(((1.0 - uv[:, 1]) * (height - 1)).astype(int), 0, height - 1)
    colours = texture[rows, columns]
    distance = np.linalg.norm(colours[:, None, :] - np.asarray(palette, dtype=np.float64)[None], axis=2)
    return mesh, distance.argmin(axis=1).astype(np.int32), distance.min(axis=1)


def tidy_labels(mesh, labels, n_parts, smooth_iterations=2, min_patch_ratio=0.001):
    """Smooth the labels and absorb speckle, using the same passes as the lifting path.

    Where two regions both claim a voxel -- the hands gripping the staff, most obviously --
    the winner comes down to a hair's difference between two near-binary masks, and the
    result is salt and pepper along the contact. That is decided per voxel with no notion of
    the surface, so it is cleaned up here, on the mesh, where neighbours are known.
    """
    from data_toolkit.lift_sam3 import absorb_small_patches
    from data_toolkit.parts_rebake import smooth_labels, welded_face_adjacency

    adjacency = welded_face_adjacency(mesh)
    labels = smooth_labels(adjacency, labels, n_parts, smooth_iterations)
    labels, absorbed = absorb_small_patches(
        adjacency, labels, np.asarray(mesh.area_faces), min_patch_ratio
    )
    print(f"smoothed over {len(adjacency)} adjacent pairs, absorbed {absorbed} speckle patches")
    return labels


def combine(runs, threshold=0.5, rest_name="rest"):
    """One label per face: the most confident part that clears the threshold."""
    names = [name for name, _ in runs]
    confidence = np.stack([values for _, values in runs])
    winner = confidence.argmax(axis=0)
    labels = np.where(confidence.max(axis=0) >= threshold, winner, len(names))
    return labels.astype(np.int32), names + [rest_name]


def main():
    parser = argparse.ArgumentParser(description="Turn interactive output into per-face labels.")
    parser.add_argument("--run", action="append", metavar="NAME=GLB",
                        help="A single-part mask glb, repeated once per part, e.g. --run staff=staff.glb")
    parser.add_argument("--palette_glb", help="A combined glb painted one palette colour per part")
    parser.add_argument("--parts", help="JSON list of part names in palette order, written next to --palette_glb")
    parser.add_argument("--out_labels", required=True)
    parser.add_argument("--out_names", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--smooth_iterations", type=int, default=2,
                        help="Neighbour-vote passes over the faces, to settle contacts where two "
                             "regions both claimed the voxel.")
    parser.add_argument("--min_patch_ratio", type=float, default=0.001,
                        help="Absorb connected patches below this share of the model's area.")
    args = parser.parse_args()

    if args.palette_glb:
        if not args.parts:
            raise SystemExit("--palette_glb needs --parts")
        with open(os.path.abspath(args.parts), "r", encoding="utf-8") as handle:
            names = json.load(handle)
        from data_toolkit.parts_rebake import LABEL_COLORS
        mesh, labels, error = labels_from_palette(args.palette_glb, LABEL_COLORS[: len(names)])
        print(f"{len(mesh.faces)} faces, colour match error: median {np.median(error):.1f}, "
              f"{int((error > 60).sum())} faces further than 60 from any palette entry")
        if args.smooth_iterations or args.min_patch_ratio:
            labels = tidy_labels(mesh, labels, len(names), args.smooth_iterations, args.min_patch_ratio)
        for index, name in enumerate(names):
            selection = labels == index
            print(f"  {name:>10}: {int(selection.sum())} faces, "
                  f"{mesh.area_faces[selection].sum() / mesh.area_faces.sum() * 100:.1f}% area")
        np.save(os.path.abspath(args.out_labels), labels)
        with open(os.path.abspath(args.out_names), "w", encoding="utf-8") as handle:
            json.dump(names, handle, indent=2)
        print(f"wrote {args.out_labels} and {args.out_names}")
        return

    if not args.run:
        raise SystemExit("pass either --palette_glb with --parts, or --run once per part")
    runs, mesh = [], None
    for entry in args.run:
        if "=" not in entry:
            raise SystemExit(f"--run wants NAME=GLB, got {entry!r}")
        name, path = entry.split("=", 1)
        mesh, values = face_confidence(path)
        runs.append((name, values))
        claimed = values >= args.threshold
        print(f"{name:>10}: {claimed.sum()} of {len(values)} faces claimed "
              f"({mesh.area_faces[claimed].sum() / mesh.area_faces.sum() * 100:.1f}% area), "
              f"{int(((values > 0.15) & (values < args.threshold)).sum())} faces undecided")

    sizes = {len(values) for _, values in runs}
    if len(sizes) > 1:
        raise SystemExit(f"runs disagree on face count: {sizes}")

    labels, names = combine(runs, args.threshold)
    for index, name in enumerate(names):
        selection = labels == index
        print(f"  -> {name:>10}: {int(selection.sum())} faces, "
              f"{mesh.area_faces[selection].sum() / mesh.area_faces.sum() * 100:.1f}% area")

    np.save(os.path.abspath(args.out_labels), labels)
    with open(os.path.abspath(args.out_names), "w", encoding="utf-8") as handle:
        json.dump(names, handle, indent=2)
    print(f"wrote {args.out_labels} and {args.out_names}")


if __name__ == "__main__":
    main()
