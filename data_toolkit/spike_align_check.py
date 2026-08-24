"""Find how a SegviGen output mesh is oriented relative to the source model.

The remeshed seg.glb carries a baked axis swap, and stacking that on top of the
glTF->Blender conversion is easy to get wrong by hand. Every axis-aligned rotation
is scored against the source model's silhouettes instead, so the answer is measured.
"""
import argparse
import itertools
import os
import sys

import numpy as np
import trimesh

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_toolkit.multiview import gltf_to_blender, rasterize_face_ids
from data_toolkit.spike_camera_check import iou, load_reference


def axis_aligned_rotations():
    """The 24 rotations that map axes onto axes, as (matrix, label) pairs."""
    rotations = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            matrix = np.zeros((3, 3), dtype=np.float64)
            for row, column in enumerate(permutation):
                matrix[row, column] = signs[row]
            if np.linalg.det(matrix) > 0:
                axes = "xyz"
                label = " ".join(
                    f"{'-' if signs[row] < 0 else '+'}{axes[column]}"
                    for row, column in enumerate(permutation)
                )
                rotations.append((matrix, label))
    return rotations


def normalize_with_reference(vertices, reference):
    """Scale/centre by the reference bbox so a remesh keeps the source's pose."""
    low, high = reference.min(axis=0), reference.max(axis=0)
    scale = 1.0 / float((high - low).max())
    return (np.asarray(vertices, dtype=np.float64) - (low + high) / 2.0) * scale


def main():
    parser = argparse.ArgumentParser(description="Score axis-aligned orientations of a remeshed glb.")
    parser.add_argument("--source_glb", required=True, help="Model the reference views were rendered from.")
    parser.add_argument("--target_glb", required=True, help="Remeshed glb whose orientation is unknown.")
    parser.add_argument("--views_dir", required=True)
    args = parser.parse_args()

    manifest, cameras, masks = load_reference(os.path.abspath(args.views_dir))
    reference = np.stack(masks)

    source = trimesh.load(os.path.abspath(args.source_glb), force="mesh")
    source_blender = gltf_to_blender(source.vertices)

    target = trimesh.load(os.path.abspath(args.target_glb), force="mesh")
    raw = np.asarray(target.vertices, dtype=np.float64)
    print(f"target: {len(target.faces)} faces, extent {raw.max(0) - raw.min(0)}")

    scores = []
    for matrix, label in axis_aligned_rotations():
        rotated = raw @ matrix.T
        # Both meshes are normalised by their own bbox, then compared in the same frame:
        # a remesh never reproduces the source bbox exactly, so a shared reference bbox
        # would bias every candidate by the same unknown amount.
        vertices = normalize_with_reference(rotated, rotated)
        ids = rasterize_face_ids(
            vertices,
            target.faces,
            cameras,
            manifest["camera_angle_x"],
            resolution=manifest["resolution"],
        )
        per_view = [iou(reference[i], ids[i] > 0) for i in range(len(cameras))]
        scores.append((float(np.mean(per_view)), label, per_view))

    scores.sort(reverse=True)
    print(f"\nsource extent (blender) {source_blender.max(0) - source_blender.min(0)}")
    for mean, label, _ in scores[:5]:
        print(f"  {label:>12}  mean IoU {mean:.4f}")
    best = scores[0]
    print(f"\nbest {best[1]}  mean IoU {best[0]:.4f}")
    for camera, score in zip(cameras, best[2]):
        print(f"  {camera.name:>16}  IoU {score:.4f}")


if __name__ == "__main__":
    main()
