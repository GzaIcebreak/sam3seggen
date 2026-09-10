"""Common refinement ("meet") of several SegviGen full_seg samples -> over-segmented atoms.

full_seg has no granularity knob and a single sample regularly glues neighbouring parts
together (shoulder plate + arm, thigh + shin). Every sample is a valid partition of the
same surface, so intersecting them keeps every boundary any sample drew: two faces stay
in one atom only if *every* sample coloured them alike. The result can only be finer than
any single sample, which is harmless for the semantic vote that follows (it only merges).

Each sample is a different remesh of the same voxel grid, so labels are moved onto one
reference mesh by nearest face centroid before intersecting.

Mirroring counts as a free extra sample. full_seg knows nothing about symmetry, so on a
symmetric model it draws a boundary on one side and misses its counterpart -- on the robot
test model the left ankle was cut in some sample while the right foot stayed welded to the
shin in all five. Reflecting a sample's labels across the model's symmetry plane turns
"cut on one side" into "cut on both".
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from trimesh.graph import connected_components

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_toolkit.parts_rebake import (  # noqa: E402
    absorb_small_fragments,
    cluster_parts,
    face_base_colors,
    load_single_mesh,
    welded_face_adjacency,
)

# Finer than parts_rebake's 40: a single full_seg sample already carries distinct
# clusters ~25 apart that 40 folds together, and the meet needs every cut it can get.
DEFAULT_COLOR_TOL = 20.0
DEFAULT_MIN_FACES = 150
# A face counts as mirrored if its reflection lands this close (share of the bbox
# diagonal) to some face; the plane is used only if that holds for MIRROR_SHARE of them.
DEFAULT_MIRROR_TOLERANCE = 0.01
DEFAULT_MIRROR_SHARE = 0.9


def mirror_map(mesh, axis, tolerance=DEFAULT_MIRROR_TOLERANCE):
    """(face -> its reflection across the model's mid-plane, share of faces that matched).

    The plane is the bounding box's mid-plane, which is exact for a model that is
    symmetric at all: reflecting its surface maps the bbox onto itself.
    """
    centers = np.asarray(mesh.triangles_center)
    reflected = centers.copy()
    reflected[:, axis] = mesh.bounds[:, axis].sum() - reflected[:, axis]
    distance, index = cKDTree(centers).query(reflected)
    diagonal = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    return index, float((distance < tolerance * diagonal).mean())


def detect_mirror_axis(mesh, min_share=DEFAULT_MIRROR_SHARE,
                       tolerance=DEFAULT_MIRROR_TOLERANCE):
    """The axis the model is most symmetric about, or (None, share) if none is."""
    best_axis, best_share, best_index = None, 0.0, None
    for axis in range(3):
        index, share = mirror_map(mesh, axis, tolerance)
        if share > best_share:
            best_axis, best_share, best_index = axis, share, index
    if best_share < min_share:
        return None, best_share, None
    return best_axis, best_share, best_index


def sample_labels(mesh, reference, color_tol=DEFAULT_COLOR_TOL):
    """Colour-cluster labels of one sample mesh, expressed on `reference`'s faces."""
    labels, _ = cluster_parts(face_base_colors(mesh), np.asarray(mesh.area_faces), color_tol)
    if mesh is reference:
        return labels
    _, index = cKDTree(mesh.triangles_center).query(reference.triangles_center)
    return labels[index]


def meet_labels(label_stack, adjacency, min_faces=DEFAULT_MIN_FACES):
    """Intersect partitions given as [n_faces, n_samples]; return (atoms, n_raw).

    Slivers under `min_faces` (where two samples' boundaries almost but not quite
    coincide) go to their majority neighbour, then atoms are split by connectivity so a
    coincidental colour collision across the body cannot glue two pieces together.
    """
    _, meet = np.unique(label_stack, axis=0, return_inverse=True)
    meet = meet.ravel()
    n_raw = len(np.unique(meet))
    meet = absorb_small_fragments(adjacency, meet, min_faces=min_faces, iterations=5)
    same = meet[adjacency[:, 0]] == meet[adjacency[:, 1]]
    components = connected_components(adjacency[same], nodes=np.arange(len(meet)))
    atoms = np.zeros(len(meet), dtype=np.int64)
    for index, component in enumerate(components):
        atoms[component] = index
    return atoms, n_raw


def meet_samples(sample_glbs, color_tol=DEFAULT_COLOR_TOL, min_faces=DEFAULT_MIN_FACES,
                 mirror="auto"):
    """Return (reference mesh, atom label per face, report dict). First glb is the reference.

    `mirror` is "auto" (use the symmetry plane if the model has one), "none", or an axis
    name; see mirror_map.
    """
    reference = load_single_mesh(sample_glbs[0])
    stack, counts = [], []
    for index, path in enumerate(sample_glbs):
        mesh = reference if index == 0 else load_single_mesh(path)
        labels = sample_labels(mesh, reference, color_tol)
        stack.append(labels)
        counts.append(int(len(np.unique(labels))))

    axis, share, index = None, 0.0, None
    if mirror == "auto":
        axis, share, index = detect_mirror_axis(reference)
    elif mirror != "none":
        axis = "xyz".index(mirror)
        index, share = mirror_map(reference, axis)
    if index is not None:
        stack += [labels[index] for labels in stack]

    adjacency = welded_face_adjacency(reference)
    atoms, n_raw = meet_labels(np.stack(stack, axis=1), adjacency, min_faces)
    areas = np.asarray(reference.area_faces)
    shares = np.bincount(atoms, weights=areas) / areas.sum()
    report = {
        "samples": [os.path.abspath(p) for p in sample_glbs],
        "labels_per_sample": counts,
        "mirror_axis": None if axis is None or index is None else "xyz"[axis],
        "mirror_share": round(share, 4),
        "raw_atoms": int(n_raw),
        "atoms": int(atoms.max() + 1),
        "largest_share": float(shares.max()),
        "shares": [float(s) for s in np.sort(shares)[::-1]],
    }
    return reference, atoms, report


def atoms_debug_glb(mesh, atoms, path, seed=3):
    """Vertex-coloured copy of the reference mesh, one random colour per atom."""
    palette = np.random.default_rng(seed).integers(40, 230, size=(atoms.max() + 1, 3))
    colors = np.concatenate([palette[atoms], np.full((len(atoms), 1), 255)], axis=1).astype(np.uint8)
    out = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
    out.visual = trimesh.visual.ColorVisuals(out, face_colors=colors)
    out.export(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", nargs="+", required=True,
                        help="full_seg seg.glb files; the first one is the reference mesh")
    parser.add_argument("--out_labels", required=True, help="npy: atom id per reference face")
    parser.add_argument("--out_report", default=None, help="json summary (default: next to labels)")
    parser.add_argument("--debug_glb", default=None, help="optional vertex-coloured atoms glb")
    parser.add_argument("--color_tol", type=float, default=DEFAULT_COLOR_TOL)
    parser.add_argument("--min_faces", type=int, default=DEFAULT_MIN_FACES,
                        help="slivers under this many faces join their majority neighbour")
    parser.add_argument("--mirror", default="auto", choices=("auto", "none", "x", "y", "z"),
                        help="also intersect each sample reflected across this plane")
    args = parser.parse_args()

    reference, atoms, report = meet_samples(args.samples, args.color_tol, args.min_faces,
                                            args.mirror)
    out_labels = os.path.abspath(args.out_labels)
    os.makedirs(os.path.dirname(out_labels) or ".", exist_ok=True)
    np.save(out_labels, atoms)
    report_path = args.out_report or os.path.splitext(out_labels)[0] + "_report.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    if args.debug_glb:
        atoms_debug_glb(reference, atoms, os.path.abspath(args.debug_glb))
    mirror = (f"mirrored about {report['mirror_axis']}" if report["mirror_axis"]
              else f"not mirrored (symmetry {report['mirror_share']:.2f})")
    print(f"labels per sample: {report['labels_per_sample']}, {mirror}; "
          f"meet {report['raw_atoms']} raw -> {report['atoms']} atoms, "
          f"largest {report['largest_share']:.3f} of the area")
    print(f"saved {out_labels}")


if __name__ == "__main__":
    main()
