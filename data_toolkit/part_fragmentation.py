"""Measure how fragmented the parts of a split glb are.

Counting connected components alone is misleading: a remesh usually emits a second
inner wall, doubling every count, and a grouped part is legitimately disconnected (a
hand is not attached to a leg). What actually looks broken is speckle -- patches too
small to be a real region -- so that is reported as a share of each part's area.
"""
import argparse
import os
import sys

import numpy as np
import trimesh
from trimesh.graph import connected_components

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_toolkit.parts_rebake import welded_face_adjacency


def cut_quality(scene):
    """How ragged the seams between parts are, measured on the reassembled surface.

    Speckle counts detached patches, but the visible "torn" look comes from single
    triangles flipping back and forth along a seam, and those stay attached to their
    part. Reassembling the parts and measuring the length of the cut catches that: for
    the same partition, a longer cut means a more ragged one.
    """
    vertices, faces, labels = [], [], []
    for index, geometry in enumerate(scene.geometry.values()):
        faces.append(np.asarray(geometry.faces) + sum(len(v) for v in vertices))
        vertices.append(np.asarray(geometry.vertices))
        labels.append(np.full(len(geometry.faces), index, dtype=np.int64))

    merged = trimesh.Trimesh(
        vertices=np.concatenate(vertices), faces=np.concatenate(faces), process=False
    )
    labels = np.concatenate(labels)
    adjacency = welded_face_adjacency(merged)
    if len(adjacency) == 0:
        return None

    crossing = labels[adjacency[:, 0]] != labels[adjacency[:, 1]]
    # A triangle is "on the seam" if any of its neighbours belongs to another part; the
    # share of such triangles is what reads as a clean edge versus a torn one.
    on_seam = np.zeros(len(labels), dtype=bool)
    on_seam[adjacency[crossing].ravel()] = True
    return {
        "cut_edges": int(crossing.sum()),
        "seam_faces": int(on_seam.sum()),
        "seam_share": 100.0 * on_seam.mean(),
    }


def main():
    parser = argparse.ArgumentParser(description="Report per-part fragmentation of split glbs.")
    parser.add_argument("glb", nargs="+", help="One or more combined part glbs to compare")
    parser.add_argument("--speckle_ratio", type=float, default=0.01,
                        help="Patches below this share of a part's area count as speckle.")
    args = parser.parse_args()

    for path in args.glb:
        scene = trimesh.load(os.path.abspath(path), force="scene")
        print(f"\n{path}")
        for name, geometry in scene.geometry.items():
            adjacency = welded_face_adjacency(geometry)
            components = connected_components(adjacency, nodes=np.arange(len(geometry.faces)))
            areas = geometry.area_faces
            shares = np.array([areas[patch].sum() for patch in components]) / areas.sum()
            order = np.argsort(-shares)
            speckle = shares[shares < args.speckle_ratio]
            print(
                f"  {name:>22}: {len(geometry.faces):>6} faces | {len(components):>4} regions | "
                f"largest {' '.join(f'{100 * shares[i]:.1f}%' for i in order[:4])} | "
                f"speckle {100 * speckle.sum():.2f}% in {len(speckle)} patches"
            )
        seams = cut_quality(scene)
        if seams:
            print(
                f"  {'seams':>22}: {seams['cut_edges']} cut edges | "
                f"{seams['seam_faces']} faces on a seam ({seams['seam_share']:.2f}% of the mesh)"
            )


if __name__ == "__main__":
    main()
