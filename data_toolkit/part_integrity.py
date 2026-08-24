"""Measure part integrity the same way across pipelines that disagree on topology.

Integrity is the requirement that a part be one solid piece: a segmentation whose parts
are shredded is useless no matter how good its semantics are. But integrity read on its
own has a degenerate optimum -- not segmenting at all scores perfectly -- so this reports
it next to how many parts were actually found, and refuses to rank one number alone.

Everything is measured on area rather than face counts, because the pipelines being
compared emit different meshes: SegviGen remeshes to ~100k faces while the fusion route
keeps the original half-million, and face counts are not comparable across the two.
Area shares are.

Labels come from whatever the mesh already carries -- a palette texture, vertex colours,
face colours, or a separate npy -- so a pipeline's own output can be measured without
first being split into a scene.
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


def face_colours(mesh):
    """Per-face RGB, from a texture, vertex colours or face colours, whichever exists."""
    visual = mesh.visual
    material = getattr(visual, "material", None)
    texture = getattr(material, "baseColorTexture", None) if material else None
    if texture is not None and getattr(visual, "uv", None) is not None:
        image = np.asarray(texture.convert("RGB"))
        height, width, _ = image.shape
        uv = np.asarray(visual.uv)[np.asarray(mesh.faces)].mean(axis=1)
        rows = np.clip(((1.0 - uv[:, 1]) * (height - 1)).astype(int), 0, height - 1)
        cols = np.clip((uv[:, 0] * (width - 1)).astype(int), 0, width - 1)
        return image[rows, cols]
    if getattr(visual, "kind", None) == "face":
        return np.asarray(visual.face_colors)[:, :3]
    if getattr(visual, "kind", None) == "vertex":
        colours = np.asarray(visual.vertex_colors)[:, :3].astype(np.float64)
        return colours[np.asarray(mesh.faces)].mean(axis=1).astype(np.uint8)
    raise SystemExit("mesh carries no colour to read labels from; pass --labels instead")


def labels_from_colours(colours, levels):
    """Cluster near-identical colours into labels.

    A palette baked into a texture comes back with hundreds of variants of each colour
    because the image was compressed, so quantising first is what recovers the intended
    part count instead of reporting compression noise as parts.
    """
    step = 256 // levels
    keys = (colours.astype(np.int64) // step)
    keys = keys[:, 0] * levels * levels + keys[:, 1] * levels + keys[:, 2]
    uniq, labels = np.unique(keys, return_inverse=True)
    return labels.astype(np.int64), len(uniq)


def integrity(mesh, labels, speckle_ratio):
    areas = mesh.area_faces
    total = areas.sum()
    adjacency = welded_face_adjacency(mesh)
    rows = []
    for k in [int(i) for i in np.unique(labels) if i >= 0]:
        member = labels == k
        share = areas[member].sum() / total
        # Components are found within the part, so a part that merely touches another
        # is not counted as broken; only a part that is itself in pieces is.
        sub = adjacency[member[adjacency[:, 0]] & member[adjacency[:, 1]]]
        index = np.full(len(labels), -1, dtype=np.int64)
        faces = np.nonzero(member)[0]
        index[faces] = np.arange(len(faces))
        local = index[sub] if len(sub) else np.zeros((0, 2), dtype=np.int64)
        comps = connected_components(local, nodes=np.arange(len(faces)))
        comp_areas = np.array([areas[faces[c]].sum() for c in comps])
        order = np.argsort(-comp_areas)
        cum = np.cumsum(comp_areas[order]) / max(comp_areas.sum(), 1e-12)
        speck = comp_areas[comp_areas / max(comp_areas.sum(), 1e-12) < speckle_ratio]
        rows.append({
            "label": k, "area_share": share, "faces": int(member.sum()),
            "comps": len(comps),
            "core95": int(np.searchsorted(cum, 0.95) + 1),
            "largest": float(comp_areas[order[0]] / max(comp_areas.sum(), 1e-12)),
            "speckle": float(speck.sum() / max(comp_areas.sum(), 1e-12)),
            "speckle_patches": int(len(speck)),
        })
    return sorted(rows, key=lambda r: -r["area_share"])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("glb", nargs="+", help="Meshes whose colours encode the parts")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="One npy of per-face labels per glb, when colour is absent")
    parser.add_argument("--levels", type=int, default=8,
                        help="Colour quantisation per channel; raise it if distinct parts "
                             "share a hue, lower it if compression noise splits one part.")
    parser.add_argument("--speckle_ratio", type=float, default=0.01,
                        help="A component below this share of its part counts as speckle.")
    parser.add_argument("--shredded_ratio", type=float, default=0.05,
                        help="A part losing this much of its area to speckle is broken.")
    parser.add_argument("--top", type=int, default=8, help="Parts listed per mesh")
    args = parser.parse_args()

    for i, path in enumerate(args.glb):
        mesh = trimesh.load(os.path.abspath(path), force="mesh")
        if args.labels:
            labels = np.load(os.path.abspath(args.labels[i]))
            n_raw = len(np.unique(labels[labels >= 0]))
        else:
            labels, n_raw = labels_from_colours(face_colours(mesh), args.levels)

        rows = integrity(mesh, labels, args.speckle_ratio)
        areas = mesh.area_faces
        # Whole-mesh shells give context: a remesh that emits an inner wall doubles every
        # component count, so a high shell count means the comps column is inflated.
        shells = len(connected_components(welded_face_adjacency(mesh),
                                          nodes=np.arange(len(mesh.faces))))
        speck_area = sum(r["speckle"] * r["area_share"] for r in rows)

        print(f"\n{path}")
        print(f"  {len(mesh.faces)} faces, {shells} shells, {len(rows)} parts "
              f"(raw colour clusters {n_raw})")
        print(f"  {'part':>6}{'area':>9}{'comps':>7}{'core95':>8}{'largest':>9}"
              f"{'speckle':>9}{'patches':>9}")
        for r in rows[:args.top]:
            print(f"  {r['label']:>6}{r['area_share']:>8.1%}{r['comps']:>7}{r['core95']:>8}"
                  f"{r['largest']:>8.1%}{r['speckle']:>8.2%}{r['speckle_patches']:>9}")
        if len(rows) > args.top:
            print(f"  ... {len(rows) - args.top} smaller parts")
        # A part in two halves of equal area is not shredded: that is either the inner
        # wall a remesh leaves behind or a symmetric pair (two hands are one part and two
        # pieces). Shredding is speckle, so that is what decides whether a part is broken,
        # and core95 is left as context rather than a verdict.
        big = [r for r in rows if r["area_share"] >= 0.01]
        shredded = [r for r in big if r["speckle"] >= args.shredded_ratio]
        print(f"  SUMMARY parts>=1% area: {len(big)} | shredded "
              f"(speckle>={args.shredded_ratio:.0%} of the part): {len(shredded)}"
              f"{' ' + str([r['label'] for r in shredded]) if shredded else ''}"
              f" | model speckle {speck_area:.2%}")


if __name__ == "__main__":
    main()
