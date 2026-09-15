"""Cut the source model itself by labels that were voted on the remesh.

The split runs on TRELLIS's remesh (~100k faces); the source model can carry far more --
the monk's hand has 22k faces there and 5k on the remesh, so fingers that the source
has are gone before X-Part ever sees the part. Exporting from the remesh also means a
Blender bake to get the texture back. Here the per-face labels are carried over to the
source's own faces (nearest remesh face, then a majority smooth), and the parts are cut
straight out of the source with its UVs and material intact: full resolution, no bake.

Only a single-mesh, single-material source qualifies; anything else falls back to the
remesh + bake path (returns None), which handles every glb.
"""
from __future__ import annotations

import json
import os

import numpy as np
import trimesh
from scipy.spatial import cKDTree

# seg.glb comes out of to_glb with an axis swap relative to the source glTF; which one is
# measured, not assumed (see data_toolkit/spike_align_check.py).
CANDIDATE_ROTATIONS = {
    "identity": np.eye(3),
    "x-half-turn": np.diag([1.0, -1.0, -1.0]),
    "y<->z": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float),
    "y<->z-flip": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float),
}
DEFAULT_SMOOTH_ITERATIONS = 2


def fit_frame(remesh_centroids, source_centroids, sample=4000, seed=0):
    """(rotation, scale, shift, mean distance): the candidate frame that lands the remesh
    on the source, with scale and shift fitted from the bounding boxes."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(remesh_centroids), min(sample, len(remesh_centroids)), replace=False)
    tree = cKDTree(source_centroids)
    src_lo, src_hi = source_centroids.min(axis=0), source_centroids.max(axis=0)
    best = None
    for name, rotation in CANDIDATE_ROTATIONS.items():
        turned = remesh_centroids @ rotation.T
        lo, hi = turned.min(axis=0), turned.max(axis=0)
        scale = float(np.max(src_hi - src_lo) / max(np.max(hi - lo), 1e-9))
        shift = (src_lo + src_hi) / 2 - scale * (lo + hi) / 2
        distance = float(tree.query(turned[pick] * scale + shift)[0].mean())
        if best is None or distance < best[-1]:
            best = (name, rotation, scale, shift, distance)
    return best


def smooth_labels(mesh, labels, iterations=DEFAULT_SMOOTH_ITERATIONS):
    """Majority vote over face neighbours; a face keeps its label on a tie."""
    labels = np.array(labels).copy()
    if iterations <= 0 or labels.max() < 0:
        return labels
    adjacency = np.asarray(mesh.face_adjacency)
    count = len(labels)
    for _ in range(iterations):
        votes = np.zeros((count, labels.max() + 2), dtype=np.int32)  # last column: unlabelled
        col = np.where(labels < 0, labels.max() + 1, labels)
        np.add.at(votes, (adjacency[:, 0], col[adjacency[:, 1]]), 1)
        np.add.at(votes, (adjacency[:, 1], col[adjacency[:, 0]]), 1)
        np.add.at(votes, (np.arange(count), col), 1)  # own label breaks ties
        winner = votes.argmax(axis=1)
        labels = np.where(winner == labels.max() + 1, -1, winner)
    return labels


def transfer_labels(remesh, remesh_labels, source, smooth_iterations=DEFAULT_SMOOTH_ITERATIONS):
    """Per-face labels for `source`, from the nearest remesh face after frame fitting."""
    remesh_labels = np.asarray(remesh_labels)
    name, rotation, scale, shift, distance = fit_frame(
        np.asarray(remesh.triangles_center), np.asarray(source.triangles_center))
    aligned = np.asarray(remesh.triangles_center) @ rotation.T * scale + shift
    nearest = cKDTree(aligned).query(np.asarray(source.triangles_center))[1]
    labels = smooth_labels(source, remesh_labels[nearest], smooth_iterations)
    return labels, {"frame": name, "scale": scale, "mean_distance": distance}


def load_single_textured(source_glb):
    """The source as one textured Trimesh, or None when it is not that simple."""
    scene = trimesh.load(source_glb, force="scene")
    if len(scene.geometry) != 1:
        return None
    mesh = next(iter(scene.geometry.values())).copy()
    node = scene.graph.geometry_nodes[next(iter(scene.geometry))][0]
    transform, _ = scene.graph.get(node)
    mesh.apply_transform(transform)
    visual = getattr(mesh, "visual", None)
    if getattr(visual, "uv", None) is None or getattr(visual, "material", None) is None:
        return None
    return mesh


def export_from_source(mesh_path, source_glb, labels_npy, names_json, out_glb,
                       smooth_iterations=DEFAULT_SMOOTH_ITERATIONS, step="[export]"):
    """parts.json-style manifest, or None if the source is not a single textured mesh."""
    from data_toolkit.parts_rebake import LABEL_COLORS, load_single_mesh

    source = load_single_textured(source_glb)
    if source is None:
        print(f"{step} source is not one textured mesh; exporting from the remesh instead")
        return None
    remesh = load_single_mesh(mesh_path)
    remesh_labels = np.load(labels_npy)
    if len(remesh_labels) != len(remesh.faces):
        raise SystemExit(f"{len(remesh_labels)} labels for {len(remesh.faces)} remesh faces")
    with open(names_json, "r", encoding="utf-8") as handle:
        names = json.load(handle)
    labels, fit = transfer_labels(remesh, remesh_labels, source, smooth_iterations)
    print(f"{step} cutting the source model ({len(source.faces)} faces) by labels carried "
          f"over from the remesh ({len(remesh.faces)} faces; frame {fit['frame']}, "
          f"scale {fit['scale']:.4f}, fit {fit['mean_distance']:.4f})")

    texture = getattr(source.visual.material, "image", None)
    if texture is None:
        texture = getattr(source.visual.material, "baseColorTexture", None)
    texture_size = None if texture is None else int(max(texture.size))
    total_area = float(source.area)
    scene = trimesh.Scene()
    manifest = []
    for label, name in enumerate(names):
        faces = np.flatnonzero(labels == label)
        if not len(faces):
            continue
        part = source.submesh([faces], append=True)
        node = f"part_{label:02d}_{name}"
        scene.add_geometry(part, node_name=node, geom_name=node)
        area = float(source.area_faces[faces].sum())
        manifest.append({
            "label": int(label), "name": name, "node": node,
            "part_color": [int(c) for c in LABEL_COLORS[label % len(LABEL_COLORS)]],
            "faces": int(len(faces)), "area": area, "texture_size": texture_size,
            "area_share": area / max(total_area, 1e-12), "source": "original",
        })
        print(f"  {node}: {len(faces)} faces, {100.0 * area / total_area:.1f}% of the area")
    dropped = int((labels < 0).sum())
    if dropped:
        print(f"  {dropped} source faces carried no label and were left out")
    out_dir = os.path.dirname(os.path.abspath(out_glb)) or "."
    os.makedirs(out_dir, exist_ok=True)
    scene.export(out_glb)
    with open(os.path.join(out_dir, "parts.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest
