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


SHELL_WHOLE_SHARE = 0.02   # a welded shell under this share of the area is one piece


def shell_labels(source, labels, max_share=SHELL_WHOLE_SHARE):
    """One label per small welded shell: the area-weighted majority of its faces.

    Game assets are built from hundreds of loose pieces (a knight of 426 shells, a
    bicycle of 827); a per-face nearest-remesh-face transfer gives the faces of one
    rivet or plate different labels and the part comes out as a scatter of crumbs.
    A piece that small is one thing and belongs to one part. Shells above `max_share`
    of the area keep their per-face labels: a single-shell body still has to be cut.
    Returns (labels, number of shells relabelled)."""
    labels = np.array(labels).copy()
    if labels.max() < 0 or not len(source.faces):
        return labels, 0
    welded = trimesh.Trimesh(np.asarray(source.vertices), np.asarray(source.faces), process=False)
    welded.merge_vertices(merge_tex=True, merge_norm=True)
    components = trimesh.graph.connected_components(
        welded.face_adjacency, nodes=np.arange(len(welded.faces)), min_len=1)
    if len(components) <= 1:
        return labels, 0
    areas = np.asarray(source.area_faces)
    total = max(float(areas.sum()), 1e-12)
    changed = 0
    for faces in components:
        faces = np.asarray(faces)
        if areas[faces].sum() > max_share * total:
            continue
        here = labels[faces]
        valid = here >= 0
        if not valid.any():
            continue
        votes = np.bincount(here[valid], weights=areas[faces][valid])
        winner = int(votes.argmax())
        if (here != winner).any():
            labels[faces] = winner
            changed += 1
    return labels, changed


def transfer_labels(remesh, remesh_labels, source, smooth_iterations=DEFAULT_SMOOTH_ITERATIONS,
                    shell_share=SHELL_WHOLE_SHARE):
    """Per-face labels for `source`: nearest remesh face after frame fitting, a majority
    smooth, then one label per small welded shell (see shell_labels)."""
    remesh_labels = np.asarray(remesh_labels)
    name, rotation, scale, shift, distance = fit_frame(
        np.asarray(remesh.triangles_center), np.asarray(source.triangles_center))
    aligned = np.asarray(remesh.triangles_center) @ rotation.T * scale + shift
    nearest = cKDTree(aligned).query(np.asarray(source.triangles_center))[1]
    labels = smooth_labels(source, remesh_labels[nearest], smooth_iterations)
    labels, shells = shell_labels(source, labels, shell_share) if shell_share > 0 else (labels, 0)
    return labels, {"frame": name, "scale": scale, "mean_distance": distance,
                    "shells_relabelled": shells}


def load_single_textured(source_glb):
    """The source as one Trimesh (textured or flat-coloured), or None if it is several.

    An untextured source is still worth cutting: the remesh thickens thin geometry (a
    sword blade), which X-Part then rejects as a different shape from the source.
    """
    scene = trimesh.load(source_glb, force="scene")
    if len(scene.geometry) != 1:
        return None
    mesh = next(iter(scene.geometry.values())).copy()
    node = scene.graph.geometry_nodes[next(iter(scene.geometry))][0]
    transform, _ = scene.graph.get(node)
    mesh.apply_transform(transform)
    return mesh


def export_from_source(mesh_path, source_glb, labels_npy, names_json, out_glb,
                       smooth_iterations=DEFAULT_SMOOTH_ITERATIONS, step="[export]"):
    """parts.json-style manifest, or None if the source is not a single mesh."""
    from data_toolkit.parts_rebake import LABEL_COLORS, load_single_mesh

    source = load_single_textured(source_glb)
    if source is None:
        print(f"{step} source is not one mesh; exporting from the remesh instead")
        return None
    remesh = load_single_mesh(mesh_path)
    remesh_labels = np.load(labels_npy)
    if len(remesh_labels) != len(remesh.faces):
        raise SystemExit(f"{len(remesh_labels)} labels for {len(remesh.faces)} remesh faces")
    with open(names_json, "r", encoding="utf-8") as handle:
        names = json.load(handle)
    labels, fit = transfer_labels(remesh, remesh_labels, source, smooth_iterations)
    if not (labels >= 0).any():
        raise ValueError(
            "no part received any faces: SAM3 recognised none of the prompts and nothing "
            "absorbs the unclaimed faces. Try other words, pass unassigned_to, or "
            "merge=off for the geometric units.")
    print(f"{step} cutting the source model ({len(source.faces)} faces) by labels carried "
          f"over from the remesh ({len(remesh.faces)} faces; frame {fit['frame']}, "
          f"scale {fit['scale']:.4f}, fit {fit['mean_distance']:.4f}; "
          f"{fit.get('shells_relabelled', 0)} small shells given one label)")

    material = getattr(source.visual, "material", None)
    texture = getattr(material, "image", None)
    if texture is None:
        texture = getattr(material, "baseColorTexture", None)
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
    # submesh() copied the texture and PIL copies drop their format: without this the
    # source 8K JPEGs are re-encoded as ~23 MB PNGs (~10 s each)
    from glb_images import keep_jpeg, mesh_materials

    keep_jpeg(mesh_materials(scene), source_glb)
    scene.export(out_glb)
    with open(os.path.join(out_dir, "parts.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest
