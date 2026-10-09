"""Split a voted unit where SAM3's masks, read per face, name a coherent patch of it differently.

The pipeline lets geometry draw the boundaries and language only name them: each unit of
the over-segmentation takes one name from the multi-view vote. That is the right default,
and it fails in exactly one way -- when the geometry never separated two things a human
would. On the monk figure the back of each hand sits in the same unit as the gauntlet
cuff (no crease between glove and cuff), the unit votes "armor" 4:0, and the "body" hand
that reaches X-Part is missing its upper half.

The masks themselves know better: lifted per face (data_toolkit.lift_sam3), they colour
both fists "body" end to end. So after the vote, a unit is cut along the mask boundary
wherever a *coherent* patch of it -- connected, at least `min_share` of the unit's area --
consistently carries another part's name. Specks and single mislabelled faces stay with
the unit; only a patch worth a part gets split off.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

DEFAULT_MIN_SHARE = 0.05
DEFAULT_MIN_FACES = 50
# seg.glb already carries to_glb's baked axis swap: see data_toolkit/spike_lift.py.
SEG_GLB_ROTATION = np.diag([1.0, -1.0, -1.0])


def welded_adjacency(mesh):
    """Face adjacency after welding coincident vertices: seg.glb keeps a vertex per face
    corner, so the mesh's own face_adjacency is almost empty and no patch could ever
    reach min_faces."""
    import trimesh

    welded = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.faces), process=False)
    welded.merge_vertices(merge_tex=True, merge_norm=True)
    return welded.face_adjacency


def relabel_patches(mesh, units, labels, per_face, min_share=DEFAULT_MIN_SHARE,
                    min_faces=DEFAULT_MIN_FACES):
    """Give a coherent patch of a unit the label `per_face` says, if it is big enough.

    Returns (labels, changes): a copy of `labels` with the patches relabelled, and one
    row per patch that moved. A patch is a connected set of faces of one unit whose
    per-face label agrees with itself and disagrees with the unit's label.
    """
    units = np.asarray(units)
    labels = np.array(labels).copy()
    per_face = np.asarray(per_face)
    area = np.asarray(mesh.area_faces)
    disagree = (per_face >= 0) & (labels >= 0) & (per_face != labels)
    if not disagree.any():
        return labels, []
    adjacency = np.asarray(welded_adjacency(mesh))
    a, b = adjacency[:, 0], adjacency[:, 1]
    keep = (disagree[a] & disagree[b] & (units[a] == units[b]) & (per_face[a] == per_face[b]))
    edges = adjacency[keep]
    count = len(labels)
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(count, count))
    _, component = connected_components(graph, directed=False)
    component = np.where(disagree, component, -1)
    safe_units = np.where(units < 0, units.max() + 1, units)
    unit_area = np.bincount(safe_units, weights=area)
    changes = []
    for index in np.unique(component[component >= 0]):
        faces = np.flatnonzero(component == index)
        unit = safe_units[faces[0]]
        share = float(area[faces].sum() / max(unit_area[unit], 1e-12))
        if len(faces) < min_faces or share < min_share:
            continue
        old, new = int(labels[faces[0]]), int(per_face[faces[0]])
        labels[faces] = new
        changes.append({"unit": int(units[faces[0]]), "faces": int(len(faces)),
                        "share_of_unit": share, "from": old, "to": new})
    return labels, changes


def refine_labels_by_masks(mesh, units, labels, names, mask_set, cameras, camera_angle_x,
                           resolution, min_share=DEFAULT_MIN_SHARE,
                           min_faces=DEFAULT_MIN_FACES, smoothness=0.4):
    """Lift the masks per face and hand the vote's labels to relabel_patches.

    `names` orders the label indices (the vote's part order); the lift's own part order
    is matched to it by name, so a concept the vote never used cannot relabel anything.
    """
    from data_toolkit.lift_sam3 import lift

    part_labels, part_names, _, _ = lift(
        mesh, SEG_GLB_ROTATION, mask_set, cameras, camera_angle_x, resolution,
        smoothness=smoothness)
    to_label = {index: names.index(name) for index, name in enumerate(part_names)
                if name in names}
    per_face = np.array([to_label.get(int(label), -1) for label in part_labels])
    return relabel_patches(mesh, units, labels, per_face, min_share, min_faces)
