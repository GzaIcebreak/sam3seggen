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
# a face's per-face label counts as evidence only if this many views' masks back it: on
# the puppy one rear view's torso mask covered the back of the head, and refine moved an
# ear's worth of the head to the torso on that view alone
DEFAULT_MIN_VIEWS = 2
# a move is vetoed when, in the views that support it, at least this share of the
# patch's pixels has the old part's mask both left and right of it on the image row
# (the back of a dog's head between its two ears, which the rear views call torso)
FLANKED_SHARE = 0.5
SUPPORT_SHARE = 0.3       # a view supports a move if the new part's mask covers this much of the patch
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


def flanked_share(patch, old):
    """Share of the patch's pixels lying on image rows where the old part's mask shows
    up both left and right of the patch. `patch`, `old`: [H, W] bool."""
    rows = np.flatnonzero(patch.any(axis=1))
    total = flanked = 0
    for row in rows:
        cols = np.flatnonzero(patch[row])
        total += len(cols)
        if old[row, :cols[0]].any() and old[row, cols[-1] + 1:].any():
            flanked += len(cols)
    return flanked / total if total else 0.0


def patch_flanked_by(faces, old_name, new_name, face_ids, mask_set, support=SUPPORT_SHARE):
    """Mean flanked_share over the views whose `new_name` mask covers the patch, or None
    when no view supports the move. `face_ids` is lift's raster (face index + 1, 0 =
    background), possibly supersampled relative to the masks."""
    from data_toolkit.lift_sam3 import mask_lookup

    if face_ids is None or mask_set is None:
        return None
    resolution = mask_set.masks.shape[-1]
    lookup = mask_lookup(resolution, face_ids.shape[-1] // resolution)
    member = np.zeros(int(face_ids.max()) + 1, dtype=bool)
    member[np.asarray(faces) + 1] = True
    shares = []
    for view in range(len(face_ids)):
        hit = np.flatnonzero(member[face_ids[view].ravel()])
        if not len(hit):
            continue
        patch = np.zeros(resolution * resolution, dtype=bool)
        patch[hit if lookup is None else lookup[hit]] = True
        patch = patch.reshape(resolution, resolution)
        live = mask_set.scores[view] > 0
        old = np.zeros_like(patch)
        new = np.zeros_like(patch)
        for concept, owner in enumerate(mask_set.owners):
            if not live[concept]:
                continue
            if owner == old_name:
                old |= mask_set.masks[view, concept]
            elif owner == new_name:
                new |= mask_set.masks[view, concept]
        if (new & patch).sum() < support * patch.sum():
            continue
        shares.append(flanked_share(patch, old))
    return float(np.mean(shares)) if shares else None


def relabel_patches(mesh, units, labels, per_face, min_share=DEFAULT_MIN_SHARE,
                    min_faces=DEFAULT_MIN_FACES, veto=None):
    """Give a coherent patch of a unit the label `per_face` says, if it is big enough.

    Returns (labels, changes): a copy of `labels` with the patches relabelled, and one
    row per patch that moved. A patch is a connected set of faces of one unit whose
    per-face label agrees with itself and disagrees with the unit's label.
    `veto(faces, old, new)` may keep a patch where it is by returning True.
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
        if veto is not None and veto(faces, old, new):
            continue
        labels[faces] = new
        changes.append({"unit": int(units[faces[0]]), "faces": int(len(faces)),
                        "share_of_unit": share, "from": old, "to": new})
    return labels, changes


def refine_labels_by_masks(mesh, units, labels, names, mask_set, cameras, camera_angle_x,
                           resolution, min_share=DEFAULT_MIN_SHARE,
                           min_faces=DEFAULT_MIN_FACES, smoothness=0.4,
                           min_views=DEFAULT_MIN_VIEWS):
    """Lift the masks per face and hand the vote's labels to relabel_patches.

    `names` orders the label indices (the vote's part order); the lift's own part order
    is matched to it by name, so a concept the vote never used cannot relabel anything.
    """
    from data_toolkit.lift_sam3 import lift

    # keep_unvoted=False: a face no mask covered carries a label grown from a neighbour,
    # which is not evidence -- on the lying cat a head label grown from one view's tail
    # tip took the hips and hind legs the vote had given to the torso
    part_labels, part_names, face_ids, _ = lift(
        mesh, SEG_GLB_ROTATION, mask_set, cameras, camera_angle_x, resolution,
        smoothness=smoothness, keep_unvoted=False, min_views=min_views)
    to_label = {index: names.index(name) for index, name in enumerate(part_names)
                if name in names}
    per_face = np.array([to_label.get(int(label), -1) for label in part_labels])

    def flanked(faces, old, new):
        share = patch_flanked_by(faces, names[old], names[new], face_ids, mask_set)
        if share is not None and share >= FLANKED_SHARE:
            print(f"  unit {int(units[faces[0]]):>3}: kept {len(faces)} faces with {names[old]} -- "
                  f"in the views that call them {names[new]}, {names[old]}'s mask flanks them "
                  f"left and right ({share:.0%})")
            return True
        return False

    return relabel_patches(mesh, units, labels, per_face, min_share, min_faces, veto=flanked)
