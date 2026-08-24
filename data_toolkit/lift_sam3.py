"""Lift multi-view SAM3 masks onto mesh faces by weighted voting.

The labels here come from SAM3 alone: every face is decided by the pixels that
actually covered it, so the result is a measurable function of the 2D masks rather
than something a generative model inferred from a single colour image.

Why voting instead of one authoritative view: SAM3 misses a concept in views where
it is edge-on or occluded (on the monk figure the staff is invisible from both
profiles), and its masks bleed across contact seams differently in every view.
Summing evidence over a fixed view grid makes both failure modes cost a fraction of
one view's weight instead of an entire part.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from data_toolkit.multiview import Camera, normalize_to_unit_cube, rasterize_face_ids

# Faces seen only at a grazing angle cover few pixels and straddle silhouette edges,
# so they are down-weighted -- but never to zero, or thin geometry viewed edge-on
# from every camera would end up with no evidence at all.
GRAZING_FLOOR = 0.05


@dataclass
class MaskSet:
    masks: np.ndarray      # bool [views, concepts, height, width]
    foreground: np.ndarray  # bool [views, height, width]
    scores: np.ndarray     # float [views, concepts]
    concepts: list[str]
    owners: list[str]      # output part each concept belongs to
    views: list[str]
    part_order: list[str]
    unassigned_to: str | None


def load_masks(path) -> MaskSet:
    data = np.load(path, allow_pickle=False)
    width = int(data["width"])
    unassigned = str(data["unassigned_to"])
    return MaskSet(
        masks=np.unpackbits(data["masks"], axis=-1, count=width).astype(bool),
        foreground=np.unpackbits(data["foreground"], axis=-1, count=width).astype(bool),
        scores=data["scores"].astype(np.float64),
        concepts=[str(v) for v in data["concepts"]],
        owners=[str(v) for v in data["owners"]],
        views=[str(v) for v in data["views"]],
        part_order=[str(v) for v in data["part_order"]],
        unassigned_to=unassigned or None,
    )


def load_cameras(views_dir):
    with open(os.path.join(views_dir, "cameras.json"), "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    cameras = [
        Camera(view["azimuth"], view["elevation"], np.array(view["position"], dtype=np.float64))
        for view in manifest["views"]
    ]
    return manifest, cameras


def view_weights(centroids, normals, cameras):
    """[views, faces] reliability of each face in each view, from viewing angle.

    The magnitude of the dot product is used because a face that won the depth test is
    a front surface by definition; a negative value only means its winding is inverted.
    """
    weights = np.empty((len(cameras), len(centroids)), dtype=np.float64)
    for index, camera in enumerate(cameras):
        direction = camera.position[None, :] - centroids
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        cosine = np.abs(np.einsum("ij,ij->i", normals, direction))
        weights[index] = np.maximum(cosine, GRAZING_FLOOR)
    return weights


def mask_lookup(mask_resolution, supersample):
    """Flat indices mapping a supersampled raster pixel to its mask pixel."""
    if supersample == 1:
        return None
    coarse = np.arange(mask_resolution * supersample) // supersample
    return (coarse[:, None] * mask_resolution + coarse[None, :]).ravel()


def accumulate_votes(face_ids, mask_set: MaskSet, weights, n_faces, supersample=1):
    """[faces, concepts] evidence, plus which faces any view actually resolved."""
    votes = np.zeros((n_faces, len(mask_set.concepts)), dtype=np.float64)
    visible = np.zeros(n_faces, dtype=bool)
    lookup = mask_lookup(mask_set.masks.shape[-1], supersample)

    for view in range(len(face_ids)):
        flat = face_ids[view].ravel()
        hit = flat > 0
        if not hit.any():
            continue
        faces = flat[hit] - 1
        angle = weights[view][faces]
        visible[faces] = True
        pixels = np.nonzero(hit)[0] if lookup is None else lookup[hit]

        for concept in range(len(mask_set.concepts)):
            score = mask_set.scores[view, concept]
            if score <= 0:
                continue
            inside = mask_set.masks[view, concept].ravel()[pixels]
            if not inside.any():
                continue
            votes[:, concept] += np.bincount(
                faces[inside], weights=angle[inside] * score, minlength=n_faces
            )
    return votes, visible


def fill_unlabelled(adjacency, labels, n_labels):
    """Grow labels into faces no view voted on (interior cavities, tight crevices)."""
    labels = np.asarray(labels).copy()
    adjacency = np.asarray(adjacency)
    if len(adjacency) == 0:
        return labels
    left, right = adjacency[:, 0], adjacency[:, 1]
    while True:
        blank = labels < 0
        if not blank.any():
            return labels
        votes = np.zeros((len(labels), n_labels), dtype=np.int32)
        known_right = labels[right] >= 0
        np.add.at(votes, (left[known_right], labels[right][known_right]), 1)
        known_left = labels[left] >= 0
        np.add.at(votes, (right[known_left], labels[left][known_left]), 1)
        reachable = blank & (votes.sum(axis=1) > 0)
        if not reachable.any():
            # A shell with no labelled face anywhere on it; only proximity can help.
            return labels
        labels[reachable] = votes[reachable].argmax(axis=1)


def fill_by_proximity(centroids, labels):
    """Label whatever no camera could reach from the nearest face that was observed.

    Remeshing routinely emits a second inner wall (on the monk figure it is half of
    all faces, and no view sees a single one of them). Those faces are never visible
    in a render either, but they still have to belong to a part or splitting the mesh
    drops them.
    """
    from scipy.spatial import cKDTree

    labels = np.asarray(labels).copy()
    blank = labels < 0
    if not blank.any() or blank.all():
        return labels
    tree = cKDTree(centroids[~blank])
    _, nearest = tree.query(centroids[blank])
    labels[blank] = labels[~blank][nearest]
    return labels


def welded_topology(mesh):
    """Adjacency, shared-edge lengths and dihedral angles over spatially welded vertices.

    glTF splits a vertex per UV corner, so the mesh's own adjacency sees a shattered
    surface. Welding is needed for the edge geometry too, not just the pairs, because
    the cut cost below is measured in real edge length and crease angle.
    """
    import trimesh

    positions, inverse = np.unique(np.asarray(mesh.vertices).round(6), axis=0, return_inverse=True)
    faces = inverse[np.asarray(mesh.faces)]
    if ((faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 0] == faces[:, 2])).any():
        # Face order has to keep lining up with the labels, so refuse to re-index.
        welded = mesh
    else:
        welded = trimesh.Trimesh(vertices=positions, faces=faces, process=False)
    adjacency = np.asarray(welded.face_adjacency)
    edges = np.asarray(welded.face_adjacency_edges)
    lengths = np.linalg.norm(
        np.asarray(welded.vertices)[edges[:, 0]] - np.asarray(welded.vertices)[edges[:, 1]], axis=1
    )
    return adjacency, lengths, np.asarray(welded.face_adjacency_angles), np.asarray(
        welded.face_adjacency_convex
    )


def regularize_boundaries(adjacency, weights, votes, labels, smoothness):
    """Shorten and straighten the seams by minimising vote loss plus cut length.

    Per-face voting follows SAM3's pixel-level disagreement between views, which reads
    as a torn edge even when every face is semantically plausible. Charging for each
    unit of cut makes a face give up a weak vote rather than leave a tooth sticking out,
    and because the charge is lower across concave creases the seam settles into the
    model's own folds instead of wandering across a smooth panel.

    Solved by iterated local minimisation: each pass moves faces whose neighbourhood
    makes a different label cheaper, which lowers the energy monotonically.
    """
    labels = np.asarray(labels).copy()
    if smoothness <= 0 or len(adjacency) == 0:
        return labels, 0.0

    n_labels = votes.shape[1]
    scale = float(votes.max()) or 1.0
    data = -votes / scale
    left, right = adjacency[:, 0], adjacency[:, 1]

    def cut_length():
        return float(weights[labels[left] != labels[right]].sum())

    before = cut_length()
    for _ in range(24):
        # Cost of each label per face: its own vote loss, plus what it would cost to
        # disagree with every neighbour.
        neighbour = np.zeros((len(labels), n_labels), dtype=np.float64)
        total = np.zeros(len(labels), dtype=np.float64)
        for source, target in ((left, right), (right, left)):
            np.add.at(neighbour, (source, labels[target]), weights)
            np.add.at(total, source, weights)
        cost = data + smoothness * (total[:, None] - neighbour)
        updated = cost.argmin(axis=1)
        if np.array_equal(updated, labels):
            break
        labels = updated
    return labels, before - cut_length()


def absorb_small_patches(adjacency, labels, areas, min_patch_ratio=0.001):
    """Give connected patches below `min_patch_ratio` of total area to their surroundings.

    Contact seams leave speckle: where SAM3's masks disagree between views, single
    triangles win a label their neighbours do not share. The threshold is an absolute
    share of the model's surface, deliberately not a share of the part it belongs to --
    a grouped part is legitimately made of disconnected regions (a hand is not attached
    to a leg), and comparing against the part's largest region would delete them.
    """
    from trimesh.graph import connected_components

    adjacency = np.asarray(adjacency)
    labels = np.asarray(labels).copy()
    if len(adjacency) == 0 or min_patch_ratio <= 0:
        return labels, 0

    threshold = min_patch_ratio * float(areas.sum())
    left, right = adjacency[:, 0], adjacency[:, 1]
    absorbed = 0
    for _ in range(8):
        same = labels[left] == labels[right]
        patches = connected_components(adjacency[same], nodes=np.arange(len(labels)))
        small = [patch for patch in patches if areas[patch].sum() < threshold]
        if not small:
            break
        patch_of = np.full(len(labels), -1, dtype=np.int64)
        for index, patch in enumerate(small):
            patch_of[patch] = index

        # Each small patch takes the label its neighbours outside it agree on, weighted
        # by the shared boundary's area so the dominant surround wins.
        border = ~same
        votes = {}
        for a, b in adjacency[border]:
            for inside, outside in ((a, b), (b, a)):
                index = patch_of[inside]
                if index < 0 or patch_of[outside] == index:
                    continue
                votes.setdefault(index, {}).setdefault(labels[outside], 0.0)
                votes[index][labels[outside]] += areas[outside]
        if not votes:
            break
        for index, tally in votes.items():
            labels[small[index]] = max(tally.items(), key=lambda item: item[1])[0]
            absorbed += 1
    return labels, absorbed


def prepare_mesh(mesh, rotation):
    """Rotate a mesh into the Blender frame the cameras live in, then unit-normalise."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64) @ np.asarray(rotation).T
    vertices, _ = normalize_to_unit_cube(vertices)
    faces = np.asarray(mesh.faces)
    centroids = vertices[faces].mean(axis=1)
    return vertices, faces, centroids


def part_votes_from_concepts(votes, owners):
    """Collapse concept evidence onto the parts that will actually be exported.

    The strongest concept wins rather than the sum, so a part built from concepts that
    overlap each other (SAM3's 'head' and 'face' cover much of the same pixels) is not
    credited twice for the same evidence.
    """
    parts = list(dict.fromkeys(owners))
    merged = np.zeros((len(votes), len(parts)), dtype=np.float64)
    for index, part in enumerate(parts):
        columns = [i for i, owner in enumerate(owners) if owner == part]
        merged[:, index] = votes[:, columns].max(axis=1)
    return merged, parts


def segvigen_seam_discount(mesh, adjacency, weights, confidence_path, gain,
                           hops=4, decay=0.8, grid=512):
    """Make a cut cheap where SegviGen's own part masks change, and leave the rest alone.

    The two approaches fail in opposite ways. SAM3's votes get the semantics right -- it can
    tell a robe from the skin under it, which is an appearance question no shape model can
    answer -- but its per-view disagreement lands on the surface as a ragged edge. SegviGen's
    interactive masks come out smooth and follow the model's own structure, yet they cannot
    express a grouped part and freely claim their neighbours. So the votes keep deciding
    *which* part a face belongs to, and SegviGen only gets a say in *where* the seam runs.

    The confidence volume is indexed in the input glb's own coordinates, so each face is
    matched by looking up the voxel under its centroid; no correspondence between this mesh
    and SegviGen's remesh is needed, which is what keeps the original topology and UVs.
    """
    from data_toolkit.multiview import normalize_to_unit_cube

    stored = np.load(confidence_path)
    coords, confidence = stored["coords"].astype(np.int64), stored["confidence"]
    keys = (coords[:, 0] * grid + coords[:, 1]) * grid + coords[:, 2]
    order = np.argsort(keys)
    keys, confidence = keys[order], confidence[order]

    vertices, _ = normalize_to_unit_cube(np.asarray(mesh.vertices, dtype=np.float64))
    centroids = vertices[np.asarray(mesh.faces)].mean(axis=1)
    voxels = np.clip(np.floor((centroids + 0.5) * grid).astype(np.int64), 0, grid - 1)
    probe = (voxels[:, 0] * grid + voxels[:, 1]) * grid + voxels[:, 2]

    position = np.clip(np.searchsorted(keys, probe), 0, len(keys) - 1)
    found = keys[position] == probe
    per_face = np.zeros((len(centroids), confidence.shape[1]), dtype=np.float64)
    per_face[found] = confidence[position[found]]

    left, right = adjacency[:, 0], adjacency[:, 1]
    # How strongly SegviGen disagrees about the two faces across this edge, over all parts.
    change = np.abs(per_face[left] - per_face[right]).max(axis=1)
    change[~(found[left] & found[right])] = 0.0

    # The masks are all but binary, so they change over a single voxel -- and on a mesh this
    # dense a voxel is about one triangle wide, which discounts too narrow a band for a seam
    # to find. Spreading it a few faces out turns the boundary into a channel the seam can
    # settle into anywhere along its width, which is the whole point: the votes say which
    # part, SegviGen's boundary says where within that freedom the cut should run.
    nearness = np.zeros(len(per_face))
    np.maximum.at(nearness, left, change)
    np.maximum.at(nearness, right, change)
    for _ in range(hops):
        spread = nearness.copy()
        np.maximum.at(spread, left, nearness[right] * decay)
        np.maximum.at(spread, right, nearness[left] * decay)
        nearness = spread

    # Both ends have to be inside the channel, or the discount would leak outwards each hop.
    discounted = weights * np.exp(-gain * np.minimum(nearness[left], nearness[right]))
    mean = float(discounted.mean())
    report = {
        "faces_found_in_confidence_volume": int(found.sum()),
        "faces_near_a_segvigen_boundary": int((nearness > 0.5).sum()),
        "seam_edges_discounted_by_segvigen": int((np.minimum(nearness[left], nearness[right]) > 0.5).sum()),
    }
    return (discounted / mean if mean > 0 else discounted), report


def cut_weights(lengths, angles, convex, crease_gain=2.0):
    """Cost per unit of seam, discounted where the surface already creases.

    A seam that follows a fold reads as a deliberate cut; the same seam across a smooth
    panel reads as damage. Concave folds are where parts genuinely meet, so they are
    discounted hardest.
    """
    sharpness = np.asarray(angles) * np.where(np.asarray(convex), 1.0, 2.0)
    weights = np.asarray(lengths) * np.exp(-crease_gain * sharpness)
    # Normalised to mean 1 so `smoothness` is a dimensionless trade-off against the vote
    # loss, instead of something that has to be retuned for every mesh's triangle size.
    mean = float(weights.mean())
    return weights / mean if mean > 0 else weights


def lift(mesh, rotation, mask_set: MaskSet, cameras, camera_angle_x, resolution,
         smooth_iterations=2, supersample=4, min_patch_ratio=0.001, smoothness=0.0,
         crease_gain=2.0, confidence_path=None, confidence_gain=4.0, confidence_hops=4):
    """Return per-face part labels, the part names and a coverage report.

    `supersample` rasterises face ids finer than the SAM3 masks. A remesh's triangles
    are routinely smaller than a pixel -- on the monk figure 97k faces compete for
    ~150k silhouette pixels -- so at 1x a tenth of the visible surface wins no pixel
    at all and has to be guessed instead of read off a mask.

    `smoothness` trades semantic fidelity for intact boundaries: at 0 every face keeps
    whichever label the masks voted for, teeth and all; raising it charges for seam
    length until the cut follows the model's own folds.
    """
    from data_toolkit.parts_rebake import smooth_labels

    vertices, faces, centroids = prepare_mesh(mesh, rotation)
    adjacency, lengths, angles, convex = welded_topology(mesh)
    face_ids = rasterize_face_ids(
        vertices, faces, cameras, camera_angle_x, resolution=resolution * supersample
    )
    weights = view_weights(centroids, np.asarray(mesh.face_normals), cameras)
    votes, visible = accumulate_votes(
        face_ids, mask_set, weights, len(faces), supersample=supersample
    )
    # Merge to parts before anything is cleaned up: a seam between two concepts of the
    # same part does not exist in the output, so there is nothing there worth tidying.
    votes, part_names = part_votes_from_concepts(votes, mask_set.owners)

    labels = np.where(votes.sum(axis=1) > 0, votes.argmax(axis=1), -1)
    voted = int((labels >= 0).sum())
    labels = fill_unlabelled(adjacency, labels, len(part_names))
    propagated = int((labels >= 0).sum()) - voted
    labels = fill_by_proximity(centroids, labels)
    labels = smooth_labels(adjacency, labels, len(part_names), smooth_iterations)

    seam_weights = cut_weights(lengths, angles, convex, crease_gain)
    confidence_report = {}
    if confidence_path:
        seam_weights, confidence_report = segvigen_seam_discount(
            mesh, adjacency, seam_weights, confidence_path, confidence_gain, confidence_hops
        )
    labels, shortened = regularize_boundaries(adjacency, seam_weights, votes, labels, smoothness)
    labels, absorbed = absorb_small_patches(
        adjacency, labels, np.asarray(mesh.area_faces), min_patch_ratio
    )

    report = {
        **confidence_report,
        "faces": int(len(faces)),
        "faces_resolved_by_a_view": int(visible.sum()),
        "faces_with_votes": voted,
        "faces_filled_along_surface": propagated,
        "faces_filled_by_proximity": int(len(faces) - voted - propagated),
        "adjacency_pairs": int(len(adjacency)),
        "speckle_patches_absorbed": absorbed,
        "seam_cost_removed": round(shortened, 4),
    }
    return labels, part_names, face_ids, report
