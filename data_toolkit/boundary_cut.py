"""Cut an under-segmented atom, with SAM3 saying where and the mesh saying exactly where.

meet_samples can only keep boundaries some full_seg sample drew. When every sample misses
one -- on the robot test model the right ankle was welded shut in all five, while the left
one was cut -- no amount of voting recovers it, because naming only ever merges atoms.

The vote itself detects this: an atom is under-segmented exactly when a second concept
claims a large, coherent part of it (the fused shin+foot was claimed by `leg` at 0.96
coverage and by `foot` at 0.39). What it cannot do is place the boundary, and painting
faces straight from the 2D masks is what tore boundaries apart in the 2D-guided pipeline:
mask edges wobble by a few pixels per view and cut across flat surfaces.

So the boundary is placed by a min cut on the atom's face graph:

* the data term is one concept's per-face mask coverage -- SAM3 votes for *whether* a face
  belongs to the challenger, pooled over every view that saw it;
* the smoothness term is the mesh's own geometry -- cutting across a concave crease is
  nearly free, cutting across a flat or convex region is expensive.

The result is a boundary that runs along the seam a modeller would have used, in the
region SAM3 pointed at. Only the *partition* is taken from the cut; the pieces are named
by the usual vote afterwards, so this stage cannot mislabel anything, only split.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, maximum_flow

DEFAULT_LAMBDA = 1.0
# A concave crease this sharp (radians) costs e^-1 of a flat edge to cut through.
DEFAULT_CREASE_SCALE = 0.5
DEFAULT_MIN_CHALLENGER_RECALL = 0.25
DEFAULT_MAX_CHALLENGER_RECALL = 0.9
CAPACITY_SCALE = 1000


def crease_weights(mesh, adjacency, crease_scale=DEFAULT_CREASE_SCALE):
    """Cost of cutting each adjacency edge: short and concave is cheap, long and flat dear.

    Convex folds are left at full price. A part boundary on a hard-surface model is a
    concave seam (the shin meets the foot in a valley); a convex fold is usually a feature
    line *inside* one part, and cutting there is how a split ends up looking wrong.
    """
    positions, welded = np.unique(np.asarray(mesh.vertices).round(6), axis=0,
                                  return_inverse=True)
    welded_faces = welded.ravel()[np.asarray(mesh.faces)]
    centers = np.asarray(mesh.triangles_center)

    left, right = welded_faces[adjacency[:, 0]], welded_faces[adjacency[:, 1]]
    shared = (left[:, :, None] == right[:, None, :]).any(axis=2)
    # Two triangles sharing an edge share exactly two corners; a degenerate pair that
    # shares three keeps the first two, which measures the same edge.
    order = np.argsort(~shared, axis=1, kind="stable")[:, :2]
    corners = np.take_along_axis(left, order, axis=1)
    length = np.linalg.norm(positions[corners[:, 0]] - positions[corners[:, 1]], axis=1)
    # Pairs that only meet at a point come from the non-manifold branch of the adjacency;
    # measure those by how far apart the faces are instead.
    lonely = shared.sum(axis=1) < 2
    if lonely.any():
        length[lonely] = np.linalg.norm(
            centers[adjacency[lonely, 1]] - centers[adjacency[lonely, 0]], axis=1)

    normals = np.asarray(mesh.face_normals)
    left_normal, right_normal = normals[adjacency[:, 0]], normals[adjacency[:, 1]]
    angle = np.arccos(np.clip((left_normal * right_normal).sum(axis=1), -1.0, 1.0))
    direction = centers[adjacency[:, 1]] - centers[adjacency[:, 0]]
    concave = (left_normal * direction).sum(axis=1) > 0
    crease = np.where(concave, angle, 0.0)
    return length / max(length.mean(), 1e-9) * np.exp(-crease / crease_scale)


def face_mask_coverage(face_ids, n_faces, mask_set):
    """(coverage[face, concept] in [0,1], pixels per face) pooled over every view."""
    n_concepts = len(mask_set.concepts)
    pixels = np.zeros(n_faces)
    votes = np.zeros((n_faces, n_concepts))
    for view in range(face_ids.shape[0]):
        flat = face_ids[view].ravel()
        hit = flat > 0
        seen = flat[hit] - 1
        pixels += np.bincount(seen, minlength=n_faces)
        for concept in range(n_concepts):
            if mask_set.scores[view, concept] <= 0:
                continue
            covered = mask_set.masks[view, concept].ravel()[hit]
            votes[:, concept] += np.bincount(seen[covered], minlength=n_faces)
    return votes / np.maximum(pixels[:, None], 1), pixels


def min_cut(unary, pairs, weights):
    """Boolean label per node minimising sum(unary) + sum(weights over cut edges).

    `unary` is [n, 2]: the cost of giving a node label False and label True.
    """
    n = len(unary)
    source, sink = n, n + 1
    cost = np.rint(np.asarray(unary) * CAPACITY_SCALE).astype(np.int32)
    edge = np.rint(np.asarray(weights) * CAPACITY_SCALE).astype(np.int32)
    rows = np.concatenate([np.full(n, source), np.arange(n), pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([np.arange(n), np.full(n, sink), pairs[:, 1], pairs[:, 0]])
    data = np.concatenate([cost[:, 1], cost[:, 0], edge, edge])
    graph = csr_matrix((np.maximum(data, 0), (rows, cols)), shape=(n + 2, n + 2),
                       dtype=np.int32)
    flow = maximum_flow(graph, source, sink).flow
    residual = (graph - flow).tocsr()
    residual.data = np.where(residual.data > 0, residual.data, 0)
    residual.eliminate_zeros()
    reachable = breadth_first_order(residual, source, directed=True,
                                    return_predecessors=False)
    label = np.ones(n, dtype=bool)
    label[reachable[reachable < n]] = False
    return label


def cut_atom(faces, coverage, seen, pairs, weights, lam=DEFAULT_LAMBDA):
    """Which of an atom's faces belong to the challenger, per the cut. `pairs` are local."""
    probability = np.clip(coverage[faces], 0.01, 0.99)
    unary = np.stack([-np.log(1.0 - probability), -np.log(probability)], axis=1)
    # A face no camera saw votes for nothing; the creases decide which side it lands on.
    unary[~seen[faces]] = 0.0
    return min_cut(unary, pairs, lam * weights)


def local_pairs(adjacency, faces, weights):
    """Adjacency rows with both ends inside `faces`, reindexed to 0..len(faces)-1."""
    member = np.zeros(int(adjacency.max()) + 2, dtype=bool)
    member[faces] = True
    inside = member[adjacency[:, 0]] & member[adjacency[:, 1]]
    return np.searchsorted(faces, adjacency[inside]), weights[inside]
