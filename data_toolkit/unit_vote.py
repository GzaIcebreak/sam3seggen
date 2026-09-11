"""Name over-segmented atoms from multi-view SAM3 masks: unit-level IoU voting.

Input is a face-labelled reference mesh (atoms from meet_samples.py) plus the raw
per-view masks sam3_multiview.py saved. Output is one part name per face. Faces are
never cut inside a unit: the vote can only merge atoms, never redraw a boundary.

Four decisions that matter, all measured on the robot test case:

* units are the welded connected components of each atom (>= `min_unit_faces`), so an
  atom SegviGen painted one colour but that is physically two pieces can take two names;
* coverage (recall) decides whether a mask claims a unit at all, but among the masks that
  do, the most specific one wins (highest IoU) -- plain coverage lets superset concepts
  win (leg over foot, arm over hand) because the larger mask always covers the smaller
  unit completely, while plain IoU starves small units of an over-segmentation, whose
  IoU with any part-sized mask is low by construction;
* that decision is made *per view*, and each view where the unit is big enough to judge
  casts one vote. Pooling pixels across views instead lets whichever view happens to see
  the unit head-on outvote all the others, which is exactly the view where a part half
  hidden behind another one gets its neighbour's name;
* the remesh's inner walls are invisible to every camera and would otherwise all fall to
  the same name; hidden units inherit the label of the nearest visible face instead.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree
from trimesh.graph import connected_components

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_toolkit.lift_sam3 import load_cameras, load_masks  # noqa: E402
from data_toolkit.multiview import normalize_to_unit_cube, rasterize_face_ids  # noqa: E402
from data_toolkit.parts_rebake import load_single_mesh, welded_face_adjacency  # noqa: E402

# seg.glb carries to_glb's baked axis swap; on top of glTF->Blender that is a half turn
# about X. Measured by spike_align_check.py, reused by spike_lift.py.
SEG_TO_CAMERA = np.diag([1.0, -1.0, -1.0])

DEFAULT_MIN_UNIT_FACES = 300
DEFAULT_MIN_RECALL = 0.5
DEFAULT_MIN_VISIBLE_PIXELS = 50


def split_units(mesh, atoms, adjacency, min_unit_faces=DEFAULT_MIN_UNIT_FACES):
    """Unit id per face: welded connected components inside each atom.

    Components under `min_unit_faces` are not units of their own; they join the nearest
    big component of the same atom (an atom made only of small pieces stays one unit).
    """
    units = np.full(len(atoms), -1, dtype=np.int64)
    centroids = mesh.triangles_center
    next_unit = 0
    for atom in np.unique(atoms):
        faces = np.flatnonzero(atoms == atom)
        inside = adjacency[(atoms[adjacency[:, 0]] == atom) & (atoms[adjacency[:, 1]] == atom)]
        local_edges = np.searchsorted(faces, inside)
        components = connected_components(local_edges, nodes=np.arange(len(faces)))
        big = [c for c in components if len(c) >= min_unit_faces]
        small = [c for c in components if len(c) < min_unit_faces]
        if not big:
            big, small = [np.concatenate(components)], []
        unit_local = np.full(len(faces), -1, dtype=np.int64)
        for index, component in enumerate(big):
            unit_local[component] = next_unit + index
        if small:
            anchors = np.flatnonzero(unit_local >= 0)
            tree = cKDTree(centroids[faces[anchors]])
            for component in small:
                _, nearest = tree.query(centroids[faces[component]].mean(axis=0))
                unit_local[component] = unit_local[anchors[nearest]]
        units[faces] = unit_local
        next_unit += len(big)
    return fuse_inner_shells(mesh, units)


def _outwardness(mesh, faces):
    """Share of faces whose normal points away from the component's own centroid."""
    centroids = mesh.triangles_center[faces]
    center = centroids.mean(axis=0)
    return float(((mesh.face_normals[faces] * (centroids - center)).sum(axis=1) > 0).mean())


def fuse_inner_shells(mesh, units, centroid_frac=0.08, contain=0.9, area_lo=0.4, area_hi=1.6):
    """Merge a remesh inner wall into the outer shell it sits inside.

    SegviGen's remesh is a thick shell: an outward-facing outer surface and an
    inward-facing inner wall that share no edges. They become two units and, if they
    vote separately, the hidden inner one inherits whoever happens to sit nearest --
    which on the robot was an arm, so the backpack's lining turned the whole back
    into an arm after the name merge. Pairing them first makes that inheritance a
    no-op: both faces already have the same unit id.
    """
    n_units = int(units.max()) + 1
    if n_units < 2:
        return units
    diagonal = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    stats = []
    for unit in range(n_units):
        faces = np.flatnonzero(units == unit)
        corners = mesh.vertices[np.unique(mesh.faces[faces])]
        stats.append({
            "faces": faces,
            "centroid": mesh.triangles_center[faces].mean(axis=0),
            "lo": corners.min(axis=0),
            "hi": corners.max(axis=0),
            "area": float(mesh.area_faces[faces].sum()),
            "outward": _outwardness(mesh, faces),
        })
    outer = [i for i, row in enumerate(stats) if row["outward"] >= 0.7]
    inner = [i for i, row in enumerate(stats) if row["outward"] <= 0.3]
    remap = np.arange(n_units)
    claimed = set()
    fused = 0
    for inner_id in sorted(inner, key=lambda i: -stats[i]["area"]):
        candidate, best = None, diagonal
        row = stats[inner_id]
        volume = float(np.prod(np.maximum(row["hi"] - row["lo"], 1e-9)))
        for outer_id in outer:
            if outer_id in claimed:
                continue
            other = stats[outer_id]
            distance = float(np.linalg.norm(row["centroid"] - other["centroid"]))
            if distance > centroid_frac * diagonal:
                continue
            overlap = float(np.prod(np.maximum(
                0.0, np.minimum(row["hi"], other["hi"]) - np.maximum(row["lo"], other["lo"]))))
            if overlap < contain * volume:
                continue
            ratio = row["area"] / max(other["area"], 1e-9)
            if ratio < area_lo or ratio > area_hi:
                continue
            if distance < best:
                candidate, best = outer_id, distance
        if candidate is not None:
            remap[inner_id] = candidate
            claimed.add(candidate)
            fused += 1
    if not fused:
        return units
    merged = remap[units]
    _, compact = np.unique(merged, return_inverse=True)
    print(f"fused {fused} remesh inner shells into their outer unit "
          f"({n_units} -> {compact.max() + 1} units)")
    return compact


def unit_mask_overlap(face_ids, unit_of_face, mask_set):
    """Pixel counts kept per view: (inter[view, unit, concept], unit_pixels, mask_pixels)."""
    n_views = face_ids.shape[0]
    n_units = int(unit_of_face.max()) + 1
    n_concepts = len(mask_set.concepts)
    inter = np.zeros((n_views, n_units, n_concepts))
    unit_pixels = np.zeros((n_views, n_units))
    mask_pixels = np.zeros((n_views, n_concepts))
    for view in range(n_views):
        flat = face_ids[view].ravel()
        hit = flat > 0
        pixel_unit = unit_of_face[flat[hit] - 1]
        np.add.at(unit_pixels[view], pixel_unit, 1)
        for concept in range(n_concepts):
            if mask_set.scores[view, concept] <= 0:
                continue
            mask = mask_set.masks[view, concept].ravel()[hit]
            mask_pixels[view, concept] = mask.sum()
            np.add.at(inter[view, :, concept], pixel_unit[mask], 1)
    return inter, unit_pixels, mask_pixels


def score_units(inter, unit_pixels, mask_pixels, owners):
    """Collapse concepts onto part names (best concept wins) and return (names, recall, iou).

    Shapes are passed straight through, so this works on one view or on a whole stack.
    """
    names = list(dict.fromkeys(owners))
    columns = {name: [i for i, owner in enumerate(owners) if owner == name] for name in names}
    inter_n = np.stack([inter[..., columns[n]].max(axis=-1) for n in names], axis=-1)
    mask_n = np.stack([mask_pixels[..., columns[n]].max(axis=-1) for n in names], axis=-1)
    recall = inter_n / np.maximum(unit_pixels[..., None], 1)
    iou = inter_n / np.maximum(unit_pixels[..., None] + mask_n[..., None, :] - inter_n, 1)
    return names, recall, iou


def tally_votes(recall, iou, unit_pixels, min_recall=DEFAULT_MIN_RECALL,
                min_visible_pixels=DEFAULT_MIN_VISIBLE_PIXELS):
    """Per-view winners -> (votes[unit, name], iou summed over the views that voted).

    A view only votes on units it sees enough of, and it votes for the most specific mask
    that covers the unit there. A part that one camera sees fused with its neighbour is
    then outvoted by the cameras that see it separately.
    """
    counts = np.zeros(recall.shape[1:], dtype=np.int64)
    weight = np.zeros(recall.shape[1:])
    for view in range(recall.shape[0]):
        judging = unit_pixels[view] >= min_visible_pixels
        claiming = (recall[view] >= min_recall) & judging[:, None]
        voters = np.flatnonzero(claiming.any(axis=1))
        winners = np.where(claiming[voters], iou[view][voters], -1.0).argmax(axis=1)
        counts[voters, winners] += 1
        weight[voters, winners] += iou[view][voters, winners]
    return counts, weight


def assign_units(mesh, unit_of_face, names, votes, weight, seen, unassigned_to):
    """One name (or None) per unit. Hidden units copy the nearest visible face's name."""
    assignment = [None] * len(seen)
    for unit in np.flatnonzero(seen):
        if votes[unit].any():
            # ties go to the name the views that voted for it fitted best
            best = int(np.lexsort((weight[unit], votes[unit]))[-1])
            assignment[unit] = names[best]
        else:
            assignment[unit] = unassigned_to
    visible = np.asarray(seen, dtype=bool)
    if (~visible).any():
        centroids = mesh.triangles_center
        seen_faces = np.flatnonzero(visible[unit_of_face])
        if len(seen_faces) == 0:
            raise ValueError("no unit is visible from any camera; check the view grid")
        tree = cKDTree(centroids[seen_faces])
        for unit in np.flatnonzero(~visible):
            faces = np.flatnonzero(unit_of_face == unit)
            _, nearest = tree.query(centroids[faces])
            labels = [assignment[u] for u in unit_of_face[seen_faces[nearest]]]
            assignment[unit] = max(set(labels), key=labels.count)
    return assignment, visible


def silhouette_agreement(face_ids, foreground):
    """Per-view IoU between the rasterised mesh and the silhouette SAM3 saw.

    The masks are painted on renders of one model and back-projected onto another (the
    source glb vs SegviGen's remesh), so a low value here means the two are not framed
    alike and every vote below is measuring the wrong pixels.
    """
    hit = face_ids > 0
    intersection = (hit & foreground).sum(axis=(1, 2))
    union = (hit | foreground).sum(axis=(1, 2))
    return np.divide(intersection, union, out=np.zeros(len(union)), where=union > 0)


def vote(mesh, atoms, mask_set, cameras, camera_angle_x, resolution, part_order,
         unassigned_to=None, min_unit_faces=DEFAULT_MIN_UNIT_FACES, min_recall=DEFAULT_MIN_RECALL,
         min_visible_pixels=DEFAULT_MIN_VISIBLE_PIXELS, units=None):
    """Return (face labels indexed into `part_order`, -1 for unnamed; report rows; units).

    `units` is the split's own shell-fused unit id per face. Pass it whenever the caller
    already ran the split: recomputing it here would be both slower and a chance for the
    names to be voted onto a different partition than the one that was exported.
    """
    unit_of_face = units
    if unit_of_face is None:
        unit_of_face = split_units(mesh, atoms, welded_face_adjacency(mesh), min_unit_faces)
    vertices, _ = normalize_to_unit_cube(np.asarray(mesh.vertices) @ SEG_TO_CAMERA.T)
    face_ids = rasterize_face_ids(vertices, np.asarray(mesh.faces), cameras, camera_angle_x, resolution)
    agreement = silhouette_agreement(face_ids, mask_set.foreground)
    print(f"silhouette agreement with the renders: min {agreement.min():.3f}, "
          f"mean {agreement.mean():.3f}")
    if agreement.min() < 0.8:
        print("  warning: the masks and the mesh are not framed alike; votes will be noisy")
    inter, unit_pixels, mask_pixels = unit_mask_overlap(face_ids, unit_of_face, mask_set)
    names, recall, iou = score_units(inter, unit_pixels, mask_pixels, list(mask_set.owners))
    votes, weight = tally_votes(recall, iou, unit_pixels, min_recall, min_visible_pixels)
    seen = (unit_pixels >= min_visible_pixels).sum(axis=0)
    assignment, visible = assign_units(mesh, unit_of_face, names, votes, weight, seen,
                                       unassigned_to)

    index_of = {name: i for i, name in enumerate(part_order)}
    labels = np.full(len(atoms), -1, dtype=np.int64)
    rows = []
    for unit, pick in enumerate(assignment):
        faces = np.flatnonzero(unit_of_face == unit)
        rows.append({
            "unit": int(unit),
            "atom": int(atoms[faces[0]]),
            "faces": int(len(faces)),
            "views_seen": int(seen[unit]),
            "pixels": int(unit_pixels[:, unit].sum()),
            "votes": {n: int(votes[unit, i]) for i, n in enumerate(names)},
            "name": pick,
        })
        if pick is not None:
            labels[faces] = index_of[pick]
    return labels, rows, unit_of_face


def print_report(rows, names):
    print(f"{'unit':>5} {'atom':>5} {'faces':>6} {'views':>6}  "
          + "  ".join(f"{n:>6}" for n in names) + "   -> name")
    for row in rows:
        votes = "  ".join(f"{row['votes'][n]:>6}" for n in names)
        tag = "" if row["views_seen"] else " (hidden->nearest)"
        print(f"{row['unit']:>5} {row['atom']:>5} {row['faces']:>6} {row['views_seen']:>6}  "
              f"{votes}   -> {row['name']}{tag}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mesh", required=True, help="reference seg.glb the atom labels index")
    parser.add_argument("--atoms", required=True, help="npy atom id per face (meet_samples.py)")
    parser.add_argument("--views_dir", required=True, help="render_multiview.py output (cameras.json)")
    parser.add_argument("--masks", required=True, help="sam3_multiview.py .npz")
    parser.add_argument("--out_labels", required=True, help="npy part label per face (-1 = unnamed)")
    parser.add_argument("--out_names", required=True, help="json list naming the labels")
    parser.add_argument("--report", default=None, help="json per-unit vote report")
    parser.add_argument("--min_unit_faces", type=int, default=DEFAULT_MIN_UNIT_FACES)
    parser.add_argument("--min_recall", type=float, default=DEFAULT_MIN_RECALL,
                        help="a mask claims a unit once it covers this share of the unit's pixels")
    parser.add_argument("--min_visible_pixels", type=int, default=DEFAULT_MIN_VISIBLE_PIXELS,
                        help="a view only votes on units it shows at least this many pixels of")
    args = parser.parse_args()

    mesh = load_single_mesh(os.path.abspath(args.mesh))
    atoms = np.load(os.path.abspath(args.atoms))
    if len(atoms) != len(mesh.faces):
        raise SystemExit(f"{len(atoms)} atom labels for {len(mesh.faces)} faces")
    mask_set = load_masks(os.path.abspath(args.masks))
    manifest, cameras = load_cameras(os.path.abspath(args.views_dir))
    part_order = list(mask_set.part_order)
    unassigned_to = mask_set.unassigned_to or None

    labels, rows, _ = vote(mesh, atoms, mask_set, cameras, float(manifest["camera_angle_x"]),
                           int(manifest["resolution"]), part_order, unassigned_to,
                           args.min_unit_faces, args.min_recall, args.min_visible_pixels)
    print_report(rows, list(dict.fromkeys(mask_set.owners)))
    np.save(os.path.abspath(args.out_labels), labels)
    with open(os.path.abspath(args.out_names), "w", encoding="utf-8") as handle:
        json.dump(part_order, handle, ensure_ascii=False, indent=2)
    if args.report:
        with open(os.path.abspath(args.report), "w", encoding="utf-8") as handle:
            json.dump(rows, handle, ensure_ascii=False, indent=2)
    for index, name in enumerate(part_order):
        print(f"  {name}: {int((labels == index).sum())} faces")
    print(f"  unnamed: {int((labels < 0).sum())} faces")
    print(f"saved {args.out_labels}")


if __name__ == "__main__":
    main()
