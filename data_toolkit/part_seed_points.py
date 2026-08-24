"""Pick voxel seed points inside each part, for SegviGen's interactive point channel.

The 2D map reaches the model only as a global DINOv3 embedding, so it can bias the
result but never pin a region down. The point channel is different: points arrive as a
sparse tensor at voxel coordinates, which is a real spatial constraint the model was
trained on. That makes this the place to hand SAM3's semantics over, and it leaves the
boundaries entirely to SegviGen, which is what keeps them smooth.

Two things decide where a point goes:
  * distance from the part's border, because a point near a seam is exactly where the
    multi-view vote is least certain;
  * spread, because a grouped part is often several disconnected regions (a hand, a
    head and two legs) and a point in one of them says nothing about the others.
"""
import argparse
import json
import os
import sys

import numpy as np
import trimesh

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_toolkit.parts_rebake import welded_face_adjacency

# inference_interactive.py voxelises at 512 into the aabb [-0.5, 0.5], then the texture
# encoder downsamples by 16, so two seeds closer than 1/32 collapse into one latent voxel.
GRID = 512
LATENT_STRIDE = 16
MAX_POINTS = 10


def normalize_like_pipeline(vertices):
    """Match process_glb_to_vxz: centre the bounding box, fit it inside the unit cube."""
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    center = (low + high) / 2
    scale = 0.99999 / float((high - low).max())
    return (vertices - center) * scale


def depth_from_border(adjacency, labels):
    """Hops from the nearest face of another part, so interior faces score highest."""
    adjacency = np.asarray(adjacency)
    depth = np.full(len(labels), -1, dtype=np.int32)
    if len(adjacency) == 0:
        return depth
    left, right = adjacency[:, 0], adjacency[:, 1]
    crossing = labels[left] != labels[right]
    frontier = np.unique(adjacency[crossing].ravel())
    depth[frontier] = 0

    neighbours_of = {}
    for a, b in adjacency:
        neighbours_of.setdefault(a, []).append(b)
        neighbours_of.setdefault(b, []).append(a)

    level = 0
    while len(frontier):
        level += 1
        candidates = np.unique(
            np.concatenate([neighbours_of.get(face, [face]) for face in frontier])
        )
        candidates = candidates[depth[candidates] < 0]
        if not len(candidates):
            break
        depth[candidates] = level
        frontier = candidates
    # Parts that never touch another part have no border at all; treat them as interior.
    depth[depth < 0] = level + 1
    return depth


def part_components(adjacency, labels, areas, min_ratio):
    """Split each part into its connected regions, keeping those worth asking about.

    Interactive mode answers "which part does this point belong to", and it answers well
    when that part is one connected structure: seeded on the staff or the base it returns
    exactly those. Seeded on a grouped part whose regions are scattered -- a head, two
    hands, two boots -- it has no single structure to converge on and returns the whole
    model instead. Asking per region plays to what it does well, and the grouping is
    restored afterwards by unioning the regions back together.
    """
    from trimesh.graph import connected_components

    adjacency = np.asarray(adjacency)
    same = labels[adjacency[:, 0]] == labels[adjacency[:, 1]]
    components = connected_components(adjacency[same], nodes=np.arange(len(labels)))
    total = float(areas.sum())
    kept = []
    for faces in components:
        share = float(areas[faces].sum()) / total
        if share >= min_ratio:
            kept.append((int(labels[faces[0]]), faces, share))
    kept.sort(key=lambda entry: -entry[2])
    return kept


def farthest_points(points, count, min_separation):
    """Greedy spread, refusing points that would collapse into an already taken voxel."""
    chosen = [int(np.argmax(np.linalg.norm(points - points.mean(axis=0), axis=1)))]
    distance = np.linalg.norm(points - points[chosen[0]], axis=1)
    while len(chosen) < count:
        candidate = int(np.argmax(distance))
        if distance[candidate] < min_separation:
            break
        chosen.append(candidate)
        distance = np.minimum(distance, np.linalg.norm(points - points[candidate], axis=1))
    return chosen


def main():
    parser = argparse.ArgumentParser(description="Voxel seed points per part for point-guided SegviGen.")
    parser.add_argument("--glb", required=True, help="The model that will be voxelised, i.e. the original")
    parser.add_argument("--labels", help="npy of one part label per face of --glb")
    parser.add_argument("--label_names")
    parser.add_argument("--auto_seeds", type=int, default=0,
                        help="Ignore --labels and spread this many single-point queries over the "
                             "whole model, which is how an over-segmentation is obtained with no "
                             "prior labelling of any kind. One point per region on purpose: a "
                             "point asks 'which part contains this', whereas several spread points "
                             "ask for a part spanning all of them, and that is what makes a region "
                             "swallow its neighbours.")
    parser.add_argument("--out", required=True, help="JSON of seed points per part")
    parser.add_argument("--points", type=int, default=MAX_POINTS)
    parser.add_argument("--interior_quantile", type=float, default=0.6,
                        help="Keep only faces this deep inside the part before spreading seeds.")
    parser.add_argument("--per_component", action="store_true",
                        help="Emit seeds per connected region rather than per part, which is what "
                             "interactive mode can actually answer; the grouping is kept in the "
                             "output so the regions can be unioned back afterwards.")
    parser.add_argument("--min_component_ratio", type=float, default=0.005,
                        help="Ignore regions below this share of the model's area, which are speckle.")
    args = parser.parse_args()

    mesh = trimesh.load(os.path.abspath(args.glb), force="mesh")
    if not args.auto_seeds and not (args.labels and args.label_names):
        raise SystemExit("pass --auto_seeds N, or both --labels and --label_names")

    vertices = normalize_like_pipeline(np.asarray(mesh.vertices, dtype=np.float64))
    centroids = vertices[np.asarray(mesh.faces)].mean(axis=1)
    areas = np.asarray(mesh.area_faces)

    if args.auto_seeds:
        # Spread over the surface, weighted by nothing at all: the whole point is that no
        # semantics enter here, so the granularity comes from how many queries are asked.
        picked = farthest_points(centroids, args.auto_seeds,
                                 min_separation=2.0 * LATENT_STRIDE / GRID)
        voxels = np.clip(np.floor((centroids[picked] + 0.5) * GRID).astype(int), 0, GRID - 1)
        result = {f"region_{i:02d}": {"part": f"region_{i:02d}", "points": [voxel.tolist()]}
                  for i, voxel in enumerate(voxels)}
        print(f"{len(mesh.faces)} faces -> {len(result)} single-point queries, "
              f"{len(np.unique(voxels // LATENT_STRIDE, axis=0))} distinct latent voxels")
        out = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(f"\nwrote {out}")
        return

    labels = np.load(os.path.abspath(args.labels))
    with open(os.path.abspath(args.label_names), "r", encoding="utf-8") as handle:
        names = json.load(handle)
    if len(labels) != len(mesh.faces):
        raise SystemExit(f"labels cover {len(labels)} faces but {args.glb} has {len(mesh.faces)}")

    adjacency = welded_face_adjacency(mesh)
    depth = depth_from_border(adjacency, labels)
    print(f"{len(names)} parts, {len(mesh.faces)} faces, border depth up to {depth.max()}")

    if args.per_component:
        regions = [
            (f"{names[part]}#{index}", names[part], faces)
            for index, (part, faces, _) in enumerate(part_components(adjacency, labels, areas, args.min_component_ratio))
        ]
        print(f"{len(regions)} regions above {args.min_component_ratio:.1%} of the model's area")
    else:
        regions = [(name, name, np.flatnonzero(labels == index)) for index, name in enumerate(names)]

    result = {}
    for region, part, selection in regions:
        if not len(selection):
            print(f"  {region}: no faces, skipped")
            continue
        cutoff = np.quantile(depth[selection], args.interior_quantile)
        interior = selection[depth[selection] >= cutoff]
        picked = farthest_points(
            centroids[interior], args.points, min_separation=2.0 * LATENT_STRIDE / GRID
        )
        voxels = np.clip(np.floor((centroids[interior[picked]] + 0.5) * GRID).astype(int), 0, GRID - 1)
        result[region] = {"part": part, "points": voxels.tolist()}
        print(
            f"  {region:>12}: {len(selection):>7} faces, {areas[selection].sum() / areas.sum() * 100:5.1f}% area | "
            f"depth >= {cutoff:.0f} | {len(voxels)} seeds -> "
            f"{len(np.unique(voxels // LATENT_STRIDE, axis=0))} distinct latent voxels"
        )

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
