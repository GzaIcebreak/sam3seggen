"""Turn our open, surface-split parts into closed solids with X-Part (Hunyuan3D-Part).

Splitting one shell into parts leaves every part open where it was cut, and the remesh's
inner walls sit inside as separate shells. X-Part regenerates each part as a complete
watertight shape from the whole object plus a bounding box prompt, so the cuts are healed
by generation rather than by capping geometry we do not have.

The bridge is the box list. X-Part's own P3-SAM predicts boxes; here they come from our
segmentation instead, which is the point -- our part decomposition is the thing we want
completed. Two things have to happen first:

* boxes must be per instance, not per name. Our parts.glb holds one node per name, so
  "arm" is both arms and its box spans the whole body; each node is therefore split into
  welded connected components.
* the remesh puts an inner wall inside every part, as a second shell whose box the outer
  shell's box swallows; a component contained in a bigger component *of the same part* is
  therefore dropped. Containment across parts is left alone -- a hand sits inside the
  arm's box and is still its own prompt.

Two things about X-Part are worth knowing before reading the rest. It is not a function of
its prompt -- a part-identity embedding is drawn by `torch.randperm` every forward pass, so
the same part from the same prompt can come back many times its own size, and --redraws
exists to draw it again. And a box is a lossy way to describe a part, which is the next
paragraph.

A box on its own is a lossy way to describe a part, and it is where the quality goes.
X-Part conditions each part on the source surface it finds *inside the box*, so anything
else that passes through the box is handed over as part of the prompt: the robot's torso
box also contains the tops of both legs, and what comes back is a torso with legs. The box
cannot say otherwise, because a box is all it is.

We are not limited to a box. The split already decided, per face, which part each triangle
belongs to, so --condition surface samples the conditioning points from exactly those
faces and passes them as `part_surface_inbbox` -- the same tensor X-Part would have built
by cropping, only built from the assignment instead. The box still goes along; it is what
sizes the token budget. This is the tight version of the handoff, and the box-cropped one
is kept as --condition box to compare against.

Run with the X-Part venv (see --xpart_root); nothing here imports SegviGen.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys

import numpy as np
import trimesh

DEFAULT_XPART_ROOT = "/root/autodl-tmp/Hunyuan3D-Part/XPart"
DEFAULT_MIN_AREA_SHARE = 0.005
DEFAULT_CONTAINMENT = 0.98
DEFAULT_MERGE_MAX_SHARE = 0.05
DEFAULT_FIT_TOLERANCE = 0.05
# What X-Part samples per part; the conditioner's positional encoding is fitted to it.
XPART_CONDITION_POINTS = 81920
CONDITION_MODES = ("surface", "box")
# A completion is meant to close the cut, which grows the part a little. Half the box again
# is far past that: the good draws of Mickey's parts all came in under 9% of their box and
# the bad ones overran by 174% and 675%, so anywhere in between separates them.
BOX_ESCAPE_WARNING = 0.5


def box_escape(generated_bounds, box):
    """How far a generated solid reaches outside its prompt box, per the box's own size.

    Zero if it stays inside; 1.0 if it overshoots by the full width of the box on some
    axis. Measured per axis rather than by volume so that one runaway direction, which is
    what the failures look like, is not averaged away by two well-behaved ones.
    """
    extent = np.maximum(box[1] - box[0], 1e-9)
    over = np.maximum(box[0] - generated_bounds[0], 0) + \
        np.maximum(generated_bounds[1] - box[1], 0)
    return float((over / extent).max())


def source_frame_transform(parts_bounds, source_bounds, tolerance=DEFAULT_FIT_TOLERANCE):
    """(scale, parts centre, source centre) mapping the parts' frame onto the source's.

    The boxes prompt X-Part about the *source* mesh, but the split normalises its output
    into a unit cube while the source may sit anywhere: Mickey stands on the ground plane,
    so his parts land half a unit below the model they are meant to describe. The robot is
    already origin-centred, which is exactly why this stayed invisible.

    The fit is a uniform scale plus a translation, recovered from the two bounding boxes.
    The three per-axis scales agreeing is what says the parts really do cover the whole
    model; when they disagree the fit means nothing, and handing X-Part boxes in the wrong
    place is worse than stopping.
    """
    parts_extent = parts_bounds[1] - parts_bounds[0]
    source_extent = source_bounds[1] - source_bounds[0]
    if np.any(parts_extent < 1e-9):
        raise SystemExit("the parts are flat in some axis; cannot fit them to the source")
    per_axis = source_extent / parts_extent
    spread = float(per_axis.max() / per_axis.min() - 1.0)
    if spread > tolerance:
        raise SystemExit(
            f"parts and source do not describe the same shape: per-axis scales "
            f"{np.round(per_axis, 4)} differ by {spread:.1%} (limit {tolerance:.0%}). "
            "Usually this means part of the model was dropped -- pass --unassigned_to.")
    return float(per_axis.mean()), parts_bounds.mean(axis=0), source_bounds.mean(axis=0)


def to_source_frame(boxes, scale, parts_centre, source_centre):
    return (boxes - parts_centre) * scale + source_centre


def load_part_nodes(parts_glb):
    """(name, mesh) per node of a parts glb, with the scene transform baked in."""
    scene = trimesh.load(parts_glb, force="scene")
    nodes = []
    for name in scene.geometry:
        geometry = scene.geometry[name].copy()
        transform, _ = scene.graph.get(scene.graph.geometry_nodes[name][0])
        geometry.apply_transform(transform)
        nodes.append((name, geometry))
    return nodes


def welded_components(mesh):
    """Face index groups of the mesh's connected components, ignoring UV seams.

    glTF stores a separate vertex per UV corner, so splitting on the raw indices reports
    far more shells than the surface really has.
    """
    _, welded = np.unique(np.asarray(mesh.vertices).round(6), axis=0, return_inverse=True)
    faces = welded.ravel()[np.asarray(mesh.faces)]
    edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    shared = trimesh.grouping.group_rows(edges, require_count=2)
    face_of_edge = np.repeat(np.arange(len(faces)), 3)
    return trimesh.graph.connected_components(
        face_of_edge[shared], nodes=np.arange(len(faces)))


def welded_pieces(nodes):
    """Every welded connected component of every part node, as (name, submesh).

    Geometry only. A textured submesh carries a copy of the part's material, so a split
    with thousands of shells (a decorated tree: ~7900) ran the 90 GB box out of memory
    before X-Part even loaded. Nothing downstream reads the pieces' textures -- the
    conditioning samples points and face normals, and the bake uses the source glb.
    """
    pieces = []
    for name, mesh in nodes:
        bare = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
        pieces += [(name, bare.submesh([faces], append=True))
                   for faces in welded_components(bare)]
    return pieces


def bounding_box(mesh):
    return np.stack([np.asarray(mesh.bounds[0]), np.asarray(mesh.bounds[1])])


def drop_inner_shells(pieces, containment=DEFAULT_CONTAINMENT):
    """Drop a piece sitting inside a bigger piece of the same part: the remesh's inner wall.

    Dropped rather than folded in, unlike the slivers below. An inner wall is a duplicate
    of the outer one with its normals facing the other way, and conditioning on both would
    describe a shape that is inside out in half its points.

    Containment across parts is left alone -- a hand sits inside the arm's box and is
    still its own part.
    """
    if not pieces:
        return []
    boxes = np.stack([bounding_box(piece) for _, piece in pieces])
    volumes = np.prod(np.maximum(boxes[:, 1] - boxes[:, 0], 1e-9), axis=1)
    _, name_ids = np.unique([name for name, _ in pieces], return_inverse=True)
    keep = []
    # Vectorised over the rivals: a decorated tree splits into ~7900 pieces, and the
    # pairwise Python loop this replaces was 63 million iterations.
    for index in range(len(pieces)):
        rivals = (name_ids == name_ids[index]) & (volumes > volumes[index])
        if rivals.any():
            low, high = boxes[index]
            overlap = np.maximum(0.0, np.minimum(high, boxes[rivals, 1])
                                 - np.maximum(low, boxes[rivals, 0]))
            if np.any(np.prod(overlap, axis=1) >= containment * volumes[index]):
                continue
        keep.append(index)
    return [pieces[i] for i in keep]


def fold_small_pieces(pieces, min_area_share=DEFAULT_MIN_AREA_SHARE,
                      part_min_area_share=None, fold_within_part=False):
    """Fold a component too small to be worth its own prompt into the nearest bigger one.

    X-Part is not reliable on a sliver: on both test models exactly one component under 1%
    of the surface came back 15 to 59 times its own volume. A sliver is usually not a part
    anyway but a leftover of where the split cut, and the thing to do with it is to let it
    ride along with whatever it is attached to.

    Folding is also what the old behaviour should have been. Dropping these left a hole:
    the surface went into no prompt at all, so nothing X-Part returned covered it.

    The target is the nearest bigger piece by surface distance, whatever its name. Name is
    not a useful tie-breaker here -- three of Mickey's foot components are nowhere near
    each other, and merging them because they share a name would make one box spanning the
    gaps between them.

    `fold_within_part` is the exception for a part made of many small separate things
    stuck onto a bigger one (ornaments on a tree): by distance alone every ornament sliver
    goes to the branch it touches and the part disappears. With it, a sliver joins the
    nearest kept piece of its own part, and falls back to any part only when its own part
    kept nothing. `part_min_area_share` ({name: share}) sets the floor per part.
    """
    from scipy.spatial import cKDTree

    total_area = sum(float(piece.area) for _, piece in pieces)
    floors = part_min_area_share or {}
    keep = [index for index, (name, piece) in enumerate(pieces)
            if piece.area >= floors.get(name, min_area_share) * total_area]
    if not keep:
        raise SystemExit(
            f"every component is below --min_area_share {min_area_share}; lower it")

    groups = {index: [pieces[index][1]] for index in keep}
    # One tree per candidate set (all kept pieces, or one part's) instead of one query per
    # kept piece: 30 s -> under a second on the plane (77 slivers x 14 pieces). The
    # nearest piece is the owner of the nearest vertex; on an exact tie (a sliver touching
    # two pieces) the first candidate wins, as with the per-piece min() it replaces.
    forests = {}

    def forest(candidates):
        key = tuple(candidates)
        if key not in forests:
            points = [np.asarray(pieces[i][1].vertices) for i in candidates]
            owner = np.concatenate([np.full(len(p), k) for k, p in enumerate(points)])
            forests[key] = (cKDTree(np.concatenate(points)), owner)
        return forests[key]

    def nearest(vertices, candidates):
        tree, owner = forest(candidates)
        dist, idx = tree.query(vertices)
        best = float(dist.min())
        tied = set()
        for point in vertices[dist <= best]:
            for hit in tree.query_ball_point(point, best * (1 + 1e-9) + 1e-12):
                tied.add(int(owner[hit]))
        return candidates[min(tied)] if tied else candidates[int(owner[idx[int(dist.argmin())]])]

    for index, (name, piece) in enumerate(pieces):
        if index in groups:
            continue
        vertices = np.asarray(piece.vertices)
        own = [i for i in keep if pieces[i][0] == name] if fold_within_part else []
        target = nearest(vertices, own or keep)
        print(f"  {name} component at {piece.area / total_area:.2%} of the surface is too "
              f"small to generate; folded into the {pieces[target][0]} next to it")
        groups[target].append(piece)
    return [(pieces[index][0],
             trimesh.util.concatenate(groups[index]) if len(groups[index]) > 1
             else pieces[index][1])
            for index in keep]


def merge_split_fragments(pieces, gap, max_share=DEFAULT_MERGE_MAX_SHARE):
    """Rejoin same-name pieces that only came apart where another part cut through them.

    The monk's hands are one part with the body, but the staff he grips splits each hand
    into two shells a staff-width apart, and X-Part turned those half-hands into slabs three
    to five times their size. The two halves sit 0.3% of the model diagonal from each
    other; joined, each is a whole hand again and a prompt X-Part can close.

    Two pieces of the same part are joined when their surfaces come within gap (a share
    of the model diagonal) and at least one of them is under max_share of the surface.
    Two big pieces never join: the head and the legs are both "body" and both belong on
    their own. Joining is transitive, so a staff cut into four by two hands is one staff.
    A gap of 0 turns this off.
    """
    from scipy.spatial import cKDTree

    if gap <= 0 or len(pieces) < 2:
        return pieces
    boxes = np.stack([bounding_box(piece) for _, piece in pieces])
    diag = float(np.linalg.norm(boxes[:, 1].max(axis=0) - boxes[:, 0].min(axis=0)))
    reach = gap * diag
    total_area = sum(float(piece.area) for _, piece in pieces)
    small = np.array([piece.area < max_share * total_area for _, piece in pieces])
    _, name_ids = np.unique([name for name, _ in pieces], return_inverse=True)
    parent = list(range(len(pieces)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    trees = {}
    for i in range(len(pieces)):
        # Boxes grown by the reach must overlap before the surfaces are worth measuring.
        near = (name_ids == name_ids[i]) & (small | small[i]) & (np.arange(len(pieces)) > i)
        near &= np.all(boxes[:, 0] <= boxes[i, 1] + reach, axis=1)
        near &= np.all(boxes[:, 1] >= boxes[i, 0] - reach, axis=1)
        for j in np.flatnonzero(near):
            if find(i) == find(j):
                continue
            if j not in trees:
                trees[j] = cKDTree(np.asarray(pieces[j][1].vertices))
            if trees[j].query(np.asarray(pieces[i][1].vertices), distance_upper_bound=reach)[0].min() <= reach:
                parent[find(j)] = find(i)

    groups = {}
    for index in range(len(pieces)):
        groups.setdefault(find(index), []).append(index)
    merged = []
    for members in groups.values():
        name = pieces[members[0]][0]
        if len(members) > 1:
            shares = ", ".join(f"{pieces[m][1].area / total_area:.1%}" for m in members)
            print(f"  {name}: {len(members)} pieces within {gap:.1%} of the diagonal "
                  f"({shares}) joined into one prompt")
        merged.append((name, trimesh.util.concatenate([pieces[m][1] for m in members])
                       if len(members) > 1 else pieces[members[0]][1]))
    return merged


def component_boxes(nodes, min_area_share=DEFAULT_MIN_AREA_SHARE,
                    containment=DEFAULT_CONTAINMENT, part_min_area_share=None,
                    fold_within_part=False, merge_gap=0.0,
                    merge_max_share=DEFAULT_MERGE_MAX_SHARE):
    """Per part instance: (boxes [K,2,3], rows of metadata, the surfaces they came from)."""
    pieces = fold_small_pieces(
        merge_split_fragments(drop_inner_shells(welded_pieces(nodes), containment),
                              merge_gap, merge_max_share),
        min_area_share, part_min_area_share, fold_within_part)
    total_area = sum(float(piece.area) for _, piece in pieces)
    boxes = np.stack([bounding_box(piece) for _, piece in pieces])
    rows = [{"name": name, "instance": index, "faces": int(len(piece.faces)),
             "area_share": float(piece.area) / total_area, "box": box.tolist()}
            for index, ((name, piece), box) in enumerate(zip(pieces, boxes))]
    return boxes, rows, [piece for _, piece in pieces]


def group_solids(names, solids):
    """Put independently repaired instances back into the named groups they came from.

    `parts.glb` is one node per prompt — both hands live in `hand`. Repair has to treat
    those as two objects (a shared box would span the body), then this concatenates the
    closed solids so the completed glb has the same grouping again.
    """
    order, buckets = [], {}
    for name, solid in zip(names, solids):
        if solid is None:
            continue
        if name not in buckets:
            order.append(name)
            buckets[name] = []
        buckets[name].append(solid)
    grouped = []
    for name in order:
        meshes = buckets[name]
        grouped.append((name, trimesh.util.concatenate(meshes) if len(meshes) > 1
                        else meshes[0]))
    return grouped


def xpart_normalization(bounds):
    """The (centre, scale) X-Part's normalize_mesh derives from a mesh's bounding box.

    Reproduced rather than called because the conditioning points have to land in the same
    frame as the mesh the pipeline normalises internally, and by the time it has done so it
    no longer accepts anything of ours.
    """
    centre = bounds.mean(axis=0)
    return centre, float(np.max(bounds[1] - bounds[0]) / 2 / 0.8)


def part_surface_condition(surfaces, centre, scale, num_points=XPART_CONDITION_POINTS,
                           seed=42):
    """[K, N, 7] of (point, normal, sharp-edge flag) sampled from each part's own faces.

    The flag is the seventh channel X-Part's own sampler fills with zeros; it marks points
    taken from sharp edges, which it never does for a box crop and we do not either.
    """
    samples = []
    for surface in surfaces:
        if surface.area <= 0:
            raise SystemExit("a part component has no area; cannot sample its surface")
        points, face_index = trimesh.sample.sample_surface(surface, num_points, seed=seed)
        normals = surface.face_normals[face_index]
        samples.append(np.hstack([
            (np.asarray(points) - centre) / scale,
            np.asarray(normals),
            np.zeros((num_points, 1)),
        ]))
    return np.stack(samples).astype(np.float32)


_RANDOM_INITS = ("uniform_", "normal_", "trunc_normal_", "kaiming_uniform_", "kaiming_normal_",
                 "xavier_uniform_", "xavier_normal_", "orthogonal_")


@contextlib.contextmanager
def no_random_init():
    """Make torch.nn.init's random fills no-ops while modules are built.

    Every parameter X-Part builds is then overwritten by a strict load_state_dict, so the
    random fill is pure cost (21 of the 38 s load). Deterministic fills (zeros_, ones_,
    constant_) are left alone.
    """
    import torch

    saved = {name: getattr(torch.nn.init, name) for name in _RANDOM_INITS
             if hasattr(torch.nn.init, name)}
    for name in saved:
        setattr(torch.nn.init, name, lambda tensor, *args, **kwargs: tensor)
    try:
        yield
    finally:
        for name, fn in saved.items():
            setattr(torch.nn.init, name, fn)


def load_pipeline(model_path, skip_init=True):
    """X-Part's pipeline without the P3-SAM box predictor it builds unconditionally.

    That predictor is the part of X-Part we are replacing: it downloads facebook/sonata
    on construction, and the pipeline only ever calls it when `aabb` is None, which ours
    never is. Building it would make our segmentation depend on the box guesser it exists
    to override -- and on a network round trip -- for nothing.
    """
    from partgen import partformer_pipeline

    target = None
    config_path = os.path.join(model_path, "p3sam", "config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as handle:
            target = json.load(handle).get("target")

    original = partformer_pipeline.instantiate_from_config

    def skip_box_predictor(config, **kwargs):
        if target and config.get("target") == target:
            print(f"  not building {target}; the boxes are ours")
            return None
        return original(config, **kwargs)

    partformer_pipeline.instantiate_from_config = skip_box_predictor
    try:
        with (no_random_init() if skip_init else contextlib.nullcontext()):
            return partformer_pipeline.PartFormerPipeline.from_pretrained(
                model_path=model_path, verbose=True)
    finally:
        partformer_pipeline.instantiate_from_config = original


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--glb", required=True, help="The original, whole model")
    parser.add_argument("--parts", required=True,
                        help="Our segmentation (segment_parts.py output); only its boxes are used")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--min_area_share", type=float, default=DEFAULT_MIN_AREA_SHARE,
                        help="Drop components below this share of the model's surface")
    parser.add_argument("--containment", type=float, default=DEFAULT_CONTAINMENT,
                        help="Drop a box this fully inside a bigger box of the same part "
                             "(that is what the remesh's inner walls look like)")
    parser.add_argument("--part_min_area_share", action="append", default=[],
                        metavar="NAME=SHARE",
                        help="Per-part --min_area_share, repeatable (e.g. 装饰品=0.001)")
    parser.add_argument("--merge_gap", type=float, default=0.0,
                        help="Join same-part pieces whose surfaces come within this share "
                             "of the model diagonal (a hand the staff cut in two). 0 = off")
    parser.add_argument("--merge_max_share", type=float, default=DEFAULT_MERGE_MAX_SHARE,
                        help="Only a piece under this share of the surface joins a "
                             "neighbour; two big pieces stay apart")
    parser.add_argument("--fold_within_part", action="store_true",
                        help="Fold a small component only into its own part's pieces "
                             "(many small things on a big one: ornaments on a tree)")
    parser.add_argument("--octree_resolution", type=int, default=512,
                        help="Marching-cubes resolution X-Part reconstructs each part at")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parts_per_batch", type=int, default=4,
                        help="Boxes generated per forward pass; twelve at once needs "
                             "more than 24 GB")
    parser.add_argument("--num_chunks", type=int, default=50000,
                        help="Query points per marching-cubes block; X-Part's own 400000 "
                             "needs 3 GiB a block and runs out on a 32 GB card")
    parser.add_argument("--redraws", type=int, default=2,
                        help="How many times to draw a part again when its solid comes "
                             "back far bigger than its box; X-Part's part-identity "
                             "embedding is random per pass, so a redraw is a new draw")
    parser.add_argument("--condition", choices=CONDITION_MODES, default="surface",
                        help="What describes a part to X-Part: the faces the split "
                             "assigned to it, or (box) whatever of the source falls "
                             "inside its bounding box, which is X-Part's own default")
    parser.add_argument("--boxes_only", action="store_true",
                        help="Write the box prompts and a preview, without loading X-Part")
    parser.add_argument("--no_source_frame", dest="source_frame", action="store_false",
                        help="Prompt with the boxes as they are, skipping the fit onto the "
                             "source model's frame (they only coincide for a model that "
                             "was already origin-centred)")
    parser.add_argument("--fit_tolerance", type=float, default=DEFAULT_FIT_TOLERANCE,
                        help="How far the three per-axis scales of that fit may disagree")
    parser.add_argument("--xpart_root", default=DEFAULT_XPART_ROOT)
    parser.add_argument("--model_path", default="tencent/Hunyuan3D-Part",
                        help="Local weights directory or HF repo id")
    args = parser.parse_args()
    part_floors = {}
    for item in args.part_min_area_share:
        name, _, share = item.rpartition("=")
        if not name:
            parser.error(f"--part_min_area_share expects NAME=SHARE, got {item!r}")
        part_floors[name] = float(share)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    nodes = load_part_nodes(os.path.abspath(args.parts))
    boxes, rows, surfaces = component_boxes(nodes, args.min_area_share, args.containment,
                                            part_floors, args.fold_within_part,
                                            args.merge_gap, args.merge_max_share)
    if not len(boxes):
        raise SystemExit("no part component survived the filters; lower --min_area_share")

    source = trimesh.load(os.path.abspath(args.glb), force="mesh")
    if args.source_frame:
        parts_bounds = np.stack([
            np.min([mesh.bounds[0] for _, mesh in nodes], axis=0),
            np.max([mesh.bounds[1] for _, mesh in nodes], axis=0),
        ])
        scale, parts_centre, source_centre = source_frame_transform(
            parts_bounds, source.bounds, args.fit_tolerance)
        shift = source_centre - parts_centre
        print(f"parts -> source frame: scale {scale:.4f}, shift {np.round(shift, 4)}")
        boxes = to_source_frame(boxes, scale, parts_centre, source_centre)
        for row, box in zip(rows, boxes):
            row["box"] = box.tolist()
        for surface in surfaces:
            surface.vertices = to_source_frame(
                np.asarray(surface.vertices), scale, parts_centre, source_centre)

    print(f"{len(boxes)} box prompts:")
    for row in rows:
        print(f"  {row['name']:<18} {row['faces']:>7} faces  {row['area_share']:.1%} of the area")
    with open(os.path.join(out_dir, "boxes.json"), "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    open_instances = trimesh.Scene()
    for index, (row, surface) in enumerate(zip(rows, surfaces)):
        open_instances.add_geometry(surface, geom_name=f"{index:02d}_{row['name']}")
    open_instances.export(os.path.join(out_dir, "open_instances.glb"))

    from glb_images import keep_jpeg, mesh_materials

    keep_jpeg(mesh_materials(source), args.glb)   # the preview re-encoded 8K JPEGs as PNG (21 s)
    preview = trimesh.Scene()
    preview.add_geometry(source)
    for box in boxes:
        outline = trimesh.path.creation.box_outline()
        outline.vertices *= (box[1] - box[0])
        outline.vertices += (box[0] + box[1]) / 2
        preview.add_geometry(outline)
    preview.export(os.path.join(out_dir, "boxes.glb"))
    if args.boxes_only:
        print(f"saved {out_dir}/boxes.glb")
        return

    # X-Part builds its own P3-SAM box predictor at load time even though we supply the
    # boxes ourselves, and that module reaches for its sibling with a *relative*
    # sys.path.append("../P3-SAM") -- which only resolves when cwd happens to be XPart/.
    # Add both roots absolutely so the import works wherever this is dispatched from.
    xpart_root = os.path.abspath(args.xpart_root)
    for path in (xpart_root, os.path.join(os.path.dirname(xpart_root), "P3-SAM")):
        if path not in sys.path:
            sys.path.insert(0, path)
    import torch

    pipeline = load_pipeline(args.model_path)
    pipeline.to(device="cuda", dtype=torch.float32)

    condition = None
    if args.condition == "surface":
        centre, norm_scale = xpart_normalization(source.bounds)
        print(f"sampling {XPART_CONDITION_POINTS} points from each part's own faces "
              f"(normalisation centre {np.round(centre, 4)}, scale {norm_scale:.4f})")
        condition = torch.from_numpy(part_surface_condition(
            surfaces, centre, norm_scale, seed=args.seed))

    names = [row["name"] for row in rows]
    solids = generate(pipeline, os.path.abspath(args.glb), boxes, condition, names, args)

    solids = redraw_escapees(pipeline, os.path.abspath(args.glb), boxes, condition, names,
                             solids, args)

    instances = trimesh.Scene()
    for index, solid in enumerate(solids):
        if solid is not None:
            instances.add_geometry(solid, geom_name=f"{index:02d}_{names[index]}")
    instances.export(os.path.join(out_dir, "xpart_instances.glb"))
    print(f"saved {out_dir}/xpart_instances.glb ({len(instances.geometry)} solids)")

    out = trimesh.Scene()
    for name, mesh in group_solids(names, solids):
        out.add_geometry(mesh, geom_name=name)
    out.export(os.path.join(out_dir, "xpart_parts.glb"))
    print(f"saved {out_dir}/xpart_parts.glb "
          f"({len(out.geometry)} groups from {len(instances.geometry)} solids)")


def generate(pipeline, glb, boxes, condition, names, args):
    """One solid per box, None where X-Part returned nothing for it.

    Parts are the batch dimension -- attention never crosses them, and the part-id
    embedding is re-randomised on every call anyway -- so generating them a few at a time
    is not an approximation. It is also the only way twelve of them fit in memory.
    """
    import torch

    solids = [None] * len(boxes)
    for start in range(0, len(boxes), args.parts_per_batch):
        chunk = boxes[start:start + args.parts_per_batch]
        print(f"[{start + 1}-{start + len(chunk)}/{len(boxes)}] "
              f"{', '.join(names[start:start + len(chunk)])}")
        # The two branches disagree on the boxes' shape on purpose. check_inputs adds the
        # batch dimension itself, but only on the path where it also samples the
        # conditioning; hand it the conditioning and that line is skipped, so the batch
        # dimension becomes ours to add. The docstring's [B,K,2,3] is right for one and
        # wrong for the other.
        prompt = ({"aabb": chunk.astype(np.float32)} if condition is None else
                  {"aabb": chunk.astype(np.float32)[None],
                   "part_surface_inbbox": condition[start:start + len(chunk)][None]})
        parts, _ = pipeline(
            mesh_path=glb,
            **prompt,
            octree_resolution=args.octree_resolution,
            # The decode queries the implicit function in blocks of this many points and
            # X-Part defaults to 400k, which alone wants 3 GiB on top of everything the
            # diffusion pass is still holding. It only trades speed for memory.
            num_chunks=args.num_chunks,
            seed=args.seed,
            output_type="trimesh",
        )
        # X-Part drops a box whose surface sample came back empty, so the geometry it
        # returns is not guaranteed to line up one-for-one with the chunk; positional is
        # all we can do then, and it is worth saying so.
        geometries = list(parts.geometry.values())
        if len(geometries) != len(chunk):
            print(f"  X-Part returned {len(geometries)} solids for {len(chunk)} boxes; "
                  "matching them positionally")
        for offset, geometry in enumerate(geometries[:len(chunk)]):
            solids[start + offset] = geometry
        torch.cuda.empty_cache()
    return solids


def redraw_escapees(pipeline, glb, boxes, condition, names, solids, args):
    """Draw a part again when the solid it produced is far too big for its box.

    X-Part is not a function of its prompt. `partformer_dit` adds a part-identity embedding
    picked by `torch.randperm` on every forward pass, so a part's result depends on the
    draw and on which other parts came with it. The same foot, from the same conditioning,
    overran its box by 675%, then 174%, then 7% across three runs -- so the occasional
    part that comes back many times its own size is bad luck, not a bad prompt, and the
    answer is another draw rather than a weaker prompt.

    The box is what makes this checkable at all: it says how big the part was, and nothing
    that closes a cut should need half the box again.
    """
    for attempt in range(max(args.redraws, 0)):
        escaped = {index: box_escape(solid.bounds, boxes[index])
                   for index, solid in enumerate(solids) if solid is not None}
        runaway = sorted(i for i, over in escaped.items() if over > BOX_ESCAPE_WARNING)
        if not runaway:
            return solids
        print(f"{len(runaway)} of {len(boxes)} parts overran their box "
              f"({', '.join(f'{names[i]} by {escaped[i]:.0%}' for i in runaway)}); "
              f"redrawing them (attempt {attempt + 1} of {args.redraws}) ...")
        replacements = generate(
            pipeline, glb, boxes[runaway],
            None if condition is None else condition[runaway],
            [names[i] for i in runaway], args)
        for index, replacement in zip(runaway, replacements):
            if replacement is None:
                continue
            after = box_escape(replacement.bounds, boxes[index])
            if after >= escaped[index]:
                print(f"  {names[index]}: the new draw overruns by {after:.0%}; keeping "
                      f"the {escaped[index]:.0%} one")
                continue
            print(f"  {names[index]}: {escaped[index]:.0%} -> {after:.0%} overrun, kept")
            solids[index] = replacement
    still = [names[i] for i, solid in enumerate(solids)
             if solid is not None and box_escape(solid.bounds, boxes[i]) > BOX_ESCAPE_WARNING]
    if still:
        print(f"still overrunning after {args.redraws} redraws: {', '.join(still)}")
    return solids


if __name__ == "__main__":
    main()
