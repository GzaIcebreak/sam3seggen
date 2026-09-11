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

Run with the X-Part venv (see --xpart_root); nothing here imports SegviGen.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import trimesh

DEFAULT_XPART_ROOT = "/root/autodl-tmp/Hunyuan3D-Part/XPart"
DEFAULT_MIN_AREA_SHARE = 0.005
DEFAULT_CONTAINMENT = 0.98
DEFAULT_FIT_TOLERANCE = 0.05


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


def component_boxes(nodes, min_area_share=DEFAULT_MIN_AREA_SHARE,
                    containment=DEFAULT_CONTAINMENT):
    """One axis-aligned box per part instance: (boxes [K,2,3], rows of metadata)."""
    total_area = sum(float(mesh.area) for _, mesh in nodes)
    candidates = []
    for name, mesh in nodes:
        for faces in welded_components(mesh):
            area = float(mesh.area_faces[faces].sum())
            if area < min_area_share * total_area:
                continue
            corners = mesh.vertices[np.unique(mesh.faces[faces])]
            candidates.append({
                "name": name,
                "faces": int(len(faces)),
                "area_share": area / total_area,
                "box": np.stack([corners.min(axis=0), corners.max(axis=0)]),
            })

    boxes = np.stack([c["box"] for c in candidates]) if candidates else np.zeros((0, 2, 3))
    keep = []
    for index, candidate in enumerate(candidates):
        low, high = candidate["box"]
        volume = float(np.prod(np.maximum(high - low, 1e-9)))
        inside = False
        for other in range(len(candidates)):
            if other == index or candidates[other]["name"] != candidate["name"]:
                continue
            other_low, other_high = boxes[other]
            overlap = np.maximum(0.0, np.minimum(high, other_high) - np.maximum(low, other_low))
            bigger = float(np.prod(np.maximum(other_high - other_low, 1e-9))) > volume
            if bigger and float(np.prod(overlap)) >= containment * volume:
                inside = True
                break
        if not inside:
            keep.append(index)
    rows = [dict(candidates[i], box=candidates[i]["box"].tolist()) for i in keep]
    return boxes[keep], rows


def load_pipeline(model_path):
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
    parser.add_argument("--octree_resolution", type=int, default=512,
                        help="Marching-cubes resolution X-Part reconstructs each part at")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parts_per_batch", type=int, default=4,
                        help="Boxes generated per forward pass; twelve at once needs "
                             "more than 24 GB")
    parser.add_argument("--num_chunks", type=int, default=50000,
                        help="Query points per marching-cubes block; X-Part's own 400000 "
                             "needs 3 GiB a block and runs out on a 32 GB card")
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

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    nodes = load_part_nodes(os.path.abspath(args.parts))
    boxes, rows = component_boxes(nodes, args.min_area_share, args.containment)
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

    print(f"{len(boxes)} box prompts:")
    for row in rows:
        print(f"  {row['name']:<18} {row['faces']:>7} faces  {row['area_share']:.1%} of the area")
    with open(os.path.join(out_dir, "boxes.json"), "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

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

    # Parts are the batch dimension -- attention never crosses them, and the part-id
    # embedding is re-randomised on every call anyway -- so generating them a few at a
    # time is not an approximation. It is also the only way twelve of them fit in memory.
    out = trimesh.Scene()
    for start in range(0, len(boxes), args.parts_per_batch):
        chunk = boxes[start:start + args.parts_per_batch]
        named = [rows[start + i]["name"] for i in range(len(chunk))]
        print(f"[{start + 1}-{start + len(chunk)}/{len(boxes)}] {', '.join(named)}")
        parts, _ = pipeline(
            mesh_path=os.path.abspath(args.glb),
            # [K, 2, 3], not the [B, K, 2, 3] the docstring claims: check_inputs indexes
            # the boxes directly and adds the batch dimension itself.
            aabb=chunk.astype(np.float32),
            octree_resolution=args.octree_resolution,
            # The decode queries the implicit function in blocks of this many points and
            # X-Part defaults to 400k, which alone wants 3 GiB on top of everything the
            # diffusion pass is still holding. It only trades speed for memory.
            num_chunks=args.num_chunks,
            seed=args.seed,
            output_type="trimesh",
        )
        # X-Part drops a box whose surface sample came back empty, so the geometry it
        # returns is not guaranteed to line up one-for-one with the chunk; keep our names
        # only when the counts agree rather than mislabelling the output.
        geometries = list(parts.geometry.values())
        aligned = len(geometries) == len(chunk)
        if not aligned:
            print(f"  X-Part returned {len(geometries)} parts for {len(chunk)} boxes; "
                  "falling back to positional names")
        for offset, geometry in enumerate(geometries):
            label = named[offset] if aligned else f"part_{start + offset:02d}"
            out.add_geometry(geometry, geom_name=f"{start + offset:02d}_{label}")
        torch.cuda.empty_cache()

    out.export(os.path.join(out_dir, "xpart_parts.glb"))
    print(f"saved {out_dir}/xpart_parts.glb ({len(out.geometry)} solids)")


if __name__ == "__main__":
    main()
