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
    parser.add_argument("--boxes_only", action="store_true",
                        help="Write the box prompts and a preview, without loading X-Part")
    parser.add_argument("--xpart_root", default=DEFAULT_XPART_ROOT)
    parser.add_argument("--model_path", default="tencent/Hunyuan3D-Part",
                        help="Local weights directory or HF repo id")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    boxes, rows = component_boxes(
        load_part_nodes(os.path.abspath(args.parts)), args.min_area_share, args.containment)
    if not len(boxes):
        raise SystemExit("no part component survived the filters; lower --min_area_share")
    print(f"{len(boxes)} box prompts:")
    for row in rows:
        print(f"  {row['name']:<18} {row['faces']:>7} faces  {row['area_share']:.1%} of the area")
    with open(os.path.join(out_dir, "boxes.json"), "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    preview = trimesh.Scene()
    preview.add_geometry(trimesh.load(os.path.abspath(args.glb), force="mesh"))
    for box in boxes:
        outline = trimesh.path.creation.box_outline()
        outline.vertices *= (box[1] - box[0])
        outline.vertices += (box[0] + box[1]) / 2
        preview.add_geometry(outline)
    preview.export(os.path.join(out_dir, "boxes.glb"))
    if args.boxes_only:
        print(f"saved {out_dir}/boxes.glb")
        return

    sys.path.insert(0, os.path.abspath(args.xpart_root))
    import torch
    from partgen.partformer_pipeline import PartFormerPipeline

    pipeline = PartFormerPipeline.from_pretrained(model_path=args.model_path, verbose=True)
    pipeline.to(device="cuda", dtype=torch.float32)
    parts, (out_bbox, mesh_gt_bbox, exploded) = pipeline(
        mesh_path=os.path.abspath(args.glb),
        aabb=boxes[None].astype(np.float32),
        octree_resolution=args.octree_resolution,
        seed=args.seed,
        output_type="trimesh",
    )
    parts.export(os.path.join(out_dir, "xpart_parts.glb"))
    exploded.export(os.path.join(out_dir, "xpart_exploded.glb"))
    out_bbox.export(os.path.join(out_dir, "xpart_parts_bbox.glb"))
    mesh_gt_bbox.export(os.path.join(out_dir, "xpart_input_bbox.glb"))
    print(f"saved {out_dir}/xpart_parts.glb")


if __name__ == "__main__":
    main()
