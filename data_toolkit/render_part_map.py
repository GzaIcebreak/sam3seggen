"""Render a split glb with one flat colour per part, straight from the rasteriser.

Exporting a debug glb with vertex colours and rendering it in Blender shows nothing
unless a material wires COLOR_0 into the shader, which is why the earlier colour
previews came out blank. Colouring by node during rasterisation needs no material at
all, so what you see is exactly the part assignment.
"""
import argparse
import json
import os
import sys

import numpy as np
import trimesh
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_toolkit.multiview import (
    camera_ring,
    gltf_to_blender,
    normalize_to_unit_cube,
    rasterize_face_ids,
)
from data_toolkit.parts_rebake import LABEL_COLORS


def main():
    parser = argparse.ArgumentParser(description="Render a split glb coloured by part.")
    parser.add_argument("--glb", required=True, help="Combined glb with one node per part")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--labels", help="npy of one part index per face; colours the mesh by these "
                                         "instead of by node, so labels can be checked before any split")
    parser.add_argument("--label_names", help="JSON list naming each label index")
    parser.add_argument("--texture", action="store_true",
                        help="Colour faces by the mesh's own baseColor texture, which is how "
                             "SegviGen encodes its segmentation; use it to inspect the raw "
                             "output before any clustering into parts.")
    parser.add_argument("--up", choices=["y", "z", "-z"], default="y",
                        help="Up axis of the input. glTF files are y-up, but SegviGen's decoder "
                             "writes meshes in voxel order, which is z-up pointing down (-z); "
                             "converting those as glTF views the model down its own up axis.")
    parser.add_argument("--azimuths", default="0,90,135,225")
    parser.add_argument("--elevations", default="10")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--camera_angle_x", type=float, default=0.6981317007977318)
    parser.add_argument("--explode", type=float, default=0.0,
                        help="Push each part outward by this fraction of the model size. An "
                             "exploded view is the quickest way to see whether a seam is clean, "
                             "since nothing else is drawn on top of it.")
    parser.add_argument("--layout", choices=["centroid", "fan"], default="fan",
                        help="centroid pushes each part away from the model centre, which barely "
                             "separates parts that are concentric; fan spreads them evenly around "
                             "the up axis so every part is seen in isolation.")
    parser.add_argument("--sheet", action="store_true",
                        help="Also write a contact sheet: the whole model, then each part alone in "
                             "its original pose. Unlike an exploded view this can never let one "
                             "part hide another, so a torn boundary has nowhere to hide.")
    parser.add_argument("--radius", type=float, default=2.0,
                        help="Camera distance; raise it when exploding so parts stay in frame.")
    args = parser.parse_args()

    if args.texture:
        # SegviGen's own colouring, shown as it is rather than after clustering it into
        # labels, so what the model actually painted can be judged without a middle step.
        mesh = trimesh.load(os.path.abspath(args.glb), force="mesh")
        if getattr(mesh.visual, "kind", None) == "vertex":
            # Some pipelines encode the segmentation as vertex colours rather than a texture.
            colours = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.float64)
            face_colours = colours[np.asarray(mesh.faces)].mean(axis=1).astype(np.uint8)
        else:
            image = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB"))
            height, width, _ = image.shape
            uv = np.asarray(mesh.visual.uv)[np.asarray(mesh.faces)].mean(axis=1)
            rows = np.clip(((1.0 - uv[:, 1]) * (height - 1)).astype(int), 0, height - 1)
            columns = np.clip((uv[:, 0] * (width - 1)).astype(int), 0, width - 1)
            face_colours = image[rows, columns]
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        owner = np.arange(len(faces))
        names = ["texture"]
        distinct = np.unique(face_colours // 24, axis=0)
        print(f"{len(faces)} faces, {len(distinct)} distinct colours at 24-level quantisation")
    elif args.labels:
        mesh = trimesh.load(os.path.abspath(args.glb), force="mesh")
        owner = np.load(os.path.abspath(args.labels))
        if len(owner) != len(mesh.faces):
            raise SystemExit(f"labels cover {len(owner)} faces but the mesh has {len(mesh.faces)}")
        if args.label_names:
            with open(os.path.abspath(args.label_names), "r", encoding="utf-8") as handle:
                names = json.load(handle)
        else:
            names = [str(i) for i in range(int(owner.max()) + 1)]
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        print(f"{len(names)} parts from labels: {names}")
        for index, name in enumerate(names):
            print(f"  {name}: {int((owner == index).sum())} faces")
    else:
        scene = trimesh.load(os.path.abspath(args.glb), force="scene")
        names = list(scene.geometry.keys())
        print(f"{len(names)} parts: {names}")

        vertices, faces, owner = [], [], []
        for index, name in enumerate(names):
            geometry = scene.geometry[name]
            transform = scene.graph.get(scene.graph.geometry_nodes[name][0])[0]
            points = trimesh.transform_points(np.asarray(geometry.vertices), transform)
            faces.append(np.asarray(geometry.faces) + sum(len(v) for v in vertices))
            vertices.append(points)
            owner.append(np.full(len(geometry.faces), index, dtype=np.int32))
            print(f"  {name}: {len(geometry.faces)} faces")

        owner = np.concatenate(owner)
        faces = np.concatenate(faces)
        vertices = np.concatenate(vertices)

    if args.up == "y":
        vertices = gltf_to_blender(vertices)
    elif args.up == "-z":
        vertices = vertices * np.array([1.0, 1.0, -1.0])
    vertices, _ = normalize_to_unit_cube(vertices)

    if args.explode:
        for index in range(len(names)):
            used = np.unique(faces[owner == index])
            if args.layout == "fan":
                angle = 2.0 * np.pi * index / len(names)
                direction = np.array([np.cos(angle), np.sin(angle), 0.0])
            else:
                direction = vertices[used].mean(axis=0)
                norm = np.linalg.norm(direction)
                direction = direction / norm if norm > 1e-6 else np.zeros(3)
            vertices[used] += args.explode * direction

    cameras = camera_ring(
        [float(v) for v in args.azimuths.split(",")],
        [float(v) for v in args.elevations.split(",")],
        radius=args.radius,
    )
    ids = rasterize_face_ids(vertices, faces, cameras, args.camera_angle_x, args.resolution)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    def paint(buffer, keep=None):
        canvas = np.full((*buffer.shape, 3), 255, dtype=np.uint8)
        hit = buffer > 0
        if args.texture:
            canvas[hit] = face_colours[buffer[hit] - 1]
            return canvas
        part = owner[buffer[hit] - 1]
        if keep is not None:
            hit[hit] = part == keep
            part = part[part == keep]
        canvas[hit] = LABEL_COLORS[part % len(LABEL_COLORS)].astype(np.uint8)
        return canvas

    for camera, buffer in zip(cameras, ids):
        path = os.path.join(out_dir, f"parts_{camera.name}.png")
        Image.fromarray(paint(buffer)).save(path)
        print(f"saved {path}")

    if args.sheet:
        for camera, buffer in zip(cameras, ids):
            tiles = [paint(buffer)] + [paint(buffer, keep=i) for i in range(len(names))]
            sheet = np.concatenate(tiles, axis=1)
            path = os.path.join(out_dir, f"sheet_{camera.name}.png")
            Image.fromarray(sheet).save(path)
            print(f"saved {path} ({['all'] + names})")


if __name__ == "__main__":
    main()
