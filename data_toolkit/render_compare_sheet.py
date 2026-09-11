"""Three-panel sheet: original albedo, part stain, exploded repaired solids.

`--completed` is the X-Part (or otherwise closed) glb; the third panel pulls those
solids apart. Without it the open split is exploded instead.

parts.glb is in the source glTF frame; scene-graph transforms are baked first so a
model that stores parts in local nodes does not come out on its side.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_toolkit.multiview import (
    GLTF_TO_BLENDER, camera_ring, normalize_to_unit_cube, rasterize_face_ids,
)
from data_toolkit.parts_rebake import LABEL_COLORS

CAMERA_ANGLE = 0.6981317007977318


def bake_nodes(scene):
    names, meshes = [], []
    for name in sorted(scene.geometry):
        mesh = scene.geometry[name].copy()
        if name in scene.graph.geometry_nodes:
            transform, _ = scene.graph.get(scene.graph.geometry_nodes[name][0])
            mesh.apply_transform(transform)
        names.append(name)
        meshes.append(mesh)
    return names, meshes


def albedo_face_colors(mesh):
    visual = mesh.visual
    image = getattr(getattr(visual, "material", None), "baseColorTexture", None)
    if image is None or getattr(visual, "uv", None) is None:
        if getattr(visual, "kind", None) == "vertex":
            return np.asarray(visual.vertex_colors)[:, :3][
                np.asarray(mesh.faces)].mean(axis=1).astype(np.float32)
        return np.full((len(mesh.faces), 3), 180, dtype=np.float32)
    tex = np.asarray(image.convert("RGB"))
    height, width = tex.shape[:2]
    uv = np.asarray(visual.uv, dtype=np.float64)
    centroids = uv[mesh.faces].mean(axis=1)
    columns = np.clip((centroids[:, 0] % 1.0) * (width - 1), 0, width - 1).astype(int)
    rows = np.clip((1.0 - centroids[:, 1] % 1.0) * (height - 1), 0, height - 1).astype(int)
    return tex[rows, columns].astype(np.float32)


def explode_vertices(vertices, faces, owner, amount):
    if not amount:
        return vertices
    moved = vertices.copy()
    count = int(owner.max()) + 1
    for index in range(count):
        used = np.unique(faces[owner == index])
        if not len(used):
            continue
        angle = 2.0 * np.pi * index / max(count, 1)
        direction = np.array([np.cos(angle), np.sin(angle), 0.15])
        direction = direction / np.linalg.norm(direction)
        moved[used] += amount * direction
    return moved


def paint(face_ids, colors, normals, camera, resolution):
    light = np.asarray(camera.position, dtype=np.float64)
    light = light / np.linalg.norm(light)
    shade = 0.55 + 0.45 * np.clip(normals @ light, 0.0, 1.0)
    canvas = np.full((resolution, resolution, 3), 255, dtype=np.uint8)
    inside = face_ids > 0
    hit = face_ids[inside] - 1
    canvas[inside] = np.clip(colors[hit] * shade[hit][:, None], 0, 255).astype(np.uint8)
    return canvas


def caption(image, text):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 22)
    except OSError:
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
        except OSError:
            font = ImageFont.load_default()
    draw.rectangle([0, 0, image.width, 32], fill=(255, 255, 255))
    draw.text((10, 5), text, fill=(30, 30, 30), font=font)
    return image


def has_texture(meshes):
    return any(
        getattr(getattr(item.visual, "material", None), "baseColorTexture", None) is not None
        for item in meshes
    )


def pack(meshes):
    mesh = trimesh.util.concatenate(meshes)
    owner = np.concatenate(
        [np.full(len(item.faces), index, dtype=np.int32)
         for index, item in enumerate(meshes)])
    stain = LABEL_COLORS[owner % len(LABEL_COLORS)].astype(np.float32)
    albedo = np.concatenate([albedo_face_colors(item) for item in meshes])
    return mesh, owner, stain, albedo if has_texture(meshes) else stain


def render_sheet(source_glb, parts_glb, out_png, azimuth=20.0, elevation=15.0,
                 explode=0.45, resolution=640, radius=2.3, completed_glb=None):
    _, source_meshes = bake_nodes(trimesh.load(source_glb, process=False))
    part_names, part_meshes = bake_nodes(trimesh.load(parts_glb, process=False))
    source = trimesh.util.concatenate(source_meshes)
    src_albedo = np.concatenate([albedo_face_colors(item) for item in source_meshes])
    parts, part_owner, part_stain, _ = pack(part_meshes)

    if completed_glb:
        done_names, done_meshes = bake_nodes(trimesh.load(completed_glb, process=False))
        done, done_owner, done_stain, done_color = pack(done_meshes)
        explode_title = "修复后"
    else:
        done_names, done, done_owner, done_color = part_names, parts, part_owner, part_stain
        explode_title = "爆照"

    src_vertices, _ = normalize_to_unit_cube(
        np.asarray(source.vertices) @ GLTF_TO_BLENDER.T)
    part_vertices, _ = normalize_to_unit_cube(
        np.asarray(parts.vertices) @ GLTF_TO_BLENDER.T)
    done_vertices, _ = normalize_to_unit_cube(
        np.asarray(done.vertices) @ GLTF_TO_BLENDER.T)
    src_faces = np.asarray(source.faces)
    part_faces = np.asarray(parts.faces)
    done_faces = np.asarray(done.faces)
    exploded = explode_vertices(done_vertices, done_faces, done_owner, explode)

    src_normals = np.asarray(source.face_normals) @ GLTF_TO_BLENDER.T
    part_normals = np.asarray(parts.face_normals) @ GLTF_TO_BLENDER.T
    done_normals = np.asarray(done.face_normals) @ GLTF_TO_BLENDER.T
    cameras = camera_ring([azimuth], [elevation], radius)
    camera = cameras[0]

    panels = [
        ("original", paint(
            rasterize_face_ids(src_vertices, src_faces, cameras, CAMERA_ANGLE, resolution)[0],
            src_albedo, src_normals, camera, resolution), "原模型"),
        ("stain", paint(
            rasterize_face_ids(part_vertices, part_faces, cameras, CAMERA_ANGLE, resolution)[0],
            part_stain, part_normals, camera, resolution), "染色"),
        ("repaired", paint(
            rasterize_face_ids(exploded, done_faces, cameras, CAMERA_ANGLE, resolution)[0],
            done_color, done_normals, camera, resolution), explode_title),
    ]
    images = [caption(Image.fromarray(canvas), title) for _, canvas, title in panels]
    sheet = Image.new("RGB", (resolution * 3, resolution), (255, 255, 255))
    for index, image in enumerate(images):
        sheet.paste(image, (index * resolution, 0))
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    sheet.save(out_png)
    print(f"saved {out_png} ({len(part_names)} parts / {len(done_names)} repaired)")
    return out_png


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True)
    parser.add_argument("--parts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--azimuth", type=float, default=20.0)
    parser.add_argument("--elevation", type=float, default=15.0)
    parser.add_argument("--explode", type=float, default=0.45)
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--radius", type=float, default=2.3)
    parser.add_argument("--completed", default=None,
                        help="Closed solids (X-Part xpart_parts.glb) for the third panel")
    args = parser.parse_args()
    render_sheet(os.path.abspath(args.source), os.path.abspath(args.parts),
                 os.path.abspath(args.out), args.azimuth, args.elevation, args.explode,
                 args.resolution, args.radius,
                 os.path.abspath(args.completed) if args.completed else None)


if __name__ == "__main__":
    main()
