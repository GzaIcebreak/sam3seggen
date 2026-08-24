"""Clean a segmented part: keep only the components that actually own pixels of the
part's colour on the SAM3 2D map, i.e. components visible from the conditioning
camera along rays through the part's mask region.

Rationale: the 2D-guided flow model colours unseen back-side voxels too, so a part
can pick up stray fragments. Those fragments sit *behind* the real part along the
same camera rays, so rasterising the whole model from the conditioning camera
(face-ID z-buffer) separates them: the true part owns the pixels under its mask
colour; the fragments own none.

The camera convention is pinned empirically against the conditioning render's alpha
(IoU ~0.99): gltf_to_blender + unit-cube normalisation + transforms.json camera,
world_up=Z, flip_rows=True, flip_cols=False (see multiview.rasterize_face_ids).
"""
import argparse
import json
import os
import sys

import numpy as np
import trimesh
from PIL import Image, ImageFilter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_toolkit.multiview import (
    Camera,
    gltf_to_blender,
    normalize_to_unit_cube,
    rasterize_face_ids,
)
from data_toolkit.project_2d import camera_from_transforms


def clean_part(parts_glb, node_name, map_path, map_rgb, transforms_path, azimuth,
               out_glb, dilate=6, min_pixels=10, keep_min_faces=8):
    scene = trimesh.load(parts_glb)
    names = list(scene.geometry.keys())
    if node_name not in names:
        raise ValueError(f"node {node_name} not in {parts_glb}")

    canvas = np.asarray(Image.open(map_path).convert("RGB")).astype(np.int32)
    target = np.array(map_rgb, dtype=np.int32)
    mask = np.linalg.norm(canvas - target, axis=2) < 60
    mask = np.asarray(Image.fromarray(mask).filter(ImageFilter.MaxFilter(2 * dilate + 1)))
    resolution = mask.shape[0]

    verts, faces, owner = [], [], []
    world_meshes = {}
    for index, name in enumerate(names):
        geometry = scene.geometry[name]
        matrix = scene.graph.get(scene.graph.geometry_nodes[name][0])[0]
        points = trimesh.transform_points(np.asarray(geometry.vertices), matrix)
        world_meshes[name] = geometry.copy().apply_transform(matrix)
        faces.append(np.asarray(geometry.faces) + sum(len(v) for v in verts))
        verts.append(points)
        owner.append(np.full(len(geometry.faces), index, dtype=np.int64))
    verts = np.concatenate(verts)
    faces = np.concatenate(faces)
    owner = np.concatenate(owner)
    target_index = names.index(node_name)

    with open(transforms_path, "r", encoding="utf-8") as handle:
        entry = json.load(handle)[0]
    cam_pos, _, _, _ = camera_from_transforms(entry, azimuth=azimuth, resolution=resolution)
    vv, _ = normalize_to_unit_cube(gltf_to_blender(verts))
    ids = rasterize_face_ids(
        vv, faces, [Camera(azimuth, 0.0, cam_pos)], float(entry["camera_angle_x"]),
        resolution,
    )[0]

    # faces of the target part that own a pixel, and how many of those pixels are on-mask
    hit = ids > 0
    face_ids = ids[hit] - 1
    pixel_on_mask = mask[hit]
    owned = face_ids[owner[face_ids] == target_index]
    owned_on_mask = face_ids[(owner[face_ids] == target_index) & pixel_on_mask]
    owned_count = np.bincount(owned, minlength=len(faces))
    mask_count = np.bincount(owned_on_mask, minlength=len(faces))

    mesh = world_meshes[node_name]
    offset = sum(len(world_meshes[n].faces) for n in names[:target_index])
    local_owned = owned_count[offset:offset + len(mesh.faces)]
    local_mask = mask_count[offset:offset + len(mesh.faces)]
    print(f"{node_name}: {int(local_mask.sum())} on-mask pixels, "
          f"{int((local_owned > 0).sum())}/{len(mesh.faces)} faces visible")

    kept_faces = []
    for comp_faces in trimesh.graph.connected_components(mesh.face_adjacency):
        comp_faces = np.asarray(comp_faces)
        if len(comp_faces) < keep_min_faces:
            continue
        on = local_mask[comp_faces].sum()
        total = local_owned[comp_faces].sum()
        if on >= min_pixels and on >= 0.3 * max(total, 1):
            kept_faces.append(comp_faces)
    n_kept = sum(len(f) for f in kept_faces)
    print(f"{node_name}: kept {len(kept_faces)} components, {n_kept}/{len(mesh.faces)} faces")
    if not kept_faces:
        raise ValueError("nothing kept -- check azimuth / map colour")
    out = mesh.submesh([np.concatenate(kept_faces)], append=True)
    out.export(out_glb)
    print(f"saved {out_glb}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--glb", required=True, help="Combined parts glb (one node per part)")
    parser.add_argument("--node", required=True, help="Part node to clean, e.g. part_00_mushroom")
    parser.add_argument("--map", required=True, help="SAM3 2D map used as the conditioning image")
    parser.add_argument("--color", required=True, nargs=3, type=int,
                        help="The part's RGB colour on the map, from the legend json")
    parser.add_argument("--transforms", required=True)
    parser.add_argument("--azimuth", type=float, default=0.0,
                        help="Azimuth the conditioning render/map were made with")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    clean_part(args.glb, args.node, args.map, args.color, args.transforms,
               args.azimuth, args.out)


if __name__ == "__main__":
    main()
