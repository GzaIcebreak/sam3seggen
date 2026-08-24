"""Prove the nvdiffrast camera matches the Blender camera, by silhouette overlap.

Back-projecting SAM3 pixels onto faces is only as good as this agreement, so the
convention (which world axis is up, which way the raster buffer runs) is measured
here rather than reasoned about.
"""
import argparse
import itertools
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
    Camera,
    gltf_to_blender,
    normalize_to_unit_cube,
    rasterize_face_ids,
)

CONVENTIONS = list(
    itertools.product([(0.0, 0.0, 1.0), (0.0, 1.0, 0.0)], [True, False], [False, True])
)


def load_reference(out_dir):
    with open(os.path.join(out_dir, "cameras.json"), "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    cameras, masks = [], []
    for view in manifest["views"]:
        image = np.asarray(Image.open(os.path.join(out_dir, view["image"])).convert("RGBA"))
        masks.append(image[..., 3] > 127)
        cameras.append(Camera(view["azimuth"], view["elevation"], np.array(view["position"])))
    return manifest, cameras, masks


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def main():
    parser = argparse.ArgumentParser(description="Silhouette agreement between Blender and nvdiffrast.")
    parser.add_argument("--glb", required=True)
    parser.add_argument("--views_dir", required=True)
    args = parser.parse_args()

    manifest, cameras, masks = load_reference(os.path.abspath(args.views_dir))
    reference = np.stack(masks)

    mesh = trimesh.load(os.path.abspath(args.glb), force="mesh")
    vertices, _ = normalize_to_unit_cube(gltf_to_blender(mesh.vertices))
    print(f"mesh: {len(mesh.faces)} faces, {len(vertices)} vertices, {len(cameras)} views")

    best = None
    for world_up, flip_rows, flip_cols in CONVENTIONS:
        ids = rasterize_face_ids(
            vertices,
            mesh.faces,
            cameras,
            manifest["camera_angle_x"],
            resolution=manifest["resolution"],
            world_up=world_up,
            flip_rows=flip_rows,
            flip_cols=flip_cols,
        )
        scores = [iou(reference[i], ids[i] > 0) for i in range(len(cameras))]
        mean = float(np.mean(scores))
        label = f"up={tuple(int(v) for v in world_up)} flip_rows={flip_rows} flip_cols={flip_cols}"
        print(f"{label:>46}  mean IoU {mean:.4f}  worst {min(scores):.4f}")
        if best is None or mean > best[0]:
            best = (mean, label, scores)

    print(f"\nbest: {best[1]}  mean IoU {best[0]:.4f}")
    for camera, score in zip(cameras, best[2]):
        print(f"  {camera.name:>16}  IoU {score:.4f}")


if __name__ == "__main__":
    main()
