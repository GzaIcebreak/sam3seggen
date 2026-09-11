"""Give an untextured model a temporary flat colour so SAM3 has something to read.

An untextured model tells SAM3 almost nothing. On the robot's back 3/4 view it reports
"no instance" for both `head` and `torso`, so the whole back falls through to
`unassigned_to` and is named by a nearest-neighbour guess rather than by evidence.
Mickey, which ships a real albedo, is found in every view.

The temporary colour comes from one full_seg sample: its own part colouring, clustered
and re-mapped onto a palette whose entries are far apart, then rasterised through the
same camera the render used. Two properties matter:

* it is *flat*. This is the whole finding, not a detail: the same colours modulated by
  the render's luminance put `torso` back to "no instance" on that back view, and the
  shaded variant of full_seg's own colouring left it at 573 px against 3721 flat.
* it is rasterised, not baked. Painting vertex colours and re-rendering through Blender
  would reintroduce the axis swap baked into seg.glb; rasterising with SEG_TO_CAMERA is
  pixel-aligned with the source render by construction (silhouette agreement 1.000).

The colouring is a crutch for the *naming* stage only. It never touches the geometry,
which the split decides on its own.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_toolkit.meet_samples import DEFAULT_COLOR_TOL
from data_toolkit.multiview import normalize_to_unit_cube, rasterize_face_ids
from data_toolkit.parts_rebake import cluster_parts, face_base_colors, load_single_mesh
from data_toolkit.unit_vote import SEG_TO_CAMERA

# Measured mean foreground saturation: the untextured robot sits at 0.25-0.42 and
# Mickey at 43-53, so anything in between separates them. Two orders of magnitude of
# headroom means this needs no per-model tuning.
COLORLESS_SATURATION = 8.0

# Saturated and far apart in hue, so a boundary between two neighbouring parts survives
# even where the geometry gives no silhouette edge.
PAINT_COLORS = np.array([
    [214, 40, 40], [40, 90, 220], [40, 180, 80], [235, 195, 20],
    [30, 195, 205], [225, 70, 175], [140, 60, 200], [235, 130, 30],
    [120, 200, 60], [190, 120, 90], [90, 70, 190], [230, 160, 170],
], dtype=np.uint8)


def mean_saturation(views_dir, manifest):
    """Mean RGB spread over the rendered silhouettes: 0 for a grey model."""
    totals, pixels = 0.0, 0
    for view in manifest["views"]:
        image = np.asarray(Image.open(os.path.join(views_dir, view["image"])).convert("RGBA"))
        inside = image[..., 3] > 16
        if not inside.any():
            continue
        rgb = image[inside][..., :3].astype(np.float32)
        totals += float((rgb.max(axis=1) - rgb.min(axis=1)).sum())
        pixels += int(inside.sum())
    return totals / max(pixels, 1)


def is_colorless(views_dir, manifest, threshold=COLORLESS_SATURATION):
    saturation = mean_saturation(views_dir, manifest)
    print(f"  mean render saturation {saturation:.2f} "
          f"({'colourless' if saturation < threshold else 'textured'}, "
          f"threshold {threshold:g})")
    return saturation < threshold


def part_colors(seg_glb, color_tol=DEFAULT_COLOR_TOL):
    """Per-face paint colour from a full_seg sample, one palette entry per part.

    full_seg's own colours are not usable directly: on the robot every part came out a
    different near-identical green, which gives SAM3 no boundary to find. Clustering
    them first and then spending the palette keeps the part count it predicted while
    making the parts actually distinguishable.

    The tolerance is the meet's, not parts_rebake's 40: at 40 the robot's arms clustered
    into the same colour as its torso and SAM3's arm mask collapsed with them.
    """
    mesh = load_single_mesh(seg_glb)
    labels, _ = cluster_parts(face_base_colors(mesh), np.asarray(mesh.area_faces), color_tol)
    print(f"  {labels.max() + 1} colours from {os.path.basename(seg_glb)}")
    return mesh, PAINT_COLORS[labels % len(PAINT_COLORS)]


def paint_views(seg_glb, views_dir, out_dir, manifest, cameras, color_tol=DEFAULT_COLOR_TOL):
    """Write a flat-painted copy of every view, aligned to the same cameras."""
    mesh, colors = part_colors(seg_glb, color_tol)
    resolution = int(manifest["resolution"])
    vertices, _ = normalize_to_unit_cube(np.asarray(mesh.vertices) @ SEG_TO_CAMERA.T)
    face_ids = rasterize_face_ids(vertices, np.asarray(mesh.faces), cameras,
                                  float(manifest["camera_angle_x"]), resolution)
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(os.path.join(views_dir, "cameras.json"),
                os.path.join(out_dir, "cameras.json"))
    for index, view in enumerate(manifest["views"]):
        ids = face_ids[index]
        inside = ids > 0
        image = np.zeros((resolution, resolution, 4), dtype=np.uint8)
        image[inside, :3] = colors[ids[inside] - 1]
        image[inside, 3] = 255
        Image.fromarray(image).save(os.path.join(out_dir, view["image"]))
    print(f"  flat-painted {len(manifest['views'])} views -> {out_dir}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seg_glb", required=True, help="one full_seg sample (seg.glb)")
    parser.add_argument("--views_dir", required=True, help="render_multiview.py output")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--color_tol", type=float, default=DEFAULT_COLOR_TOL)
    args = parser.parse_args()

    from data_toolkit.lift_sam3 import load_cameras

    views_dir = os.path.abspath(args.views_dir)
    manifest, cameras = load_cameras(views_dir)
    print(f"saturation check on {views_dir}")
    is_colorless(views_dir, manifest)
    paint_views(os.path.abspath(args.seg_glb), views_dir, os.path.abspath(args.out_dir),
                manifest, cameras, args.color_tol)


if __name__ == "__main__":
    main()
