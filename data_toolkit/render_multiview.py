"""Render a GLB from a deterministic camera grid, for per-view SAM3 prompting.

Unlike render_cond_view.py this needs no hand-found "front" azimuth: it sweeps a
fixed lat/long grid, so it behaves the same on any model.
"""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_toolkit.bpy_render import BpyRenderer
from data_toolkit.multiview import camera_ring


def parse_floats(text):
    return [float(value) for value in text.split(",") if value.strip() != ""]


def main():
    parser = argparse.ArgumentParser(description="Render a GLB from a deterministic view grid.")
    parser.add_argument("--glb", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--azimuths", default="0,45,90,135,180,225,270,315")
    parser.add_argument("--elevations", default="0,35")
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--camera_angle_x", type=float, default=0.6981317007977318)
    parser.add_argument(
        "--geo_mode",
        action="store_true",
        help="Flat grey override material and 1 sample: fast, and enough for a silhouette check.",
    )
    args = parser.parse_args()

    cameras = camera_ring(parse_floats(args.azimuths), parse_floats(args.elevations), args.radius)
    out_dir = os.path.abspath(args.out_dir)
    renderer = BpyRenderer(resolution=args.resolution, geo_mode=args.geo_mode)
    paths = renderer.render_from_positions(
        os.path.abspath(args.glb),
        [camera.position for camera in cameras],
        out_dir,
        args.camera_angle_x,
        names=[camera.name for camera in cameras],
    )

    manifest = {
        "camera_angle_x": args.camera_angle_x,
        "radius": args.radius,
        "resolution": args.resolution,
        "views": [
            {
                "name": camera.name,
                "azimuth": camera.azimuth,
                "elevation": camera.elevation,
                "position": [float(v) for v in camera.position],
                "image": os.path.basename(path),
            }
            for camera, path in zip(cameras, paths)
        ],
    }
    manifest_path = os.path.join(out_dir, "cameras.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"wrote {len(paths)} views and {manifest_path}")


if __name__ == "__main__":
    main()
