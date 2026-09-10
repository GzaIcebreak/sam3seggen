"""Render a SegviGen conditioning view with Blender (bpy)."""
import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_toolkit.bpy_render import render_from_transforms


def main():
    parser = argparse.ArgumentParser(description="Render GLB from transforms.json for SAM3 / SegviGen.")
    parser.add_argument("--glb", required=True)
    parser.add_argument("--transforms", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument(
        "--ref_glb",
        default=None,
        help="Normalize scale/offset from this glb's bbox instead of --glb's own. Use the full "
             "assembly here when rendering a single split-out part, so it keeps its position "
             "relative to the whole body instead of being re-centred on itself.",
    )
    parser.add_argument(
        "--azimuths",
        default="0",
        help="Comma-separated degrees to orbit transforms.json's camera around the up axis. "
             "With data_toolkit/transforms.json and monk.glb, 0 (the calibrated camera) looks "
             "at the model's BACK, ~135 is head-on front, and 90/180 are front 3/4 views from "
             "either side; '0,135,225' is a reasonable turntable. The front offset depends on "
             "how the model itself is oriented, so render a probe before trusting any value. "
             "With more than one value, --out is used as a template: 'view.png' becomes "
             "'view_0.png', 'view_135.png', ...",
    )
    parser.add_argument("--samples", type=int, default=None,
                        help="Cycles samples (default 128). Lower values are for seed-picking previews.")
    args = parser.parse_args()

    glb = os.path.abspath(args.glb)
    transforms = os.path.abspath(args.transforms)
    out = os.path.abspath(args.out)
    ref_glb = os.path.abspath(args.ref_glb) if args.ref_glb else None
    azimuths = [float(a) for a in args.azimuths.split(",") if a.strip() != ""]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f"rendering {glb} -> {out}")
    written = render_from_transforms(glb, transforms, out, resolution=args.resolution, ref_glb=ref_glb,
                                    azimuths=azimuths, samples=args.samples)
    for path in written:
        print(f"saved {path}")


if __name__ == "__main__":
    main()
