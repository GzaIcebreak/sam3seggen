"""Full segmentation first, semantics second: a 3D model + text prompts -> named objects.

Pipeline (the inverse of segment_api.py, which lets a single 2D map steer the
generative model directly):

    1. SegviGen full_seg checkpoint, no prompts: clean anonymous geometric parts
    2. Render the split model over a fixed multi-view camera grid
    3. SAM3 segments every view with the requested prompts (-> raw masks .npz)
    4. Each anonymous part inherits the name whose masks covered most of its pixels
       (per-part voting -- a single mislabelled triangle can never tear a boundary,
       because merging whole parts is the only operation available)
    5. Parts that share a name merge into one object; the combined glb keeps every
       part's baked texture

Run with the SegviGen venv (.venv); SAM3 is dispatched to --py_sam3 like in
segment_api.py.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from prompt_specs import normalize_part_specs, part_names, validate_target_name
from segment_api import DEFAULT_PY_SAM3, DEFAULT_SAM3, _run


def segment_vote(
    glb,
    prompts,
    out_glb,
    work_dir=None,
    azimuths="0,45,90,135,180,225,270,315",
    elevations="0,35",
    radius=2.0,
    resolution=512,
    py_sam3=None,
    sam3_model=DEFAULT_SAM3,
    sam3_threshold=0.3,
    unassigned_to=None,
    min_cover=0.25,
    strict_parts=True,
    with_texture=True,
):
    """Segment `glb` into named objects via full segmentation + multi-view SAM3 voting.

    Args mirror segment_api.segment(); `azimuths`/`elevations`/`radius` define the
    camera grid SAM3 votes over (no per-model "front" azimuth is needed, which is
    the whole point of voting across views). `min_cover` keeps a part unassigned
    unless its winning name's masks covered that fraction of the pixels it owns.
    """
    from data_toolkit.part_vote import (
        assign_parts,
        load_parts,
        merge_parts,
        vote_parts,
    )
    from data_toolkit.lift_sam3 import load_cameras, load_masks

    specs = normalize_part_specs(prompts)
    names = part_names(specs)
    validate_target_name(unassigned_to, names)
    canonical_prompts = [
        name if concepts == [name] else f"{name}={'+'.join(concepts)}"
        for name, concepts in specs
    ]

    glb = os.path.abspath(glb)
    out_glb = os.path.abspath(out_glb)
    py_sam3 = py_sam3 or DEFAULT_PY_SAM3
    py_self = sys.executable

    if work_dir is None:
        work_dir = os.path.join(os.path.dirname(out_glb) or ".", "work_vote")
    work_dir = os.path.abspath(work_dir)
    views_dir = os.path.join(work_dir, "views")
    parts_glb = os.path.join(work_dir, "parts_nosam.glb")
    masks_npz = os.path.join(work_dir, "masks.npz")
    os.makedirs(views_dir, exist_ok=True)

    # Step 1 imports torch/o_voxel and takes minutes; everything else is cheap, so
    # skip it when a previous run's split is still around.
    if os.path.exists(parts_glb):
        print(f"[1/5] reusing full segmentation {parts_glb}")
    else:
        print("[1/5] SegviGen full segmentation (no prompts) ...")
        from segment_api import segment

        segment(
            glb, None, parts_glb,
            with_texture=with_texture,
            work_dir=os.path.join(work_dir, "nosam"),
            use_sam3=False,
            strict_parts=False,
        )

    print(f"[2/5] rendering view grid ({azimuths} x {elevations}) ...")
    _run([
        py_self, os.path.join(ROOT, "data_toolkit", "render_multiview.py"),
        "--glb", parts_glb, "--out_dir", views_dir,
        "--azimuths", azimuths, "--elevations", elevations,
        "--radius", radius, "--resolution", resolution,
    ])

    print(f"[3/5] SAM3 prompts {canonical_prompts} over the view grid ...")
    sam3_cmd = [
        py_sam3, os.path.join(ROOT, "sam3_multiview.py"),
        "--views_dir", views_dir, "--out", masks_npz,
        "--model", sam3_model, "--threshold", sam3_threshold,
    ]
    if unassigned_to:
        sam3_cmd += ["--unassigned_to", unassigned_to]
    _run(sam3_cmd + ["--prompts", *canonical_prompts])

    print("[4/5] voting parts by SAM3 coverage ...")
    mask_set = load_masks(masks_npz)
    _, cameras = load_cameras(views_dir)
    with open(os.path.join(views_dir, "cameras.json"), "r", encoding="utf-8") as handle:
        import json
        camera_angle_x = float(json.load(handle)["camera_angle_x"])
    _, mesh, owner, node_names = load_parts(parts_glb)
    votes, vote_names, pixels_owned = vote_parts(
        mesh, owner, mask_set, cameras, camera_angle_x, resolution)
    # vote_names follow first-seen concept order; the manifest keeps the requested order
    for index, node in enumerate(node_names):
        tally = ", ".join(f"{n}={v:.0f}" for n, v in zip(vote_names, votes[index]) if v > 0)
        cover = votes[index].max() / pixels_owned[index] if pixels_owned[index] else 0.0
        print(f"  {node}: {tally or 'no votes'} (best coverage {cover:.2f})")
    assignment = assign_parts(votes, vote_names, pixels_owned, unassigned_to, min_cover)

    print("[5/5] merging parts by name ...")
    manifest = merge_parts(parts_glb, assignment, names, out_glb, strict=strict_parts)
    print(f"saved {out_glb} ({len(manifest)} objects)")
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="model + text prompts -> named objects, via full segmentation + SAM3 part voting"
    )
    parser.add_argument("--glb", required=True, help="Input 3D model")
    parser.add_argument("--prompts", nargs="+", required=True,
                        help="One entry per output object; join concepts with '+' to merge them, "
                             "e.g. 'mushroom chair' or 'body=head+face+hand'.")
    parser.add_argument("--out", required=True, help="Output glb (one named object per prompt entry)")
    parser.add_argument("--work_dir", default=None,
                        help="Keep intermediates (nosam split, view renders, masks) here. "
                             "Default: work_vote/ next to --out.")
    parser.add_argument("--azimuths", default="0,45,90,135,180,225,270,315")
    parser.add_argument("--elevations", default="0,35")
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--unassigned_to", default=None,
                        help="Name of the prompt entry that absorbs parts no concept claimed.")
    parser.add_argument("--min_cover", type=float, default=0.25,
                        help="A part stays unassigned unless its winning name's masks covered "
                             "this fraction of the pixels it owns across the view grid.")
    parser.add_argument("--allow_partial", action="store_true",
                        help="Allow requested names that claimed no part, or leftover unnamed parts.")
    parser.add_argument("--no_texture", action="store_true",
                        help="Skip the texture bake in step 1 (faster; SAM3 then votes on "
                             "flat-colour renders, which usually still works).")
    parser.add_argument("--py_sam3", default=None, help=f"default: {DEFAULT_PY_SAM3}")
    parser.add_argument("--sam3_model", default=DEFAULT_SAM3)
    parser.add_argument("--sam3_threshold", type=float, default=0.3)
    args = parser.parse_args()

    segment_vote(
        args.glb, args.prompts, args.out,
        work_dir=args.work_dir,
        azimuths=args.azimuths,
        elevations=args.elevations,
        radius=args.radius,
        resolution=args.resolution,
        py_sam3=args.py_sam3,
        sam3_model=args.sam3_model,
        sam3_threshold=args.sam3_threshold,
        unassigned_to=args.unassigned_to,
        min_cover=args.min_cover,
        strict_parts=not args.allow_partial,
        with_texture=not args.no_texture,
    )


if __name__ == "__main__":
    main()
