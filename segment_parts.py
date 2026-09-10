"""A 3D model (+ text prompts) -> one glb with geometrically clean parts.

This replaces segment_api.segment (2D map steers the generative model) and
segment_vote.segment_vote (one full_seg sample, coverage voting). Both decided part
boundaries from evidence that can be wrong per triangle; here geometry decides every
boundary and language only chooses names:

    1. N prompt-free full_seg samples, the conditioning camera jittered around the front
    2. their partitions are intersected ("meet"), which keeps every cut any sample drew
       -> deliberately over-segmented atoms that no later step has to cut again
    3. naming, which is merge_parts.py and needs the prompts: multi-view SAM3 masks vote
       on each atom component, then same-named units are fused and exported.

Steps 1-2 are the split and cost minutes of GPU; step 3 is the merge and costs seconds
once its renders are cached. `--merge off` stops after the split and writes one node per
unit, so the two halves can be looked at, and argued about, separately:

    python segment_parts.py --glb robot.glb --merge off --out split/units.glb
    python merge_parts.py --glb robot.glb --split split/work --prompts head torso arm \
        --out named/parts.glb

Why over-segment first: full_seg has no granularity control and a single sample fuses
neighbouring parts often enough to matter (on the robot test model, shoulder armour and
both arms came out as one 24k-face atom in every same-view sample). Over-segmentation is
free for the merge, which can only fuse atoms, while under-segmentation is unrecoverable.

Run with the SegviGen venv; SAM3 is dispatched to --py_sam3 as in segment_api.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from merge_parts import (
    DEFAULT_RADIUS, DEFAULT_RESOLUTION, DEFAULT_SAM3_THRESHOLD, DEFAULT_VIEW_AZIMUTHS,
    DEFAULT_VIEW_ELEVATIONS, export_labelled, merge_parts,
)
from segment_api import DEFAULT_PY_SAM3, DEFAULT_SAM3, DEFAULT_TRANSFORMS, _run

DEFAULT_CKPT = os.path.join(ROOT, "ckpt", "full_seg.ckpt")
DEFAULT_SAMPLES = 5
DEFAULT_AZIMUTH_JITTER = 30.0


def sample_azimuths(count, azimuth, jitter):
    """Conditioning views for the full_seg samples: the base view first, then jittered.

    Noise alone is not enough diversity -- three samples of the same view fused the same
    two parts every time -- but full_seg only holds up near the front (at 90 degrees it
    painted the whole robot one colour), so the jitter stays inside a narrow window.
    """
    offsets = [0.0]
    if jitter:
        offsets += [jitter, -jitter, jitter / 2.0, -jitter / 2.0]
    return [azimuth + offsets[index % len(offsets)] for index in range(count)]


def segment_parts(
    glb,
    prompts,
    out_glb,
    work_dir=None,
    samples=DEFAULT_SAMPLES,
    azimuth=0.0,
    azimuth_jitter=DEFAULT_AZIMUTH_JITTER,
    ckpt=None,
    transforms=None,
    color_tol=None,
    min_atom_faces=None,
    mirror="auto",
    min_unit_faces=None,
    min_recall=None,
    view_azimuths=DEFAULT_VIEW_AZIMUTHS,
    view_elevations=DEFAULT_VIEW_ELEVATIONS,
    radius=DEFAULT_RADIUS,
    resolution=DEFAULT_RESOLUTION,
    py_sam3=None,
    sam3_model=DEFAULT_SAM3,
    sam3_threshold=DEFAULT_SAM3_THRESHOLD,
    unassigned_to=None,
    merge="name",
    reuse=True,
    strict_parts=True,
    with_texture=True,
    texture_size=2048,
):
    """Segment `glb` into parts and write them all into `out_glb`.

    Args:
        prompts: one entry per output part; join concepts with '+' to merge them into one
            part, optionally under a name ("body=head+face+hand"). Only `merge="off"`
            accepts no prompts at all.
        samples: how many full_seg samples to intersect. 1 reproduces the old single-sample
            behaviour; the fused-part failures start disappearing at 3 and are gone at 5.
        azimuth / azimuth_jitter: the conditioning camera, and how far the extra samples
            orbit either side of it. See sample_azimuths.
        mirror: "auto" also intersects each sample reflected across the model's symmetry
            plane, if it has one -- full_seg often cuts a joint on one side only.
        merge: "off" stops after the split and writes one node per unit; "name" and "unit"
            hand over to merge_parts.py (see its `merge`).
        view_azimuths / view_elevations / radius / resolution: the grid SAM3 votes over.
            No per-model front view is needed, which is the point of voting across views.
        unassigned_to: name of the part absorbing units no concept claimed. Without it
            those faces are dropped from the output.
        with_texture: bake the source albedo onto each part (needs bpy); off gives each
            part a flat placeholder colour instead.

    Returns a parts.json-style manifest, one row per exported part.
    """
    from data_toolkit.meet_samples import (
        DEFAULT_COLOR_TOL, DEFAULT_MIN_FACES, atoms_debug_glb, meet_samples,
    )
    from data_toolkit.parts_rebake import welded_face_adjacency
    from data_toolkit.unit_vote import DEFAULT_MIN_UNIT_FACES, split_units

    import numpy as np

    if samples < 1:
        raise ValueError(f"samples must be at least 1, got {samples}")
    if merge != "off" and not prompts:
        raise ValueError("prompts are required unless merge='off'")

    glb = os.path.abspath(glb)
    out_glb = os.path.abspath(out_glb)
    out_dir = os.path.dirname(out_glb) or "."
    ckpt = os.path.abspath(ckpt or DEFAULT_CKPT)
    transforms = os.path.abspath(transforms or DEFAULT_TRANSFORMS)
    py_sam3 = py_sam3 or DEFAULT_PY_SAM3
    py_self = sys.executable
    color_tol = DEFAULT_COLOR_TOL if color_tol is None else color_tol
    min_atom_faces = DEFAULT_MIN_FACES if min_atom_faces is None else min_atom_faces
    min_unit_faces = DEFAULT_MIN_UNIT_FACES if min_unit_faces is None else min_unit_faces

    work_dir = os.path.abspath(work_dir or os.path.join(out_dir, "work_parts"))
    atoms_npy = os.path.join(work_dir, "atoms.npy")
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    azimuths = sample_azimuths(samples, azimuth, azimuth_jitter)
    sample_glbs = []
    for index, view_azimuth in enumerate(azimuths):
        sample_dir = os.path.join(work_dir, f"sample_{index:02d}")
        os.makedirs(sample_dir, exist_ok=True)
        seg_glb = os.path.join(sample_dir, "seg.glb")
        sample_glbs.append(seg_glb)
        # Each sample costs a full flow-model run, so a rerun picks up where it stopped.
        if reuse and os.path.exists(seg_glb):
            print(f"[1/2] reusing full_seg sample {index + 1}/{samples} ({seg_glb})")
            continue
        print(f"[1/2] full_seg sample {index + 1}/{samples}, azimuth {view_azimuth:g} ...")
        _run([
            py_self, os.path.join(ROOT, "inference_full.py"),
            "--ckpt_path", ckpt, "--glb", glb,
            "--input_vxz", os.path.join(sample_dir, "input.vxz"),
            "--img", os.path.join(sample_dir, "render.png"),
            "--export_glb", seg_glb,
            "--transforms", transforms, "--azimuth", view_azimuth,
        ])

    print(f"[2/2] intersecting {samples} partitions into atoms ...")
    reference, atoms, report = meet_samples(sample_glbs, color_tol, min_atom_faces, mirror)
    np.save(atoms_npy, atoms)
    atoms_debug_glb(reference, atoms, os.path.join(work_dir, "atoms.glb"))
    with open(os.path.join(work_dir, "atoms_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    mirrored = (f"mirrored about {report['mirror_axis']}" if report["mirror_axis"]
                else f"not mirrored (symmetry {report['mirror_share']:.2f})")
    print(f"  labels per sample {report['labels_per_sample']}, {mirrored} -> "
          f"{report['atoms']} atoms, largest {report['largest_share']:.1%} of the area")

    if merge != "off":
        return merge_parts(
            glb, prompts, work_dir, out_glb,
            mesh=sample_glbs[0], atoms=atoms_npy,
            unassigned_to=unassigned_to, merge=merge,
            min_unit_faces=min_unit_faces, min_recall=min_recall,
            view_azimuths=view_azimuths, view_elevations=view_elevations,
            radius=radius, resolution=resolution,
            py_sam3=py_sam3, sam3_model=sam3_model, sam3_threshold=sam3_threshold,
            reuse=reuse, strict_parts=strict_parts,
            with_texture=with_texture, texture_size=texture_size,
        )

    units = split_units(reference, atoms, welded_face_adjacency(reference), min_unit_faces)
    labels_npy = os.path.join(work_dir, "labels.npy")
    names_json = os.path.join(work_dir, "label_names.json")
    np.save(labels_npy, units)
    with open(names_json, "w", encoding="utf-8") as handle:
        json.dump([f"unit_{unit:02d}" for unit in range(units.max() + 1)], handle, indent=2)
    manifest = export_labelled(sample_glbs[0], glb, labels_npy, names_json, out_glb,
                               with_texture, texture_size, step="[2/2]")
    print(f"saved {out_glb} ({len(manifest)} units, unnamed)")
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="model + text prompts -> named parts, via over-segmentation + SAM3 voting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--glb", required=True, help="Input 3D model")
    parser.add_argument("--prompts", nargs="+", default=[],
                        help="One entry per output part; join concepts with '+' to merge them, "
                             "e.g. 'head torso body=arm+hand'. Not needed with --merge off.")
    parser.add_argument("--out", required=True, help="Output glb, one node per part")
    parser.add_argument("--work_dir", default=None,
                        help="Keep intermediates here. Default: work_parts/ next to --out.")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                        help="full_seg samples to intersect; each costs one flow-model run")
    parser.add_argument("--azimuth", type=float, default=0.0,
                        help="Degrees to orbit transforms.json's camera for the base sample")
    parser.add_argument("--azimuth_jitter", type=float, default=DEFAULT_AZIMUTH_JITTER,
                        help="How far the extra samples orbit either side of --azimuth")
    parser.add_argument("--ckpt", default=None, help=f"default: {DEFAULT_CKPT}")
    parser.add_argument("--transforms", default=None, help=f"default: {DEFAULT_TRANSFORMS}")
    parser.add_argument("--color_tol", type=float, default=None,
                        help="RGB distance separating two colours within one sample")
    parser.add_argument("--min_atom_faces", type=int, default=None,
                        help="Slivers under this many faces join their majority neighbour")
    parser.add_argument("--mirror", default="auto", choices=("auto", "none", "x", "y", "z"),
                        help="Also intersect each sample reflected across this plane, so a "
                             "joint one sample cut on the left is cut on the right too")
    parser.add_argument("--min_unit_faces", type=int, default=None,
                        help="Connected components under this many faces are not voted on alone")
    parser.add_argument("--min_recall", type=float, default=None,
                        help="A mask claims a unit once it covers this share of the unit's pixels")
    parser.add_argument("--view_azimuths", default=DEFAULT_VIEW_AZIMUTHS)
    parser.add_argument("--view_elevations", default=DEFAULT_VIEW_ELEVATIONS)
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--unassigned_to", default=None,
                        help="Part that absorbs units no concept claimed")
    parser.add_argument("--merge", default="name", choices=("name", "unit", "off"),
                        help="name = one node per prompt. unit = one node per voted unit, "
                             "nothing merged and nothing dropped, for inspecting the vote. "
                             "off = stop after the split, no prompts and no SAM3 at all")
    parser.add_argument("--no_reuse", action="store_true",
                        help="Re-run every stage instead of reusing what is in --work_dir")
    parser.add_argument("--allow_partial", action="store_true",
                        help="Accept requested names that ended up with no faces")
    parser.add_argument("--no_texture", action="store_true",
                        help="Skip the Blender bake; parts get a flat placeholder colour")
    parser.add_argument("--texture_size", type=int, default=2048)
    parser.add_argument("--py_sam3", default=None, help=f"default: {DEFAULT_PY_SAM3}")
    parser.add_argument("--sam3_model", default=DEFAULT_SAM3)
    parser.add_argument("--sam3_threshold", type=float, default=0.3)
    args = parser.parse_args()

    segment_parts(
        args.glb, args.prompts, args.out,
        work_dir=args.work_dir,
        samples=args.samples,
        azimuth=args.azimuth,
        azimuth_jitter=args.azimuth_jitter,
        ckpt=args.ckpt,
        transforms=args.transforms,
        color_tol=args.color_tol,
        min_atom_faces=args.min_atom_faces,
        mirror=args.mirror,
        min_unit_faces=args.min_unit_faces,
        min_recall=args.min_recall,
        view_azimuths=args.view_azimuths,
        view_elevations=args.view_elevations,
        radius=args.radius,
        resolution=args.resolution,
        py_sam3=args.py_sam3,
        sam3_model=args.sam3_model,
        sam3_threshold=args.sam3_threshold,
        unassigned_to=args.unassigned_to,
        merge=args.merge,
        reuse=not args.no_reuse,
        strict_parts=not args.allow_partial,
        with_texture=not args.no_texture,
        texture_size=args.texture_size,
    )


if __name__ == "__main__":
    main()
