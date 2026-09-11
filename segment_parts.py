"""A 3D model (+ text prompts) -> one glb with geometrically clean parts.

This replaces segment_api.segment (2D map steers the generative model) and
segment_vote.segment_vote (one full_seg sample, coverage voting). Both decided part
boundaries from evidence that can be wrong per triangle; here geometry decides every
boundary and language only chooses names:

    1. paint. A model the renders show as grey gets a temporary flat colour, because SAM3
       finds nothing on an untextured one (flat_paint.py). Textured models skip this.
    2. guidance. Multi-view SAM3 masks for the prompts, plus the overlays a human reviews.
       These come before the expensive samples on purpose: a bad prompt set is visible
       here, and the split has not been paid for yet.
    3. split. N prompt-free full_seg samples, the conditioning camera jittered around the
       front, their partitions intersected ("meet") so every cut any sample drew is kept
       -> deliberately over-segmented atoms that no later step has to cut again.
    4. units. Atoms are cut into connected components and each remesh inner wall is fused
       into the shell it lines, which must happen before anything is named.
    5. merge, gated by `--merge`: the step-2 masks vote on each unit, then same-named
       units are fused and exported.
    6. complete, gated by `--complete`: our parts are open where they were cut, so X-Part
       regenerates each as a closed solid from the whole model plus a box prompt
       (xpart_complete.py). `--complete full` then bakes the source albedo onto those
       solids. Look at "boxes" first -- a box is a lossy prompt, and a part whose box
       overlaps its neighbours' comes back filled out to that box.

Steps 3 and 6 cost GPU minutes; the rest costs seconds once the renders are cached.
`--merge off` stops after step 4 and writes one node per unit -- still producing the
guidance overlays if prompts were given -- so the two halves can be looked at, and argued
about, separately:

    python segment_parts.py --glb robot.glb --merge off --out split/units.glb
    python merge_parts.py --glb robot.glb --split split/work --prompts head torso arm \
        --out named/parts.glb

Why over-segment first: full_seg has no granularity control and a single sample fuses
neighbouring parts often enough to matter (on the robot test model, shoulder armour and
both arms came out as one 24k-face atom in every same-view sample). Over-segmentation is
free for the merge, which can only fuse atoms, while under-segmentation is unrecoverable.

Run with the SegviGen venv; SAM3 is dispatched to --py_sam3 and X-Part to --py_xpart,
each in its own environment, as in segment_api.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from merge_parts import canonical_prompts, export_labelled, guidance, merge_parts
from pipeline import (  # noqa: F401 — GRANULARITY / DEFAULT_* are the public contract
    DEFAULT_AZIMUTH, DEFAULT_AZIMUTH_JITTER, DEFAULT_COMPLETE, DEFAULT_CONCEPT_BANK,
    DEFAULT_CONDITION, DEFAULT_FLAT_PAINT, DEFAULT_GRANULARITY, DEFAULT_MERGE,
    DEFAULT_MIN_AREA_SHARE, DEFAULT_MIRROR, DEFAULT_OCTREE_RESOLUTION, DEFAULT_RADIUS,
    DEFAULT_REDRAWS, DEFAULT_RESOLUTION, DEFAULT_SAM3_THRESHOLD, DEFAULT_SAMPLES,
    DEFAULT_SEED, DEFAULT_TEXTURE_SIZE, DEFAULT_VIEW_AZIMUTHS, DEFAULT_VIEW_ELEVATIONS,
    DEFAULT_XPART_ROOT, DEFAULT_XPART_WEIGHTS, GRANULARITY,
    PipelineOptions, add_cli_arguments, check_cli, floors,
)
from prompt_specs import normalize_part_specs
from segment_api import DEFAULT_PY_SAM3, DEFAULT_SAM3, DEFAULT_TRANSFORMS, _run

DEFAULT_CKPT = os.path.join(ROOT, "ckpt", "full_seg.ckpt")


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


def sample_is_current(stamp_path, azimuth):
    """True if the sample in this slot was conditioned on the angle we are asking for.

    Samples written before the stamp existed are trusted: they were produced by the
    default jitter, and rerunning every old work directory is a worse failure than
    accepting one that is almost certainly right.
    """
    if not os.path.isfile(stamp_path):
        return True
    with open(stamp_path, "r", encoding="utf-8") as handle:
        return abs(float(json.load(handle)["azimuth"]) - float(azimuth)) < 1e-6


def segment_parts(
    glb,
    prompts,
    out_glb,
    work_dir=None,
    samples=DEFAULT_SAMPLES,
    azimuth=DEFAULT_AZIMUTH,
    azimuth_jitter=DEFAULT_AZIMUTH_JITTER,
    ckpt=None,
    transforms=None,
    color_tol=None,
    granularity=DEFAULT_GRANULARITY,
    min_atom_faces=None,
    mirror=DEFAULT_MIRROR,
    min_unit_faces=None,
    min_recall=None,
    view_azimuths=DEFAULT_VIEW_AZIMUTHS,
    view_elevations=DEFAULT_VIEW_ELEVATIONS,
    radius=DEFAULT_RADIUS,
    resolution=DEFAULT_RESOLUTION,
    py_sam3=None,
    sam3_model=DEFAULT_SAM3,
    sam3_threshold=DEFAULT_SAM3_THRESHOLD,
    concept_bank=DEFAULT_CONCEPT_BANK,
    flat_paint=DEFAULT_FLAT_PAINT,
    unassigned_to=None,
    merge=DEFAULT_MERGE,
    complete=DEFAULT_COMPLETE,
    py_xpart=None,
    xpart_root=DEFAULT_XPART_ROOT,
    xpart_weights=DEFAULT_XPART_WEIGHTS,
    octree_resolution=DEFAULT_OCTREE_RESOLUTION,
    seed=DEFAULT_SEED,
    condition=DEFAULT_CONDITION,
    min_area_share=DEFAULT_MIN_AREA_SHARE,
    redraws=DEFAULT_REDRAWS,
    reuse=True,
    strict_parts=True,
    with_texture=True,
    texture_size=DEFAULT_TEXTURE_SIZE,
):
    """Segment `glb` into parts and write them all into `out_glb`.

    Args:
        prompts: one entry per output part; join concepts with '+' to merge them into one
            part, optionally under a name ("body=head+face+hand"). Only `merge="off"`
            accepts no prompts at all.
        samples: how many full_seg samples to intersect. 1 reproduces the old single-sample
            behaviour. What more samples buy is not a finer split but a less lucky one.
            One draw of 5 left Mickey with 13 atoms, the largest covering 48% of the
            surface and the ears never cut away from the head, so no prompt could name
            them; a different draw of 5 gave 34 atoms and all six parts. That spread
            between draws is wider than the gap between 5 and 9, and it is what the extra
            samples close: 7 and 9 both land on the same six parts. 9 is not worth the two
            extra passes -- it adds atoms (40 against 34) without adding a part.
            Reading each sample more finely does not substitute -- dropping color_tol from
            20 to 3 multiplied the labels per sample twentyfold and produced the same 13
            atoms, because the extra labels are speckle. Only another sample can draw a
            cut that no sample drew.
        azimuth / azimuth_jitter: the conditioning camera, and how far the extra samples
            orbit either side of it. See sample_azimuths. Widening the jitter is not a
            substitute for more samples and can cost parts: at 60 degrees several of
            Mickey's samples came back with 2-6 labels, and those near-blank partitions
            fragment the meet along boundaries that are not real. 9 samples at 60 gave
            more atoms than at 30 and two fewer named parts.
        granularity: how fine the split may get, as a name for both size floors at once
            (see GRANULARITY). Passing `min_atom_faces` or `min_unit_faces` overrides the
            corresponding half. Over-segmenting is close to free -- the merge only ever
            fuses -- so this is less about the part count than about whether the units
            stay big enough for a camera to vote on.
        mirror: "auto" also intersects each sample reflected across the model's symmetry
            plane, if it has one -- full_seg often cuts a joint on one side only.
        merge: "off" stops after the units and writes one node per unit; "name" and "unit"
            hand over to merge_parts.py (see its `merge`). Guidance overlays are written
            either way, as long as prompts were given.
        flat_paint: "auto" gives a model the renders show as grey a temporary flat colour
            before prompting; "off" always prompts on the render as it is.
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
    min_atom_faces, min_unit_faces = floors(granularity, min_atom_faces, min_unit_faces)

    work_dir = os.path.abspath(work_dir or os.path.join(out_dir, "work_parts"))
    atoms_npy = os.path.join(work_dir, "atoms.npy")
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    azimuths = sample_azimuths(samples, azimuth, azimuth_jitter)

    def full_seg(index):
        """One flow-model sample. Each costs a full run, so a rerun picks up where it stopped."""
        sample_dir = os.path.join(work_dir, f"sample_{index:02d}")
        os.makedirs(sample_dir, exist_ok=True)
        seg_glb = os.path.join(sample_dir, "seg.glb")
        # Keyed by the conditioning angle, not just the slot: --azimuth_jitter changes
        # which view each slot holds, and silently reusing the old one would compare two
        # settings that never actually differed.
        stamp = os.path.join(sample_dir, "azimuth.json")
        if reuse and os.path.exists(seg_glb) and sample_is_current(stamp, azimuths[index]):
            print(f"[split] reusing full_seg sample {index + 1}/{samples} ({seg_glb})")
            return seg_glb
        print(f"[split] full_seg sample {index + 1}/{samples}, azimuth {azimuths[index]:g} ...")
        _run([
            py_self, os.path.join(ROOT, "inference_full.py"),
            "--ckpt_path", ckpt, "--glb", glb,
            "--input_vxz", os.path.join(sample_dir, "input.vxz"),
            "--img", os.path.join(sample_dir, "render.png"),
            "--export_glb", seg_glb,
            "--transforms", transforms, "--azimuth", azimuths[index],
        ])
        with open(stamp, "w", encoding="utf-8") as handle:
            json.dump({"azimuth": azimuths[index]}, handle)
        return seg_glb

    # Sample 0 first, on its own: it is the split's reference mesh and also the source of
    # the temporary flat colour, so the guidance overlays can be drawn -- and looked at --
    # before paying for the remaining samples.
    sample_glbs = [full_seg(0)]
    if prompts:
        guidance(
            glb, work_dir, sample_glbs[0], canonical_prompts(normalize_part_specs(prompts)),
            unassigned_to, view_azimuths, view_elevations, radius, resolution,
            py_sam3, sam3_model, sam3_threshold, concept_bank, flat_paint, reuse)
    sample_glbs += [full_seg(index) for index in range(1, samples)]

    print(f"[split] intersecting {samples} partitions into atoms ...")
    reference, atoms, report = meet_samples(sample_glbs, color_tol, min_atom_faces, mirror)
    np.save(atoms_npy, atoms)
    atoms_debug_glb(reference, atoms, os.path.join(work_dir, "atoms.glb"))
    with open(os.path.join(work_dir, "atoms_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    mirrored = (f"mirrored about {report['mirror_axis']}" if report["mirror_axis"]
                else f"not mirrored (symmetry {report['mirror_share']:.2f})")
    print(f"  labels per sample {report['labels_per_sample']}, {mirrored} -> "
          f"{report['atoms']} atoms, largest {report['largest_share']:.1%} of the area")

    # split_units also fuses each remesh inner wall into the outer shell it lines. That
    # has to happen before anything is named: the two are not edge-connected, so a
    # separately-voted inner wall inherits whichever part happens to sit nearest it.
    print("[units] cutting atoms into connected units and fusing the double shells ...")
    units = split_units(reference, atoms, welded_face_adjacency(reference), min_unit_faces)
    units_npy = os.path.join(work_dir, "units.npy")
    np.save(units_npy, units)

    if merge != "off":
        return merge_parts(
            glb, prompts, work_dir, out_glb,
            mesh=sample_glbs[0], atoms=atoms_npy,
            unassigned_to=unassigned_to, merge=merge,
            min_unit_faces=min_unit_faces, min_recall=min_recall,
            view_azimuths=view_azimuths, view_elevations=view_elevations,
            radius=radius, resolution=resolution,
            py_sam3=py_sam3, sam3_model=sam3_model, sam3_threshold=sam3_threshold,
            concept_bank=concept_bank, flat_paint=flat_paint, units=units,
            complete=complete, py_xpart=py_xpart, xpart_root=xpart_root,
            xpart_weights=xpart_weights, octree_resolution=octree_resolution, seed=seed,
            condition=condition, min_area_share=min_area_share, redraws=redraws,
            reuse=reuse, strict_parts=strict_parts,
            with_texture=with_texture, texture_size=texture_size,
        )

    labels_npy = os.path.join(work_dir, "labels.npy")
    names_json = os.path.join(work_dir, "label_names.json")
    np.save(labels_npy, units)
    with open(names_json, "w", encoding="utf-8") as handle:
        json.dump([f"unit_{unit:02d}" for unit in range(units.max() + 1)], handle, indent=2)
    manifest = export_labelled(sample_glbs[0], glb, labels_npy, names_json, out_glb,
                               with_texture, texture_size)
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
    parser.add_argument("--ckpt", default=None, help=f"default: {DEFAULT_CKPT}")
    parser.add_argument("--transforms", default=None, help=f"default: {DEFAULT_TRANSFORMS}")
    parser.add_argument("--py_sam3", default=None, help=f"default: {DEFAULT_PY_SAM3}")
    parser.add_argument("--sam3_model", default=DEFAULT_SAM3)
    add_cli_arguments(parser, split=True, merge_off=True)
    args = parser.parse_args()
    check_cli(parser, args)
    options = PipelineOptions.from_namespace(args)
    segment_parts(
        args.glb, args.prompts, args.out,
        work_dir=args.work_dir,
        ckpt=args.ckpt,
        transforms=args.transforms,
        py_sam3=args.py_sam3,
        sam3_model=args.sam3_model,
        **options.segment_kwargs(),
    )


if __name__ == "__main__":
    main()
