"""High-level interface: a 3D model + semantic text prompts -> one glb with named sub-parts.

Wraps the existing render / SAM3 / SegviGen / rebake steps (see run_segvigen_sam3.bat for
the equivalent shell pipeline) behind a single function/CLI:

    input:  --glb model.glb --prompts head arm leg ...
    output: --out parts.glb   (single file, one mesh node per part)

A prompt entry can merge several concepts into one part ("body=head+face+hand"), and
--unassigned_to folds whatever no prompt claimed into a named part, so the result has
exactly as many parts as were asked for.

Pass --no_sam to drop the text prompts and run the prompt-free full_seg checkpoint on a
plain render instead; parts then come out unnamed and colour-clustered, but no SAM3 (and
no second venv) is involved.

Run this with the SegviGen venv (.venv) -- it imports torch/bpy itself for the render and
part-split steps. SAM3 needs a different environment (transformers 5.x vs SegviGen's
4.57.6, see sam3_to_2dmap.py), so that one step is dispatched to --py_sam3 as a
subprocess; everything else runs in-process.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence

from prompt_specs import (
    normalize_part_specs,
    part_names,
    validate_named_rows,
    validate_part_coverage,
    validate_target_name,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEFAULT_PY_SAM3 = os.environ.get("SEGVIGEN_PY_SAM3", r"E:\AI_New\ModelGen\.venv_holo\Scripts\python.exe")
DEFAULT_CKPT = os.path.join(ROOT, "ckpt", "full_seg_w_2d_map.ckpt")
DEFAULT_CKPT_NO_SAM = os.path.join(ROOT, "ckpt", "full_seg.ckpt")
DEFAULT_TRANSFORMS = os.path.join(ROOT, "data_toolkit", "transforms.json")


def _run(cmd):
    print("+", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True)


def _build_sam3_audit_result(render_path, map_path, legend):
    return {
        "render": os.path.abspath(render_path),
        "map": os.path.abspath(map_path),
        "legend": legend,
    }


def _select_front_view(glb, transforms, mode, canonical_prompts, expected_names,
                       py_self, py_sam3, sam3_model, sam3_threshold, work_dir,
                       azimuths=None):
    """Render an orbit of candidate views and pick the front/best-conditioned one.

    Candidates are rendered with the conditioning camera itself (render_cond_view),
    so the winning azimuth can be fed straight back into the pipeline -- no camera
    convention conversion. Returns the chosen azimuth in degrees.
    """
    from data_toolkit.front_view import (
        DEFAULT_CANDIDATE_AZIMUTHS,
        rank_views,
        sam3_legend_score,
        vlm_pick,
    )

    azimuths = list(azimuths) if azimuths else list(DEFAULT_CANDIDATE_AZIMUTHS)
    cand_dir = os.path.join(work_dir, "front_candidates")
    os.makedirs(cand_dir, exist_ok=True)
    template = os.path.join(cand_dir, "candidate.png")
    print(f"[0/4] front-view selection ({mode}): rendering {len(azimuths)} candidates ...")
    _run([
        py_self, os.path.join(ROOT, "data_toolkit", "render_cond_view.py"),
        "--glb", glb, "--transforms", transforms, "--out", template,
        "--azimuths", ",".join(f"{a:g}" for a in azimuths),
    ])
    if len(azimuths) == 1:
        paths = [template]
    else:
        paths = [os.path.join(cand_dir, f"candidate_{a:g}.png") for a in azimuths]

    ranked = rank_views(paths)
    for path, m in ranked:
        print(f"  {os.path.basename(path)}: score={m['score']:.3f} "
              f"sym={m['symmetry']:.3f} fg={m['fg_ratio']:.3f} ctr={m['centeredness']:.3f}")
    record = {"mode": mode, "azimuths": azimuths,
              "ranking": [{"image": os.path.basename(p), **m} for p, m in ranked]}

    if mode == "vlm":
        top = [path for path, _ in ranked[:4]]
        try:
            index = vlm_pick(top, object_hint=", ".join(expected_names) or None)
            best = top[index]
            record["vlm_index"] = index
        except Exception as exc:  # network/quota/parse failure: fall back to metrics
            print(f"  VLM pick failed ({exc}); falling back to metric top view")
            best = ranked[0][0]
    elif mode == "auto":
        # Metric top-K, then SAM3 itself decides which view segments best.
        best, best_score = ranked[0][0], -1.0
        for path, _ in ranked[:3]:
            map_out = path.replace(".png", "_map.png")
            legend_path = path.replace(".png", "_legend.json")
            _run([
                py_sam3, os.path.join(ROOT, "sam3_to_2dmap.py"),
                "--image", path, "--out", map_out, "--legend", legend_path,
                "--model", sam3_model, "--threshold", sam3_threshold,
                "--allow_missing",
                "--prompts", *canonical_prompts,
            ])
            with open(legend_path, "r", encoding="utf-8") as file:
                legend = json.load(file)
            score = sam3_legend_score(legend, expected_names)
            print(f"  {os.path.basename(path)}: sam3 score {score:.3f}")
            if score > best_score:
                best, best_score = path, score
        record["sam3_best_score"] = best_score
        if best_score <= 0:
            print("  SAM3 detected nothing on all candidates; using metric top view")
            best = ranked[0][0]
    else:  # metric
        best = ranked[0][0]

    azimuth = float(azimuths[paths.index(best)])
    record["chosen"] = {"image": os.path.basename(best), "azimuth": azimuth}
    with open(os.path.join(cand_dir, "front_view.json"), "w", encoding="utf-8") as file:
        json.dump(record, file, indent=2)
    print(f"  front view: azimuth {azimuth:g} ({os.path.basename(best)})")
    return azimuth



def segment(
    glb,
    prompts: str | Sequence[str],
    out_glb,
    with_texture=True,
    work_dir=None,
    ckpt=None,
    transforms=None,
    py_sam3=None,
    sam3_model="facebook/sam3",
    sam3_threshold=0.3,
    texture_size=2048,
    azimuth=0.0,
    front_view=None,
    front_view_azimuths=None,
    use_sam3=True,
    unassigned_to=None,
    strict_parts=True,
    sam3_only=False,
):
    """Segment `glb` into named parts and write them all into a single `out_glb`.

    Args:
        glb: path to the input 3D model (any GLB SegviGen/o_voxel can voxelize).
        prompts: one string or a sequence with one entry per dynamically named output part.
            An entry may merge several concepts into one part by joining them with '+',
            optionally under an explicit name, e.g. ["roof", "opening=door+window"].
            Ignored when use_sam3 is False.
        out_glb: output path; one glb, one mesh node per part.
        with_texture: True (default) re-UVs each part in Blender and bakes the source
            model's real albedo back onto it (slower, needs bpy). False skips Blender
            entirely and gives each part a flat placeholder colour instead (fast).
        work_dir: keep intermediates (render, 2D map, segmentation glb) here instead of a
            temp dir that gets deleted afterwards.
        azimuth: degrees to orbit transforms.json's camera before rendering the
            conditioning view. The calibrated camera is not necessarily in front of the
            model (for data_toolkit/transforms.json and monk.glb, 0 looks at its back and
            ~135 is head-on front), and everything downstream -- SAM3's prompts and the
            part colours SegviGen infers -- only sees the side that got rendered.
            Ignored when front_view is set.
        front_view: None keeps the fixed `azimuth` behaviour. "metric" picks the candidate
            orbit view with the best silhouette score (symmetry + coverage + centeredness);
            "auto" refines the metric top-3 with SAM3 prompt confidence (needs use_sam3);
            "vlm" asks a vision-language model which of the metric top-4 is the canonical
            front (needs MOONSHOT_API_KEY; falls back to metric on failure).
        front_view_azimuths: candidate orbit degrees for front_view; defaults to the full
            45-degree turntable in data_toolkit/front_view.py.
        unassigned_to: name of the part that absorbs foreground pixels no prompt claimed, so
            the output has exactly as many parts as `prompts` asked for. Left alone, those
            pixels become an extra "<unassigned>" part.
        use_sam3: True (default) drives the segmentation with SAM3 text prompts through
            the 2D-map checkpoint. False skips SAM3 (and its separate venv) entirely and
            runs the prompt-free full_seg checkpoint on a plain render instead, so parts
            come out unnamed and colour-clustered.
        strict_parts: require SAM3 legend and final manifest names to exactly match the
            dynamically requested output names. Disable only to inspect partial results.
        sam3_only: stop after rendering and strict SAM3 legend validation, returning the
            persistent render/map paths and legend without running SegviGen or Blender.

    Returns: in sam3_only mode, a dict containing absolute render/map paths and legend.
        Otherwise, a parts.json-style manifest (list of dicts with label/name/node/
        part_color/faces/area for each part).
    """
    if sam3_only and not use_sam3:
        raise ValueError("sam3_only requires use_sam3=True")
    if sam3_only and work_dir is None:
        raise ValueError("sam3_only requires work_dir so audit artifacts are preserved")
    if sam3_only and not strict_parts:
        raise ValueError("sam3_only requires strict_parts=True")

    specs = normalize_part_specs(prompts) if use_sam3 else []
    expected_names = part_names(specs)
    validate_target_name(unassigned_to, expected_names)
    canonical_prompts = [
        name if concepts == [name] else f"{name}={'+'.join(concepts)}"
        for name, concepts in specs
    ]

    glb = os.path.abspath(glb)
    out_glb = os.path.abspath(out_glb)
    ckpt = os.path.abspath(ckpt or (DEFAULT_CKPT if use_sam3 else DEFAULT_CKPT_NO_SAM))
    transforms = os.path.abspath(transforms or DEFAULT_TRANSFORMS)
    py_sam3 = py_sam3 or DEFAULT_PY_SAM3
    py_self = sys.executable

    own_tmp = work_dir is None
    work_dir = os.path.abspath(work_dir) if work_dir else tempfile.mkdtemp(prefix="segapi_")
    os.makedirs(work_dir, exist_ok=True)
    render_path = os.path.join(work_dir, "render.png")
    map_path = os.path.join(work_dir, "sam3_2d_map.png")
    legend_path = os.path.join(work_dir, "sam3_2d_map_legend.json")
    seg_glb = os.path.join(work_dir, "seg.glb")
    input_vxz = os.path.join(work_dir, "input.vxz")

    if front_view is not None:
        from data_toolkit.front_view import FRONT_VIEW_MODES

        if front_view not in FRONT_VIEW_MODES:
            raise ValueError(f"front_view must be one of {FRONT_VIEW_MODES}")
        if front_view == "auto" and not use_sam3:
            raise ValueError("front_view='auto' refines with SAM3 and needs use_sam3=True")
        candidate_azimuths = (
            [float(a) for a in front_view_azimuths.split(",") if a.strip()]
            if isinstance(front_view_azimuths, str) else front_view_azimuths
        )
        azimuth = _select_front_view(
            glb, transforms, front_view, canonical_prompts, expected_names,
            py_self, py_sam3, sam3_model, sam3_threshold, work_dir,
            azimuths=candidate_azimuths,
        )

    succeeded = False
    try:
        print(f"[1/4] rendering conditioning view (azimuth {azimuth:g}) ...")
        _run([
            py_self, os.path.join(ROOT, "data_toolkit", "render_cond_view.py"),
            "--glb", glb, "--transforms", transforms, "--out", render_path,
            "--azimuths", azimuth,
        ])

        if use_sam3:
            print(f"[2/4] SAM3 prompts {canonical_prompts} -> 2D part map ...")
            sam3_cmd = [
                py_sam3, os.path.join(ROOT, "sam3_to_2dmap.py"),
                "--image", render_path, "--out", map_path, "--legend", legend_path,
                "--model", sam3_model, "--threshold", sam3_threshold,
            ]
            if unassigned_to:
                sam3_cmd += ["--unassigned_to", unassigned_to]
            # --prompts is nargs="+", so it has to come last or it eats the flags above.
            _run(sam3_cmd + ["--prompts", *canonical_prompts])
            if strict_parts:
                with open(legend_path, "r", encoding="utf-8") as file:
                    legend = json.load(file)
                validate_part_coverage(expected_names, legend)
            if sam3_only:
                result = _build_sam3_audit_result(render_path, map_path, legend)
                print(f"SAM3 audit render {result['render']}")
                print(f"SAM3 audit map {result['map']}")
                print(f"SAM3 audit legend {legend_path}")
                succeeded = True
                return result
        else:
            print("[2/4] SAM3 skipped: conditioning on the plain render ...")

        print("[3/4] SegviGen part segmentation ...")
        # --two_d_map means "the image I hand you is already the conditioning view, do not
        # render one yourself", which is what we want either way: the render above already
        # applied the azimuth, whereas inference_full's own render would default to 0.
        _run([
            py_self, os.path.join(ROOT, "inference_full.py"),
            "--ckpt_path", ckpt, "--glb", glb, "--input_vxz", input_vxz,
            "--img", map_path if use_sam3 else render_path,
            "--export_glb", seg_glb, "--two_d_map",
        ])

        print(f"[4/4] splitting into parts (texture={'on' if with_texture else 'off'}) ...")
        out_dir = os.path.dirname(out_glb) or "."
        os.makedirs(out_dir, exist_ok=True)
        combined_name = os.path.basename(out_glb)
        # Dispatched as a subprocess (not imported in-process) because bpy's Cycles
        # bake can access-violate on interpreter teardown; isolating it in its own
        # process keeps that harmless crash from being mistaken for this run failing.
        rebake_cmd = [
            py_self, os.path.join(ROOT, "data_toolkit", "parts_rebake.py"),
            "--seg_glb", seg_glb, "--out_dir", out_dir,
            "--combined_name", combined_name,
        ]
        if use_sam3:
            # Without SAM3 there are no prompt names/colours to match against, so
            # parts_rebake clusters the predicted colours on its own.
            rebake_cmd += [
                "--palette", legend_path,
                "--two_d_map", map_path,
                "--transforms", transforms,
                "--azimuth", azimuth,
            ]
        if with_texture:
            rebake_cmd += ["--source_glb", glb, "--texture_size", texture_size]
        else:
            rebake_cmd += ["--no_bake"]
        _run(rebake_cmd)

        with open(os.path.join(out_dir, "parts.json"), "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if strict_parts and use_sam3:
            validate_named_rows(expected_names, manifest, key="name")
        succeeded = True
    finally:
        if own_tmp and succeeded:
            shutil.rmtree(work_dir, ignore_errors=True)
        elif own_tmp:
            print(f"pipeline failed; kept intermediates at {work_dir}")

    print(f"saved {out_glb} ({len(manifest)} parts)")
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="model + semantic text prompts -> single glb with named sub-parts"
    )
    parser.add_argument("--glb", required=True, help="Input 3D model")
    parser.add_argument("--prompts", nargs="+", default=None,
                        help="One entry per dynamically named output part. Join concepts with "
                             "'+' to merge them into one part, optionally naming it, e.g. "
                             "'roof opening=door+window'. Required unless --no_sam.")
    parser.add_argument("--unassigned_to", default=None,
                        help="Name of the part that absorbs foreground pixels no prompt claimed, "
                             "so the output has exactly as many parts as --prompts asked for.")
    parser.add_argument("--no_sam", action="store_true",
                        help="Skip SAM3 and its separate venv: segment with the prompt-free "
                             f"checkpoint ({os.path.basename(DEFAULT_CKPT_NO_SAM)}) on a plain "
                             "render. Parts come out unnamed and colour-clustered.")
    parser.add_argument(
        "--sam3_only",
        action="store_true",
        help="Stop after rendering and SAM3; output render/map/legend for review.",
    )
    parser.add_argument("--azimuth", type=float, default=0.0,
                        help="Degrees to orbit transforms.json's camera for the conditioning "
                             "view. That camera is not necessarily in front of the model: for "
                             "data_toolkit/transforms.json and monk.glb, 0 looks at its back "
                             "and ~135 is head-on front. Render a probe with "
                             "data_toolkit/render_cond_view.py --azimuths for a new model. "
                             "Ignored when --front_view is set.")
    parser.add_argument("--front_view", default=None, choices=["metric", "auto", "vlm"],
                        help="Pick the conditioning view automatically instead of using "
                             "--azimuth: 'metric' scores an orbit of renders by silhouette "
                             "symmetry/coverage/centeredness; 'auto' refines the metric top-3 "
                             "with SAM3 prompt confidence; 'vlm' asks a vision-language model "
                             "(MOONSHOT_API_KEY) which of the metric top-4 is the front.")
    parser.add_argument("--front_view_azimuths", default=None,
                        help="Comma-separated candidate orbit degrees for --front_view "
                             "(default: full 45-degree turntable).")
    parser.add_argument("--out", default=None,
                        help="Output glb path (single file, one node per part). "
                             "Not required with --sam3_only.")
    parser.add_argument(
        "--no_texture", action="store_true",
        help="Skip the Blender re-UV+bake step; parts get a flat placeholder colour "
             "instead of the source model's real texture (faster, no Blender needed). "
             "Default is to include the real texture.",
    )
    parser.add_argument("--texture_size", type=int, default=2048)
    parser.add_argument("--ckpt", default=None, help=f"default: {DEFAULT_CKPT}")
    parser.add_argument("--transforms", default=None, help=f"default: {DEFAULT_TRANSFORMS}")
    parser.add_argument("--py_sam3", default=None, help=f"default: {DEFAULT_PY_SAM3}")
    parser.add_argument("--sam3_model", default="facebook/sam3")
    parser.add_argument("--sam3_threshold", type=float, default=0.3)
    parser.add_argument(
        "--allow_partial",
        action="store_true",
        help="Return partial/extra components instead of enforcing an exact prompt-name match.",
    )
    parser.add_argument("--work_dir", default=None,
                        help="Keep intermediates (render/2D map/segmentation glb) here "
                             "instead of an auto-deleted temp dir")
    args = parser.parse_args()
    if not args.no_sam and not args.prompts:
        parser.error("--prompts is required unless --no_sam is set")
    if args.sam3_only and args.no_sam:
        parser.error("--sam3_only cannot be combined with --no_sam")
    if args.sam3_only and not args.work_dir:
        parser.error("--sam3_only requires --work_dir so audit artifacts are preserved")
    if args.sam3_only and args.allow_partial:
        parser.error("--sam3_only requires strict legend validation; remove --allow_partial")
    if not args.sam3_only and not args.out:
        parser.error("--out is required unless --sam3_only is set")

    segment(
        args.glb, args.prompts, args.out or os.path.join(args.work_dir, "unused.glb"),
        with_texture=not args.no_texture,
        work_dir=args.work_dir,
        ckpt=args.ckpt,
        transforms=args.transforms,
        py_sam3=args.py_sam3,
        sam3_model=args.sam3_model,
        sam3_threshold=args.sam3_threshold,
        texture_size=args.texture_size,
        azimuth=args.azimuth,
        front_view=args.front_view,
        front_view_azimuths=args.front_view_azimuths,
        use_sam3=not args.no_sam,
        unassigned_to=args.unassigned_to,
        strict_parts=not args.allow_partial,
        sam3_only=args.sam3_only,
    )


if __name__ == "__main__":
    main()
