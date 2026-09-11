"""Name and merge an existing over-segmentation: split artifacts + prompts -> named parts.

The second half of segment_parts.py, callable on its own so the two halves can be tested
apart. Splitting is expensive and prompt-independent; naming is cheap and is the part you
re-run while deciding what the parts should be called. Point this at a split directory
(`segment_parts.py --merge off`) and re-run it with different prompts:

    python segment_parts.py --glb robot.glb --merge off --out split/units.glb
    python merge_parts.py --glb robot.glb --split split/work --prompts head torso arm ... \
        --out named/parts.glb

Renders and masks are cached in the split directory, so a second run with the same prompts
costs only the vote (seconds), and a run with new prompts only re-runs SAM3.

Steps: render the source model over a fixed view grid -> SAM3 masks per view -> each unit
(a connected component of an atom) takes the name most views agree on -> export per name,
with the source albedo baked back on.

The vote can only merge atoms. If two parts came out of the split as one atom, no prompt
will separate them here -- look at `atoms.glb` in the split directory first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from prompt_specs import normalize_part_specs, part_names, validate_named_rows, validate_target_name
from segment_api import (
    BANK_THRESHOLD, DEFAULT_CONCEPT_BANK, DEFAULT_PY_SAM3, DEFAULT_SAM3, _run,
)

# The 3/4 pair (45 and 225), barely raised. A level azimuth-90 misses "torso" on the
# chest and the arm mask then swallows the backpack, but height costs more than it buys:
# at 35 degrees the camera looks down far enough that the torso hides the legs, feet and
# base. 10 degrees keeps the 3/4 framing and still shows the bottom half.
DEFAULT_VIEW_AZIMUTHS = "45,225"
DEFAULT_VIEW_ELEVATIONS = "10"
DEFAULT_RADIUS = 2.0
DEFAULT_RESOLUTION = 512
DEFAULT_SAM3_THRESHOLD = BANK_THRESHOLD
MERGE_MODES = ("name", "unit")
# "auto" paints only a model the renders show as grey; see flat_paint.is_colorless.
FLAT_PAINT_MODES = ("auto", "on", "off")
# Our parts are open where they were cut. X-Part regenerates each one as a closed solid
# from the whole model plus a box prompt; "boxes" writes those prompts without loading it.
COMPLETE_MODES = ("off", "boxes", "full")
# A box also contains whatever else passes through it; our split knows the faces exactly.
CONDITION_MODES = ("surface", "box")
DEFAULT_CONDITION = "surface"
DEFAULT_PY_XPART = os.environ.get(
    "SEGVIGEN_PY_XPART", "/root/autodl-tmp/envs/xpart/bin/python")
DEFAULT_XPART_ROOT = os.environ.get(
    "SEGVIGEN_XPART_ROOT", "/root/autodl-tmp/Hunyuan3D-Part/XPart")
DEFAULT_XPART_WEIGHTS = os.environ.get(
    "SEGVIGEN_XPART_WEIGHTS", "/root/autodl-tmp/Hunyuan3D-Part/weights")


def canonical_prompts(specs):
    return [name if concepts == [name] else f"{name}={'+'.join(concepts)}"
            for name, concepts in specs]


def _angles(text):
    return [float(value) for value in str(text).replace(" ", "").split(",") if value]


def views_are_current(views_dir, azimuths, elevations, radius, resolution):
    """True if views_dir already holds exactly the grid asked for."""
    manifest_path = os.path.join(views_dir, "cameras.json")
    if not os.path.isfile(manifest_path):
        return False
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("resolution") != resolution or manifest.get("radius") != radius:
        return False
    have = {(view["azimuth"], view["elevation"]) for view in manifest["views"]}
    want = {(a, e) for e in _angles(elevations) for a in _angles(azimuths)}
    if have != want:
        return False
    return all(os.path.isfile(os.path.join(views_dir, view["image"]))
               for view in manifest["views"])


def masks_name(prompts, unassigned_to, threshold, model, azimuths, elevations,
               concept_bank=None, overlay="v3", flat_paint=False):
    """Mask files are keyed by what produced them, so a new grid or prompt cannot reuse them."""
    key = json.dumps(
        [prompts, unassigned_to, threshold, model, azimuths, elevations,
         concept_bank or "", overlay, bool(flat_paint)],
        sort_keys=True)
    return f"masks_{hashlib.sha1(key.encode()).hexdigest()[:10]}.npz"


GUIDANCE_COLORS = [
    (220, 40, 40), (40, 90, 230), (30, 180, 70), (240, 210, 30),
    (40, 200, 210), (230, 70, 180), (140, 50, 200), (240, 130, 30),
]


def paint_guidance(views_dir, masks_npz, out_dir):
    """Overlay each view's SAM3 masks on the render, smaller concepts on top.

    Written next to the vote so a wrong name can be blamed on the 2D mask (or not)
    without opening the npz.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    from data_toolkit.lift_sam3 import load_masks

    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(out_dir):
        if name.endswith(".png"):
            os.remove(os.path.join(out_dir, name))
    mask_set = load_masks(masks_npz)
    written = []
    for view, name in enumerate(mask_set.views):
        image = np.asarray(Image.open(os.path.join(views_dir, f"{name}.png")).convert("RGB"))
        paint = image.astype(np.float32)
        order = sorted(range(len(mask_set.concepts)),
                       key=lambda i: int(mask_set.masks[view, i].sum()))
        for index in order:
            if mask_set.scores[view, index] <= 0:
                continue
            hit = mask_set.masks[view, index]
            color = np.array(GUIDANCE_COLORS[index % len(GUIDANCE_COLORS)], dtype=np.float32)
            paint[hit] = paint[hit] * 0.35 + color * 0.65
        canvas = Image.fromarray(paint.clip(0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(canvas)
        top = 6
        for index, (concept, owner) in enumerate(zip(mask_set.concepts, mask_set.owners)):
            if mask_set.scores[view, index] <= 0:
                continue
            color = GUIDANCE_COLORS[index % len(GUIDANCE_COLORS)]
            draw.rectangle([6, top, 22, top + 14], fill=color)
            draw.text((28, top),
                      f"{owner}  {mask_set.scores[view, index]:.2f}  "
                      f"{int(mask_set.masks[view, index].sum())}px",
                      fill=(20, 20, 20))
            top += 16
        path = os.path.join(out_dir, f"{name}.png")
        canvas.save(path)
        written.append(path)
    print(f"  guidance overlays -> {out_dir} ({len(written)} views)")
    return written


def render_views(glb, views_dir, azimuths=DEFAULT_VIEW_AZIMUTHS,
                 elevations=DEFAULT_VIEW_ELEVATIONS, radius=DEFAULT_RADIUS,
                 resolution=DEFAULT_RESOLUTION, reuse=True):
    if reuse and views_are_current(views_dir, azimuths, elevations, radius, resolution):
        print(f"[render] reusing the view grid in {views_dir}")
        return views_dir
    print(f"[render] rendering the view grid ({azimuths} x {elevations}) ...")
    os.makedirs(views_dir, exist_ok=True)
    _run([
        sys.executable, os.path.join(ROOT, "data_toolkit", "render_multiview.py"),
        "--glb", glb, "--out_dir", views_dir,
        "--azimuths", azimuths, "--elevations", elevations,
        "--radius", radius, "--resolution", resolution,
    ])
    return views_dir


def sam3_masks(views_dir, prompts, out_npz, unassigned_to=None, py_sam3=None,
               model=DEFAULT_SAM3, threshold=DEFAULT_SAM3_THRESHOLD, reuse=True,
               concept_bank=DEFAULT_CONCEPT_BANK, raw=False):
    if reuse and os.path.isfile(out_npz):
        print(f"[guidance] reusing masks for {prompts} ({os.path.basename(out_npz)})")
        return out_npz
    print(f"[guidance] SAM3 v3 prompts {prompts} over the view grid ...")
    command = [
        py_sam3 or DEFAULT_PY_SAM3, os.path.join(ROOT, "sam3_multiview.py"),
        "--views_dir", views_dir, "--out", out_npz,
        "--model", model, "--threshold", threshold,
        "--concept_bank", concept_bank or "",
    ]
    if raw:
        command.append("--raw")
    if unassigned_to:
        command += ["--unassigned_to", unassigned_to]
    # --prompts is nargs="+" and would otherwise swallow the flags after it.
    _run(command + ["--prompts", *prompts])
    return out_npz


def flat_paint_stage(seg_glb, views_dir, out_dir, mode="auto", reuse=True):
    """Return (the views SAM3 should read, whether they were painted).

    An untextured model gives SAM3 nothing to hold on to -- on the robot's back view it
    reported "no instance" for both `head` and `torso`. Painting one full_seg sample's
    parts onto the renders in flat colour puts them back. See flat_paint.py.
    """
    from data_toolkit.lift_sam3 import load_cameras
    from flat_paint import is_colorless, paint_views

    if mode not in FLAT_PAINT_MODES:
        raise ValueError(f"flat_paint must be one of {FLAT_PAINT_MODES}, got {mode!r}")
    if mode == "off":
        return views_dir, False
    manifest, cameras = load_cameras(views_dir)
    if mode == "auto" and not is_colorless(views_dir, manifest):
        return views_dir, False
    if reuse and views_complete(out_dir, manifest):
        print(f"[paint] reusing the flat-painted views in {out_dir}")
        return out_dir, True
    print("[paint] no usable colour; flat-painting one full_seg sample onto the views ...")
    paint_views(seg_glb, views_dir, out_dir, manifest, cameras)
    return out_dir, True


def views_complete(views_dir, manifest):
    """True if `views_dir` holds an image for every view in an already-rendered grid."""
    return all(os.path.isfile(os.path.join(views_dir, view["image"]))
               for view in manifest["views"])


def guidance(glb, work_dir, seg_glb, prompts, unassigned_to=None,
             view_azimuths=DEFAULT_VIEW_AZIMUTHS, view_elevations=DEFAULT_VIEW_ELEVATIONS,
             radius=DEFAULT_RADIUS, resolution=DEFAULT_RESOLUTION, py_sam3=None,
             sam3_model=DEFAULT_SAM3, sam3_threshold=DEFAULT_SAM3_THRESHOLD,
             concept_bank=DEFAULT_CONCEPT_BANK, flat_paint="auto", reuse=True):
    """Render, flat-paint if needed, run SAM3, and draw the overlays a human reviews.

    Returns (views_dir, masks_npz). Cheap to call twice: everything downstream of the
    render is keyed by what produced it, so `segment_parts` can put this before the
    split -- the point being that the overlays are worth looking at before paying for
    four more full_seg samples -- and `merge_parts` can still call it standalone.
    """
    views_dir = render_views(glb, os.path.join(work_dir, "views"), view_azimuths,
                             view_elevations, radius, resolution, reuse)
    prompt_dir, painted = flat_paint_stage(
        seg_glb, views_dir, os.path.join(work_dir, "views_flat"), flat_paint, reuse)
    bank = os.path.abspath(concept_bank) if concept_bank else ""
    masks_npz = sam3_masks(
        prompt_dir, prompts,
        os.path.join(work_dir, masks_name(
            prompts, unassigned_to, sam3_threshold, sam3_model,
            view_azimuths, view_elevations, bank, "v3", painted)),
        unassigned_to, py_sam3, sam3_model, sam3_threshold, reuse, concept_bank=bank)
    # Overlay on the real render even when SAM3 read the painted one: they are rasterised
    # through the same camera, and a reviewer needs to see the actual model under a mask.
    paint_guidance(views_dir, masks_npz, os.path.join(work_dir, "guidance"))
    return views_dir, masks_npz


def split_artifacts(split_dir, mesh=None, atoms=None):
    """(reference seg.glb, atoms.npy) of a segment_parts.py split directory."""
    mesh = mesh or os.path.join(split_dir, "sample_00", "seg.glb")
    atoms = atoms or os.path.join(split_dir, "atoms.npy")
    for path in (mesh, atoms):
        if not os.path.isfile(path):
            raise SystemExit(f"{path} not found; run segment_parts.py --merge off first")
    return os.path.abspath(mesh), os.path.abspath(atoms)


def merge_parts(
    glb,
    prompts,
    split_dir,
    out_glb,
    mesh=None,
    atoms=None,
    unassigned_to=None,
    merge="name",
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
    flat_paint="auto",
    units=None,
    complete="off",
    py_xpart=None,
    xpart_root=DEFAULT_XPART_ROOT,
    xpart_weights=DEFAULT_XPART_WEIGHTS,
    octree_resolution=512,
    seed=42,
    condition=DEFAULT_CONDITION,
    reuse=True,
    strict_parts=True,
    with_texture=True,
    texture_size=2048,
):
    """Name the atoms in `split_dir` with `prompts` and write the parts into `out_glb`.

    Args:
        split_dir: a segment_parts.py work directory (sample_00/seg.glb + atoms.npy).
        merge: "name" fuses everything the vote gave the same name into one node; "unit"
            keeps one node per unit, named `<index>_<voted name>`, so a wrong name can be
            traced to a unit before it is merged away.
        unassigned_to: the part absorbing units no concept claimed. Without it those faces
            are dropped from the output.
        flat_paint: "auto" gives a model the renders show as grey a temporary flat colour
            before prompting, because SAM3 finds nothing on an untextured one.
        units: the split's shell-fused unit ids. Left None they are recomputed here, which
            is correct but wasteful when the caller just built them.
        complete: hand the parts to X-Part afterwards. "boxes" only writes the prompts and
            a preview (cheap, no GPU); "full" also regenerates each part as a closed solid.
        reuse: keep the renders and, for these exact prompts, the masks already in
            `split_dir`. Turn off to re-render (e.g. after editing the source model).

    Returns a parts.json-style manifest, one row per exported part.
    """
    import numpy as np

    from data_toolkit.lift_sam3 import load_cameras, load_masks
    from data_toolkit.parts_rebake import load_single_mesh
    from data_toolkit.unit_vote import (
        DEFAULT_MIN_RECALL, DEFAULT_MIN_UNIT_FACES, print_report, vote,
    )

    if merge not in MERGE_MODES:
        raise ValueError(f"merge must be one of {MERGE_MODES}, got {merge!r}")
    specs = normalize_part_specs(prompts)
    expected_names = part_names(specs)
    validate_target_name(unassigned_to, expected_names)
    prompt_list = canonical_prompts(specs)

    glb = os.path.abspath(glb)
    split_dir = os.path.abspath(split_dir)
    out_glb = os.path.abspath(out_glb)
    out_dir = os.path.dirname(out_glb) or "."
    os.makedirs(out_dir, exist_ok=True)
    mesh_path, atoms_path = split_artifacts(split_dir, mesh, atoms)
    min_unit_faces = DEFAULT_MIN_UNIT_FACES if min_unit_faces is None else min_unit_faces
    min_recall = DEFAULT_MIN_RECALL if min_recall is None else min_recall

    views_dir, masks_npz = guidance(
        glb, split_dir, mesh_path, prompt_list, unassigned_to,
        view_azimuths, view_elevations, radius, resolution,
        py_sam3, sam3_model, sam3_threshold, concept_bank, flat_paint, reuse)

    print("[merge] naming units by multi-view SAM3 voting ...")
    reference = load_single_mesh(mesh_path)
    atom_labels = np.load(atoms_path)
    if len(atom_labels) != len(reference.faces):
        raise SystemExit(f"{len(atom_labels)} atom labels for {len(reference.faces)} faces")
    mask_set = load_masks(masks_npz)
    manifest_views, cameras = load_cameras(views_dir)
    labels, rows, units = vote(
        reference, atom_labels, mask_set, cameras, float(manifest_views["camera_angle_x"]),
        int(manifest_views["resolution"]), expected_names, unassigned_to,
        min_unit_faces, min_recall, units=units,
    )
    print_report(rows, list(dict.fromkeys(mask_set.owners)))
    if merge == "unit":
        # nothing is dropped here, not even units no concept claimed: this output exists
        # to be looked at, and a missing piece is the hardest kind to notice.
        label_names = [f"{row['unit']:02d}_{row['name'] or 'unnamed'}" for row in rows]
        labels = units
    else:
        label_names = expected_names

    labels_npy = os.path.join(split_dir, "labels.npy")
    names_json = os.path.join(split_dir, "label_names.json")
    np.save(labels_npy, labels)
    with open(names_json, "w", encoding="utf-8") as handle:
        json.dump(label_names, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(split_dir, "vote_report.json"), "w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)

    manifest = export_labelled(mesh_path, glb, labels_npy, names_json, out_glb,
                               with_texture, texture_size)
    if strict_parts and merge == "name":
        validate_named_rows(expected_names, manifest, key="name")
    print(f"saved {out_glb} ({len(manifest)} parts)")
    complete_parts(glb, out_glb, os.path.join(out_dir, "complete"), complete,
                   py_xpart, xpart_root, xpart_weights, octree_resolution, seed,
                   condition)
    return manifest


def export_labelled(mesh_path, source_glb, labels_npy, names_json, out_glb,
                    with_texture=True, texture_size=2048, step="[export]"):
    """Cut the reference mesh by a face label array and write one node per label."""
    print(f"{step} exporting parts (texture={'on' if with_texture else 'off'}) ...")
    out_dir = os.path.dirname(out_glb) or "."
    # parts_rebake keeps its own bpy process: Cycles can access-violate on teardown, and
    # that harmless crash should not look like this pipeline failing.
    command = [
        sys.executable, os.path.join(ROOT, "data_toolkit", "parts_rebake.py"),
        "--seg_glb", mesh_path, "--out_dir", out_dir,
        "--combined_name", os.path.basename(out_glb),
        "--labels", labels_npy, "--label_names", names_json,
    ]
    command += (["--source_glb", source_glb, "--texture_size", texture_size] if with_texture
                else ["--no_bake"])
    _run(command)
    with open(os.path.join(out_dir, "parts.json"), "r", encoding="utf-8") as handle:
        return json.load(handle)


def complete_parts(glb, parts_glb, out_dir, mode="boxes", py_xpart=None,
                   xpart_root=DEFAULT_XPART_ROOT, model_path=DEFAULT_XPART_WEIGHTS,
                   octree_resolution=512, seed=42, condition=DEFAULT_CONDITION):
    """Hand the parts to X-Part so it can close them into solids.

    Splitting one shell leaves every part open where it was cut. X-Part regenerates each
    as a watertight shape from the whole model plus a prompt, so the cut is healed by
    generation rather than by capping geometry we never had. The prompt is a box and, by
    default, the surface our split assigned to that part -- a box alone also contains
    whatever else passes through it. See xpart_complete.py -- it runs in its own venv and
    imports nothing from here, hence the dispatch.
    """
    if mode not in COMPLETE_MODES:
        raise ValueError(f"complete must be one of {COMPLETE_MODES}, got {mode!r}")
    if mode == "off":
        return None
    print(f"[complete] X-Part prompts from {os.path.basename(parts_glb)} "
          f"({mode}, {condition} conditioning) ...")
    os.makedirs(out_dir, exist_ok=True)
    command = [
        # "boxes" is pure trimesh, so it stays in this interpreter and needs no X-Part.
        sys.executable if mode == "boxes" else (py_xpart or DEFAULT_PY_XPART),
        os.path.join(ROOT, "xpart_complete.py"),
        "--glb", glb, "--parts", parts_glb, "--out_dir", out_dir,
    ]
    if mode == "boxes":
        command.append("--boxes_only")
    else:
        command += ["--xpart_root", xpart_root, "--model_path", model_path,
                    "--octree_resolution", octree_resolution, "--seed", seed,
                    "--condition", condition]
    _run(command)
    with open(os.path.join(out_dir, "boxes.json"), "r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(
        description="name an existing over-segmentation with text prompts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--glb", required=True, help="The original, whole model")
    parser.add_argument("--split", required=True,
                        help="segment_parts.py work directory (sample_00/seg.glb + atoms.npy)")
    parser.add_argument("--prompts", nargs="+", required=True,
                        help="One entry per output part; join concepts with '+' to merge them")
    parser.add_argument("--out", required=True, help="Output glb, one node per part")
    parser.add_argument("--mesh", default=None, help="Override the reference seg.glb")
    parser.add_argument("--atoms", default=None, help="Override the atom label npy")
    parser.add_argument("--unassigned_to", default=None,
                        help="Part that absorbs units no concept claimed")
    parser.add_argument("--merge", default="name", choices=MERGE_MODES,
                        help="name = one node per prompt; unit = one node per voted unit")
    parser.add_argument("--min_unit_faces", type=int, default=None,
                        help="Connected components under this many faces are not voted on alone")
    parser.add_argument("--min_recall", type=float, default=None,
                        help="A mask claims a unit in one view once it covers this share of it")
    parser.add_argument("--view_azimuths", default=DEFAULT_VIEW_AZIMUTHS)
    parser.add_argument("--view_elevations", default=DEFAULT_VIEW_ELEVATIONS)
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--no_reuse", action="store_true",
                        help="Re-render and re-mask instead of reusing what is in --split")
    parser.add_argument("--allow_partial", action="store_true",
                        help="Accept requested names that ended up with no faces")
    parser.add_argument("--no_texture", action="store_true",
                        help="Skip the Blender bake; parts get a flat placeholder colour")
    parser.add_argument("--texture_size", type=int, default=2048)
    parser.add_argument("--py_sam3", default=None, help=f"default: {DEFAULT_PY_SAM3}")
    parser.add_argument("--sam3_model", default=DEFAULT_SAM3)
    parser.add_argument("--sam3_threshold", type=float, default=DEFAULT_SAM3_THRESHOLD)
    parser.add_argument("--concept_bank", default=DEFAULT_CONCEPT_BANK,
                        help="SAM3 v3 bank.pt (the maps.png stain). Empty = raw SAM3.")
    parser.add_argument("--no_concept_bank", action="store_true",
                        help="Disable the v3 bank and fall back to raw SAM3 scores")
    parser.add_argument("--flat_paint", default="auto", choices=FLAT_PAINT_MODES,
                        help="Temporary flat colour for a model the renders show as grey")
    parser.add_argument("--complete", default="off", choices=COMPLETE_MODES,
                        help="Hand the parts to X-Part: boxes = prompts only, "
                             "full = also regenerate each part as a closed solid")
    parser.add_argument("--condition", default=DEFAULT_CONDITION, choices=CONDITION_MODES,
                        help="What describes a part to X-Part: the faces the split "
                             "assigned to it, or (box) whatever falls inside its box")
    parser.add_argument("--py_xpart", default=None, help=f"default: {DEFAULT_PY_XPART}")
    parser.add_argument("--xpart_root", default=DEFAULT_XPART_ROOT)
    parser.add_argument("--xpart_weights", default=DEFAULT_XPART_WEIGHTS)
    parser.add_argument("--octree_resolution", type=int, default=512,
                        help="Marching-cubes resolution X-Part reconstructs each part at")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.no_concept_bank and args.concept_bank != DEFAULT_CONCEPT_BANK:
        parser.error("pass either --concept_bank or --no_concept_bank, not both")

    merge_parts(
        args.glb, args.prompts, args.split, args.out,
        mesh=args.mesh,
        atoms=args.atoms,
        unassigned_to=args.unassigned_to,
        merge=args.merge,
        min_unit_faces=args.min_unit_faces,
        min_recall=args.min_recall,
        view_azimuths=args.view_azimuths,
        view_elevations=args.view_elevations,
        radius=args.radius,
        resolution=args.resolution,
        py_sam3=args.py_sam3,
        sam3_model=args.sam3_model,
        sam3_threshold=args.sam3_threshold,
        concept_bank="" if args.no_concept_bank else args.concept_bank,
        flat_paint=args.flat_paint,
        complete=args.complete,
        py_xpart=args.py_xpart,
        xpart_root=args.xpart_root,
        xpart_weights=args.xpart_weights,
        octree_resolution=args.octree_resolution,
        seed=args.seed,
        condition=args.condition,
        reuse=not args.no_reuse,
        strict_parts=not args.allow_partial,
        with_texture=not args.no_texture,
        texture_size=args.texture_size,
    )


if __name__ == "__main__":
    main()
