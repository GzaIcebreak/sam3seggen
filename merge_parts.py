"""Name and merge an existing over-segmentation: split artifacts + prompts -> named parts.

The second half of segment_parts.py, callable on its own so the two halves can be tested
apart. Splitting is expensive and prompt-independent; naming is cheap and is the part you
re-run while deciding what the parts should be called. Point this at a split directory
(`segment_parts.py --merge off`) and re-run it with different prompts:

    python segment_parts.py --glb robot.glb --merge off --out split/units.glb
    python merge_parts.py --glb robot.glb --split split/work --prompts "head, torso, arm" \
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

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pipeline import (
    COMPLETE_MODES, CONDITION_MODES, DEFAULT_COMPLETE, DEFAULT_CONDITION,
    DEFAULT_CONCEPT_BANK, DEFAULT_HOLOPART_ROOT, DEFAULT_HOLOPART_WEIGHTS,
    DEFAULT_ASSIGN, DEFAULT_RANK_MODEL, DEFAULT_RANK_DROP, DEFAULT_RANK_ADD,
    DEFAULT_FRAGMENT_SHARE, DEFAULT_MIN_AREA_SHARE, DEFAULT_OCTREE_RESOLUTION, DEFAULT_PY_HOLOPART,
    DEFAULT_EXPORT_FROM, DEFAULT_HOLOPART_LARGE, DEFAULT_PY_XPART, DEFAULT_RADIUS,
    DEFAULT_REDRAWS, DEFAULT_REFINE, DEFAULT_REFINE_MIN_SHARE, DEFAULT_RESOLUTION,
    DEFAULT_SAM3_THRESHOLD, DEFAULT_TEXTURE_SIZE, DEFAULT_UNASSIGNED_TO,
    DEFAULT_VIEW_AZIMUTHS, DEFAULT_VIEW_ELEVATIONS, DEFAULT_XPART_ROOT, DEFAULT_XPART_WEIGHTS,
    FLAT_PAINT_MODES, MERGE_MODES, PipelineOptions, add_cli_arguments, check_cli,
)
from prompt_specs import (
    normalize_part_specs, part_names, resolve_unassigned_to,
    validate_named_rows, validate_target_name,
)
from segment_api import DEFAULT_PY_SAM3, DEFAULT_SAM3, _run


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
               concept_bank=DEFAULT_CONCEPT_BANK, raw=False, require_masks=False,
               assign=DEFAULT_ASSIGN, rank_model=DEFAULT_RANK_MODEL,
               rank_drop=DEFAULT_RANK_DROP, rank_add=DEFAULT_RANK_ADD, extra_views_dir=None):
    if reuse and os.path.isfile(out_npz):
        print(f"[guidance] reusing masks for {prompts} ({os.path.basename(out_npz)})")
        return out_npz
    print(f"[guidance] SAM3 prompts {prompts} over the view grid (assign={assign}) ...")
    command = [
        py_sam3 or DEFAULT_PY_SAM3, os.path.join(ROOT, "sam3_multiview.py"),
        "--views_dir", views_dir, "--out", out_npz,
        "--model", model, "--threshold", threshold,
        "--concept_bank", concept_bank or "",
        "--assign", assign, "--rank_drop", rank_drop, "--rank_add", rank_add,
    ]
    if rank_model:
        command += ["--rank_model", rank_model]
    if extra_views_dir:
        command += ["--extra_views_dir", extra_views_dir]
    if raw:
        command.append("--raw")
    if require_masks:
        command.append("--require_masks")
    from phrase_rescue import rescue_enabled

    if not rescue_enabled():
        command.append("--no_rescue")
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
             concept_bank=DEFAULT_CONCEPT_BANK, flat_paint="auto", reuse=True,
             require_masks=False, assign=DEFAULT_ASSIGN, rank_model=DEFAULT_RANK_MODEL,
             rank_drop=DEFAULT_RANK_DROP, rank_add=DEFAULT_RANK_ADD):
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
    # The cache key names the painter: masks painted by the ranker must not be reused for
    # a plain run and vice versa.
    overlay = "v3" if assign == "paint" else (
        f"{assign}-d{float(rank_drop):g}-a{float(rank_add):g}-"
        f"{os.path.basename(rank_model) if rank_model else 'none'}")
    # A painted run also reads the grey render (see sam3_multiview --extra_views_dir); the
    # key says so, so masks from the paint-only reader are not reused.
    dual = painted and assign == "paint"
    if dual:
        overlay += "+grey"
    from phrase_rescue import rescue_enabled

    if rescue_enabled():
        # masks made without the phrase rescue must not stand in for a rescued run
        overlay += "+rescue"
    masks_npz = sam3_masks(
        prompt_dir, prompts,
        os.path.join(work_dir, masks_name(
            prompts, unassigned_to, sam3_threshold, sam3_model,
            view_azimuths, view_elevations, bank, overlay, painted)),
        unassigned_to, py_sam3, sam3_model, sam3_threshold, reuse, concept_bank=bank,
        require_masks=require_masks, assign=assign, rank_model=rank_model,
        rank_drop=rank_drop, rank_add=rank_add,
        extra_views_dir=views_dir if dual else None)
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
    unassigned_to=DEFAULT_UNASSIGNED_TO,
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
    assign=DEFAULT_ASSIGN,
    rank_model=DEFAULT_RANK_MODEL,
    rank_drop=DEFAULT_RANK_DROP,
    rank_add=DEFAULT_RANK_ADD,
    flat_paint="auto",
    units=None,
    complete=DEFAULT_COMPLETE,
    py_xpart=None,
    xpart_root=DEFAULT_XPART_ROOT,
    xpart_weights=DEFAULT_XPART_WEIGHTS,
    py_holopart=None,
    holopart_root=DEFAULT_HOLOPART_ROOT,
    holopart_weights=DEFAULT_HOLOPART_WEIGHTS,
    octree_resolution=DEFAULT_OCTREE_RESOLUTION,
    seed=42,
    condition=DEFAULT_CONDITION,
    min_area_share=DEFAULT_MIN_AREA_SHARE,
    fragment_share=DEFAULT_FRAGMENT_SHARE,
    redraws=DEFAULT_REDRAWS,
    reuse=True,
    strict_parts=False,
    with_texture=True,
    texture_size=DEFAULT_TEXTURE_SIZE,
    holopart_large=DEFAULT_HOLOPART_LARGE,
    score_candidate=None,
    score_candidate_small=None,
    score_floor=None,
    part_min_area_share=None,
    fold_within_part=False,
    merge_gap=0.0,
    merge_max_share=None,
    refine=DEFAULT_REFINE,
    refine_min_share=DEFAULT_REFINE_MIN_SHARE,
    export_from=DEFAULT_EXPORT_FROM,
    separate=None,
):
    """Name the atoms in `split_dir` with `prompts` and write the parts into `out_glb`.

    Args:
        split_dir: a segment_parts.py work directory (sample_00/seg.glb + atoms.npy).
        merge: "name" fuses everything the vote gave the same name into one node; "unit"
            keeps one node per unit, named `<index>_<voted name>`, so a wrong name can be
            traced to a unit before it is merged away; "fragments" keeps that split and
            only folds specks (tiny, unnamed, or same-name chips) back into a neighbour.
        fragment_share: with merge=fragments, a unit below this share of the surface
            is a speck. Smaller keeps more pieces.
        unassigned_to: the part absorbing units no concept claimed. Without it those faces
            are dropped from the output.
        flat_paint: "auto" gives a model the renders show as grey a temporary flat colour
            before prompting, because SAM3 finds nothing on an untextured one.
        units: the split's shell-fused unit ids. Left None they are recomputed here, which
            is correct but wasteful when the caller just built them.
        complete: close the parts afterwards. "boxes" only writes the prompts and a
            preview (cheap, no GPU); "full" is X-Part only; "hybrid" (default) runs
            X-Part then swaps a large box-escaping solid for HoloPart.
        reuse: keep the renders and, for these exact prompts, the masks already in
            `split_dir`. Turn off to re-render (e.g. after editing the source model).

    Returns a parts.json-style manifest, one row per exported part.
    """
    import numpy as np

    from data_toolkit.lift_sam3 import load_cameras, load_masks
    from data_toolkit.parts_rebake import load_single_mesh, welded_face_adjacency
    from data_toolkit.unit_vote import (
        DEFAULT_MIN_RECALL, DEFAULT_MIN_UNIT_FACES, fold_fragment_units,
        print_report, vote,
    )

    if merge not in MERGE_MODES:
        raise ValueError(f"merge must be one of {MERGE_MODES}, got {merge!r}")
    specs = normalize_part_specs(prompts)
    expected_names = part_names(specs)
    unassigned_to = resolve_unassigned_to(unassigned_to, expected_names)
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
        py_sam3, sam3_model, sam3_threshold, concept_bank, flat_paint, reuse,
        require_masks=strict_parts, assign=assign, rank_model=rank_model,
        rank_drop=rank_drop, rank_add=rank_add)

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
    if refine == "masks" and merge == "name":
        from refine_units import refine_labels_by_masks

        print("[refine] reading the masks per face to split a unit that fused two parts ...")
        labels, changes = refine_labels_by_masks(
            reference, units, labels, expected_names, mask_set, cameras,
            float(manifest_views["camera_angle_x"]), int(manifest_views["resolution"]),
            min_share=refine_min_share)
        for change in changes:
            print(f"  unit {change['unit']:>3}: {change['faces']} faces "
                  f"({change['share_of_unit']:.0%} of it) {expected_names[change['from']]} "
                  f"-> {expected_names[change['to']]}")
        print(f"[refine] {len(changes)} patch(es) moved")
        with open(os.path.join(split_dir, "refine_report.json"), "w", encoding="utf-8") as handle:
            json.dump(changes, handle, ensure_ascii=False, indent=2)
        if changes:
            # the patches follow mask pixels and their rims are jagged; a few majority
            # passes over welded neighbours straighten the cut the solids will inherit
            import trimesh

            from source_export import smooth_labels

            welded = trimesh.Trimesh(np.asarray(reference.vertices), np.asarray(reference.faces),
                                     process=False)
            welded.merge_vertices(merge_tex=True, merge_norm=True)
            labels = smooth_labels(welded, labels, iterations=3)
            print("[refine] cut lines smoothed (3 majority passes)")
    if merge == "fragments":
        names_by_unit = [row["name"] for row in rows]
        units, folded_names, absorbed = fold_fragment_units(
            units, reference.area_faces, welded_face_adjacency(reference), names_by_unit,
            max_share=fragment_share)
        print(f"[merge] fragments: share<{fragment_share:g}, absorbed {absorbed} specks -> "
              f"{int(units.max()) + 1 if len(units) else 0} units kept")
        label_names = [f"{index:02d}_{name or 'unnamed'}"
                       for index, name in enumerate(folded_names)]
        labels = units
    elif merge == "unit":
        # nothing is dropped here, not even units no concept claimed: this output exists
        # to be looked at, and a missing piece is the hardest kind to notice.
        label_names = [f"{row['unit']:02d}_{row['name'] or 'unnamed'}" for row in rows]
        labels = units
    else:
        label_names = expected_names
    if separate and merge == "name":
        labels, label_names, splits = split_separate_instances(
            labels, label_names, welded_face_adjacency(reference),
            np.asarray(reference.area_faces), separate,
            centroids=np.asarray(reference.triangles_center))
        for name, count in splits:
            print(f"[merge] {name}: {count} separately painted instances -> "
                  + ", ".join([name] + [f"{name} {k}" for k in range(2, count + 1)]))

    labels_npy = os.path.join(split_dir, "labels.npy")
    names_json = os.path.join(split_dir, "label_names.json")
    np.save(labels_npy, labels)
    with open(names_json, "w", encoding="utf-8") as handle:
        json.dump(label_names, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(split_dir, "vote_report.json"), "w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)

    if merge == "name" and not (np.asarray(labels) >= 0).any():
        raise ValueError(
            "no part received any faces: SAM3 recognised none of the prompts and nothing "
            "absorbs the unclaimed faces. Try other words, pass unassigned_to, or "
            "merge=off for the geometric units.")
    manifest = export_labelled(mesh_path, glb, labels_npy, names_json, out_glb,
                               with_texture, texture_size, export_from=export_from)
    if strict_parts and merge == "name":
        validate_named_rows(expected_names, manifest, key="name")
    print(f"saved {out_glb} ({len(manifest)} parts)")
    complete_parts(glb, out_glb, os.path.join(out_dir, "complete"), complete,
                   py_xpart, xpart_root, xpart_weights, octree_resolution, seed,
                   condition, with_texture, texture_size, min_area_share, redraws,
                   py_holopart, holopart_root, holopart_weights,
                   holopart_large=holopart_large, score_candidate=score_candidate,
                   score_candidate_small=score_candidate_small, score_floor=score_floor,
                   part_min_area_share=part_min_area_share, fold_within_part=fold_within_part,
                   merge_gap=merge_gap, merge_max_share=merge_max_share)
    return manifest


def split_separate_instances(labels, names, adjacency, areas, words, min_share=0.05,
                             centroids=None, gap=0.03):
    """Give each instance of a part its own name: `leg`, `leg 2`, ...

    `words`: part names to split -- a list, "all", a comma string, or a dict
    {name: expected count}. The pipeline otherwise merges every instance of a name into
    one node, which is right for a rack's shelves but wrong when the user's guide paints
    the left and right leg in different colours.

    Instances are found as connected components clustered by proximity: components whose
    bounding boxes come within `gap` of the part's diagonal belong together (a boot's sole
    and its upper are two components in the remesh). With an expected count, clusters are
    merged nearest-first down to that many. Clusters under `min_share` of the part's area
    join the nearest big one. Returns (labels, names, [(name, count)]).
    """
    import numpy as np
    import trimesh

    labels = np.asarray(labels).copy()
    names = list(names)
    adjacency = np.asarray(adjacency).reshape(-1, 2)
    areas = np.asarray(areas, dtype=float)
    if isinstance(words, str):
        words = [w.strip() for w in words.split(",") if w.strip()]
    counts = dict(words) if isinstance(words, dict) else {}
    targets = list(names) if "all" in words else [w for w in words if w in names]
    report = []
    for name in targets:
        index = names.index(name)
        faces = np.flatnonzero(labels == index)
        if len(faces) < 2:
            continue
        inside = np.zeros(len(labels), dtype=bool)
        inside[faces] = True
        pairs = adjacency[inside[adjacency].all(axis=1)]
        components = [np.asarray(c) for c in
                      trimesh.graph.connected_components(pairs, nodes=faces, min_len=1)]
        wanted = counts.get(name)
        if len(components) < 2 and not (wanted and wanted >= 2 and centroids is not None):
            continue
        clusters = _cluster_components(components, centroids, gap) if centroids is not None \
            else [[c] for c in components]
        total = areas[faces].sum()
        clusters = _merge_clusters(clusters, centroids, areas, total, min_share, wanted)
        if len(clusters) < 2:
            continue
        clusters.sort(key=lambda cl: -sum(areas[c].sum() for c in cl))
        for k, cluster in enumerate(clusters[1:], start=2):
            names.append(f"{name} {k}")
            for component in cluster:
                labels[component] = len(names) - 1
        report.append((name, len(clusters)))
    return labels, names, report


def _bbox(components, centroids):
    points = centroids[np.concatenate(components)]
    return points.min(axis=0), points.max(axis=0)


def _bbox_gap(a, b):
    """Smallest axis-aligned distance between two boxes (0 when they overlap)."""
    lo = np.maximum(a[0], b[0])
    hi = np.minimum(a[1], b[1])
    return float(np.linalg.norm(np.maximum(lo - hi, 0.0)))


def _cluster_components(components, centroids, gap):
    """Union components whose face-centroid boxes come within `gap` x the part diagonal."""
    centroids = np.asarray(centroids, dtype=float)
    boxes = [_bbox([c], centroids) for c in components]
    lo, hi = _bbox(components, centroids)
    limit = gap * max(float(np.linalg.norm(hi - lo)), 1e-9)
    parent = list(range(len(components)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            if _bbox_gap(boxes[i], boxes[j]) <= limit:
                parent[find(i)] = find(j)
    groups = {}
    for i, component in enumerate(components):
        groups.setdefault(find(i), []).append(component)
    return list(groups.values())


def _kmeans_split(points, weights, k, iterations=50):
    """Cut a face set into k pieces by position; returns k index arrays into `points`.

    The cut is a 1-D, area-weighted k-means along the principal axis that is cheapest to
    cut through (see _cut_cost): two legs joined at the crotch are taller than the pair is
    wide, so a 3-D k-means would settle on feet-versus-hips; along the across-axis the
    only thing in the way is the crotch. Deterministic and independent of face order."""
    points = np.asarray(points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if k <= 1 or len(points) < k:
        return [np.arange(len(points))]
    centre = np.average(points, axis=0, weights=weights)
    spread = points - centre
    axes = np.linalg.svd(spread * np.sqrt(weights)[:, None], full_matrices=False)[2]
    axis = axes[int(np.argmin([_cut_cost(spread @ a, weights) for a in axes]))]
    coordinate = spread @ axis
    order = np.argsort(coordinate, kind="stable")
    cumulative = np.cumsum(weights[order]) / weights.sum()
    # equal-area quantile seeds along the axis
    seeds = np.array([coordinate[order][min(int(np.searchsorted(cumulative, (j + 0.5) / k)),
                                             len(order) - 1)] for j in range(k)])
    assign = None
    for _ in range(iterations):
        new_assign = np.abs(coordinate[:, None] - seeds[None, :]).argmin(axis=1)
        if assign is not None and np.array_equal(new_assign, assign):
            break
        assign = new_assign
        for j in range(k):
            member = assign == j
            if member.any():
                seeds[j] = np.average(coordinate[member], weights=weights[member])
    return [np.flatnonzero(assign == j) for j in range(k)]


def _cut_cost(coordinate, weights, slab=0.1):
    """Share of the surface that a plane cut at the area-balanced position would slice
    through (faces within a slab of `slab` x the extent). Two legs joined at the crotch
    are cheap to cut left/right (only the crotch is in the slab) and expensive to cut
    top/bottom (both legs are). Ties favour the longer axis."""
    order = np.argsort(coordinate)
    cumulative = np.cumsum(weights[order])
    median = coordinate[order][int(np.searchsorted(cumulative, cumulative[-1] / 2))]
    extent = float(coordinate.max() - coordinate.min())
    if extent <= 1e-12:
        return 2.0
    inside = np.abs(coordinate - median) <= slab * extent / 2
    return float(weights[inside].sum() / max(weights.sum(), 1e-12)) - 1e-6 * extent


def _merge_clusters(clusters, centroids, areas, total, min_share, wanted):
    """Fold crumbs into the nearest big cluster; then merge nearest pairs down to `wanted`."""
    import numpy as np

    def area(cl):
        return sum(areas[c].sum() for c in cl)

    def centre(cl):
        if centroids is None:
            return None
        pts = np.concatenate([np.asarray(centroids)[c] for c in cl])
        return pts.mean(axis=0)

    big = [cl for cl in clusters if area(cl) >= min_share * total]
    small = [cl for cl in clusters if area(cl) < min_share * total]
    if not big:
        return [sum(clusters, [])]
    for cl in small:
        if centroids is None:
            target = max(range(len(big)), key=lambda i: area(big[i]))
        else:
            c = centre(cl)
            target = min(range(len(big)), key=lambda i: float(np.linalg.norm(centre(big[i]) - c)))
        big[target] = big[target] + cl
    if wanted and centroids is not None and len(big) < wanted:
        # the instances touch (two legs meet at the crotch): cut the largest cluster by
        # position into as many pieces as are missing
        big.sort(key=area, reverse=True)
        faces = np.concatenate(big[0])
        pieces = _kmeans_split(np.asarray(centroids)[faces], areas[faces], wanted - len(big) + 1)
        big = [[faces[piece]] for piece in pieces if len(piece)] + big[1:]
    while wanted and len(big) > wanted:
        if centroids is None:
            big.sort(key=area)
            big[1] = big[1] + big[0]
            big.pop(0)
            continue
        centres = [centre(cl) for cl in big]
        best = None
        for i in range(len(big)):
            for j in range(i + 1, len(big)):
                d = float(np.linalg.norm(centres[i] - centres[j]))
                if best is None or d < best[0]:
                    best = (d, i, j)
        _, i, j = best
        big[i] = big[i] + big[j]
        big.pop(j)
    return big


def export_labelled(mesh_path, source_glb, labels_npy, names_json, out_glb,
                    with_texture=True, texture_size=2048, step="[export]",
                    export_from=DEFAULT_EXPORT_FROM):
    """Cut the reference mesh by a face label array and write one node per label.

    export_from="source" cuts the source model itself (labels carried over from the
    remesh, its own texture or flat material kept) and falls back to the remesh + bake
    when the source is several meshes.
    """
    if export_from == "source" and with_texture and source_glb:
        from source_export import export_from_source

        manifest = export_from_source(mesh_path, source_glb, labels_npy, names_json, out_glb,
                                      step=step)
        if manifest is not None:
            return manifest
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
                   octree_resolution=512, seed=42, condition=DEFAULT_CONDITION,
                   with_texture=True, texture_size=DEFAULT_TEXTURE_SIZE,
                   min_area_share=DEFAULT_MIN_AREA_SHARE, redraws=DEFAULT_REDRAWS,
                   py_holopart=None, holopart_root=DEFAULT_HOLOPART_ROOT,
                   holopart_weights=DEFAULT_HOLOPART_WEIGHTS,
                   part_min_area_share=None, fold_within_part=False,
                   holopart_large="escape", score_candidate=None, score_floor=None,
                   merge_gap=0.0, merge_max_share=None, score_candidate_small=None):
    """Close the open parts. Default is hybrid: X-Part, then HoloPart on large escapees.

    Splitting one shell leaves every part open where it was cut. X-Part regenerates each
    as a watertight shape from the whole model plus a prompt. A large solid that still
    overruns its box is replaced by that instance's HoloPart draw. See xpart_complete.py
    and holopart_complete.py -- each runs in its own venv, hence the dispatch.
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
                    "--condition", condition,
                    "--min_area_share", min_area_share, "--redraws", redraws]
    for name, share in parse_part_floors(part_min_area_share).items():
        command += ["--part_min_area_share", f"{name}={share}"]
    if fold_within_part:
        command.append("--fold_within_part")
    if merge_gap:
        command += ["--merge_gap", merge_gap]
        if merge_max_share is not None:
            command += ["--merge_max_share", merge_max_share]
    _run(command)
    if mode == "hybrid":
        _hybrid_swap(out_dir, seed, py_holopart, holopart_root, holopart_weights,
                     holopart_large, score_candidate, score_floor, score_candidate_small)
    closed = os.path.join(out_dir, "xpart_parts.glb")
    if mode in ("full", "hybrid") and with_texture and os.path.isfile(closed):
        raw = os.path.join(out_dir, "xpart_parts_raw.glb")
        os.replace(closed, raw)
        print("[complete] baking source albedo onto the closed solids ...")
        _run([
            sys.executable, os.path.join(ROOT, "data_toolkit", "parts_rebake.py"),
            "--completed", raw, "--source_glb", glb, "--out_dir", out_dir,
            "--combined_name", "xpart_parts.glb", "--texture_size", texture_size,
            # Ceiling only: bake_completed tightens per part from the measured gap.
            "--cage_extrusion", "0.05", "--max_ray_distance", "0.15",
        ])
    with open(os.path.join(out_dir, "boxes.json"), "r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_part_floors(value):
    """{name: share} from a dict or a 'name=share,name=share' string (HTTP form)."""
    if not value:
        return {}
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items()}
    floors = {}
    for item in str(value).split(","):
        if not item.strip():
            continue
        name, sep, share = item.rpartition("=")
        if not sep or not name.strip():
            raise ValueError(f"part_min_area_share expects name=share[,name=share], got {value!r}")
        floors[name.strip()] = float(share)
    return floors


def _hybrid_swap(out_dir, seed, py_holopart, holopart_root, holopart_weights,
                 holopart_large="escape", score_candidate=None, score_floor=None,
                 score_candidate_small=None):
    """Run HoloPart on the instances the hybrid rule picks (only those); assemble either way.

    holopart_large: "escape" / "always" decide from box escape and size; "score" measures
    every X-Part solid against its open surface, draws HoloPart for the low scorers and
    keeps the better one (or the open surface when both are below the floor).
    """
    from hybrid_complete import (LARGE_POLICIES, apply_hybrid, apply_scored,
                                 decisions_from_instances, instance_node_name,
                                 score_candidates)

    instances = os.path.join(out_dir, "xpart_instances.glb")
    boxes_path = os.path.join(out_dir, "boxes.json")
    if not os.path.isfile(instances) or not os.path.isfile(boxes_path):
        raise SystemExit("hybrid repair needs xpart_instances.glb and boxes.json")
    with open(boxes_path, "r", encoding="utf-8") as handle:
        boxes = json.load(handle)
    if holopart_large == "score":
        thresholds = {k: v for k, v in (("candidate", score_candidate),
                                        ("candidate_small", score_candidate_small))
                      if v is not None}
        decisions = score_candidates(out_dir, **thresholds)
        wants = [decision["candidate"] for decision in decisions]
    elif holopart_large in LARGE_POLICIES:
        decisions = decisions_from_instances(instances, boxes, large_policy=holopart_large)
        wants = [decision["backend"] == "holopart" for decision in decisions]
    else:
        raise ValueError(f"holopart_large must be one of {LARGE_POLICIES + ('score',)}, "
                         f"got {holopart_large!r}")
    holopart_glb = None
    if any(wants):
        open_glb = os.path.join(out_dir, "open_instances.glb")
        if not os.path.isfile(open_glb):
            raise SystemExit("hybrid repair needs open_instances.glb from X-Part")
        chosen = [instance_node_name(row) for row, want in zip(boxes, wants) if want]
        print(f"[complete] HoloPart ({holopart_large}) on {len(chosen)} open instance(s) ...")
        command = [
            py_holopart or DEFAULT_PY_HOLOPART,
            os.path.join(ROOT, "holopart_complete.py"),
            "--parts", open_glb, "--out_dir", out_dir,
            "--holopart_root", holopart_root,
            "--weights", holopart_weights,
            "--seed", seed,
        ]
        for name in chosen:
            command += ["--only", name]
        _run(command)
        holopart_glb = os.path.join(out_dir, "holopart_instances.glb")
    if holopart_large == "score":
        apply_scored(out_dir, decisions, holopart_glb,
                     **({} if score_floor is None else {"floor": score_floor}))
    else:
        apply_hybrid(out_dir, holopart_glb, large_policy=holopart_large)


def main():
    parser = argparse.ArgumentParser(
        description="name an existing over-segmentation with text prompts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--glb", required=True, help="The original, whole model")
    parser.add_argument("--split", required=True,
                        help="segment_parts.py work directory (sample_00/seg.glb + atoms.npy)")
    parser.add_argument("--prompts", nargs="+", required=True,
                        help="One comma-separated sentence: 'head, torso, arm'. "
                             "'+' still merges concepts ('body=head+face').")
    parser.add_argument("--out", required=True, help="Output glb, one node per part")
    parser.add_argument("--mesh", default=None, help="Override the reference seg.glb")
    parser.add_argument("--atoms", default=None, help="Override the atom label npy")
    parser.add_argument("--py_sam3", default=None, help=f"default: {DEFAULT_PY_SAM3}")
    parser.add_argument("--sam3_model", default=DEFAULT_SAM3)
    add_cli_arguments(parser, split=False, merge_off=False)
    args = parser.parse_args()
    check_cli(parser, args)
    options = PipelineOptions.from_namespace(args)
    merge_parts(
        args.glb, args.prompts, args.split, args.out,
        mesh=args.mesh, atoms=args.atoms,
        py_sam3=args.py_sam3, sam3_model=args.sam3_model,
        **options.merge_kwargs(),
    )


if __name__ == "__main__":
    main()
