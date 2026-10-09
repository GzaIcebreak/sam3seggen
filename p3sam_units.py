"""P3-SAM as the pipeline's source of geometric units.

The split names units by multi-view SAM3 voting, and the units used to come only from
SegviGen's over-segmentation. SegviGen draws boundaries where its samples happen to
disagree, so on hard-surface and articulated models it fuses what a person would
separate (the dog's upper legs, tail and belly in one unit) and merges same-named
instances (four wheels as one `wheel`). P3-SAM (Hunyuan3D-Part) is a native 3D part
segmenter that cuts at geometric creases: on the figure, the mech, the cabinet, the
shield and the ship it gives clean joint-level pieces and every instance apart; on a
smooth organic blob (the dog, the cat) it finds nothing and returns one part.

So `units=auto` asks P3-SAM first and takes its pieces as the atoms when it found a real
split (at least P3SAM_MIN_PARTS parts of 1%+, the largest under P3SAM_MAX_LARGEST of the
area); otherwise the SegviGen samples are drawn as before. The naming, merging by name,
mask refinement, source cut and repair downstream are unchanged -- P3-SAM only replaces
where the geometry is cut. Its pieces are written as a seg.glb in SegviGen's convention
(un-shared vertices, one flat colour per atom, the y<->z frame) so that everything that
reads a reference mesh keeps working, including the flat paint for untextured models.
"""
from __future__ import annotations

import colorsys
import json
import os
import subprocess

import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_P3SAM_ROOT = os.environ.get("SEGVIGEN_P3SAM_ROOT", "/root/autodl-tmp/Hunyuan3D-Part/P3-SAM")
DEFAULT_P3SAM_WEIGHTS = os.environ.get(
    "SEGVIGEN_P3SAM_WEIGHTS", "/root/autodl-tmp/Hunyuan3D-Part/weights/p3sam/p3sam.safetensors")
P3SAM_MIN_PARTS = 3          # fewer real parts than this: P3-SAM did not split the model
P3SAM_MIN_PART_SHARE = 0.01  # a part under this share of the area is not a real part
P3SAM_MAX_LARGEST = 0.85     # one part this big: P3-SAM gave up (the dog, 98.8%)
P3SAM_POINT_NUM = 100000
P3SAM_PROMPT_BS = 8
# source glb frame -> SegviGen seg.glb frame: the inverse of source_export's "y<->z-flip"
SOURCE_TO_SEG = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def run_p3sam(glb, out_dir, py_xpart, p3sam_root=DEFAULT_P3SAM_ROOT, weights=DEFAULT_P3SAM_WEIGHTS,
              point_num=P3SAM_POINT_NUM, prompt_bs=P3SAM_PROMPT_BS, seed=42, reuse=True):
    """stats.json of P3-SAM on `glb`, running p3sam_segment.py in the X-Part venv unless
    `out_dir` already holds a finished run."""
    stats_path = os.path.join(out_dir, "stats.json")
    needed = ("mesh.glb", "face_ids.npy", "stats.json")
    if reuse and all(os.path.isfile(os.path.join(out_dir, name)) for name in needed):
        print(f"[units] reusing the P3-SAM split in {out_dir}")
    else:
        os.makedirs(out_dir, exist_ok=True)
        command = [py_xpart, os.path.join(ROOT, "p3sam_segment.py"), "--glb", glb,
                   "--out_dir", out_dir, "--p3sam_root", p3sam_root, "--weights", weights,
                   "--point_num", str(point_num), "--prompt_bs", str(prompt_bs), "--seed", str(seed)]
        env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF=os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"))
        subprocess.run(command, check=True, env=env)
    with open(stats_path, encoding="utf-8") as handle:
        return json.load(handle)


def accept(stats, min_parts=P3SAM_MIN_PARTS, min_share=P3SAM_MIN_PART_SHARE,
           max_largest=P3SAM_MAX_LARGEST):
    """(ok, reason): did P3-SAM actually split the model?"""
    real = [s for s in stats.get("shares", []) if s >= min_share]
    largest = stats.get("largest_share", 0.0)
    if len(real) < min_parts:
        return False, f"only {len(real)} part(s) of {min_share:.0%}+ area (need {min_parts})"
    if largest > max_largest:
        return False, f"one part is {largest:.0%} of the area (limit {max_largest:.0%})"
    return True, f"{len(real)} parts, largest {largest:.0%}"


def atom_colour(index):
    """Distinct flat colours, golden-ratio hue steps: SegviGen's seg.glb convention."""
    hue = (index * 0.6180339887) % 1.0
    value = 0.95 if index % 2 == 0 else 0.7
    r, g, b = colorsys.hsv_to_rgb(hue, 0.8, value)
    return [int(round(255 * r)), int(round(255 * g)), int(round(255 * b)), 255]


def seg_glb_from_ids(mesh_glb, face_ids, out_glb, rotation=SOURCE_TO_SEG):
    """Write P3-SAM's labelled mesh as a seg.glb in SegviGen's convention; return atoms.

    Atoms are the part ids renumbered 0..K-1 by area (largest first); a face P3-SAM left
    without a part takes the id of the nearest labelled face. Vertices are un-shared so
    each face keeps its own flat colour (what flat_paint.part_colors reads), and rotated
    from the source frame into the seg frame so the vote, the refinement and the source
    cut see what they see from a SegviGen sample."""
    from scipy.spatial import cKDTree

    mesh = trimesh.load(mesh_glb, force="mesh", process=False)
    face_ids = np.asarray(face_ids).astype(np.int64)
    if len(face_ids) != len(mesh.faces):
        raise ValueError(f"{len(face_ids)} face ids for {len(mesh.faces)} faces")
    areas = np.asarray(mesh.area_faces)
    ids = [int(i) for i in np.unique(face_ids) if i >= 0]
    if not ids:
        raise ValueError("P3-SAM labelled no face")
    order = sorted(ids, key=lambda i: -float(areas[face_ids == i].sum()))
    remap = {old: new for new, old in enumerate(order)}
    atoms = np.array([remap.get(int(i), -1) for i in face_ids], dtype=np.int64)
    missing = atoms < 0
    if missing.any():
        centroids = np.asarray(mesh.triangles_center)
        tree = cKDTree(centroids[~missing])
        atoms[missing] = atoms[~missing][tree.query(centroids[missing])[1]]
    vertices = np.asarray(mesh.vertices, dtype=np.float64) @ np.asarray(rotation).T
    faces = np.asarray(mesh.faces)
    unshared = trimesh.Trimesh(vertices[faces].reshape(-1, 3),
                               np.arange(faces.size).reshape(-1, 3), process=False)
    colours = np.array([atom_colour(a) for a in range(len(order))], dtype=np.uint8)
    unshared.visual = trimesh.visual.ColorVisuals(unshared, face_colors=colours[atoms])
    unshared.export(out_glb)
    return atoms


def p3sam_atoms(glb, work_dir, py_xpart, p3sam_root=DEFAULT_P3SAM_ROOT, weights=DEFAULT_P3SAM_WEIGHTS,
                point_num=P3SAM_POINT_NUM, prompt_bs=P3SAM_PROMPT_BS, seed=42, reuse=True,
                min_parts=P3SAM_MIN_PARTS, max_largest=P3SAM_MAX_LARGEST, runner=run_p3sam):
    """P3-SAM's split of `glb` as the pipeline's reference mesh + atoms, or None when it
    found no usable split (or failed): {"seg_glb", "atoms_npy", "atoms", "report", "stats"}."""
    out_dir = os.path.join(work_dir, "p3sam")
    try:
        stats = runner(glb, out_dir, py_xpart, p3sam_root, weights, point_num, prompt_bs, seed, reuse)
    except (subprocess.CalledProcessError, OSError, ValueError) as exc:
        print(f"[units] P3-SAM failed ({type(exc).__name__}: {str(exc)[:160]}); "
              "falling back to the SegviGen samples")
        return None
    ok, reason = accept(stats, min_parts=min_parts, max_largest=max_largest)
    if not ok:
        print(f"[units] P3-SAM did not split this model ({reason}); using the SegviGen samples")
        return None
    seg_glb = os.path.join(out_dir, "seg.glb")
    atoms = seg_glb_from_ids(os.path.join(out_dir, "mesh.glb"),
                             np.load(os.path.join(out_dir, "face_ids.npy")), seg_glb)
    atoms_npy = os.path.join(out_dir, "atoms.npy")
    np.save(atoms_npy, atoms)
    report = {
        "source": "p3sam", "samples": [], "labels_per_sample": [stats["parts"]],
        "mirror_axis": None, "mirror_share": None,
        "raw_atoms": stats["parts"], "atoms": int(atoms.max()) + 1,
        "largest_share": stats["largest_share"], "shares": stats["shares"],
        "unlabelled_share": stats.get("unlabelled_share", 0.0),
        "seconds": stats.get("seconds"), "load_seconds": stats.get("load_seconds"),
    }
    print(f"[units] P3-SAM: {reason}; {int(atoms.max()) + 1} atoms in "
          f"{stats.get('seconds', '?')} s -> {seg_glb}")
    return {"seg_glb": seg_glb, "atoms_npy": atoms_npy, "atoms": atoms, "report": report,
            "stats": stats}
