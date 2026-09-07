"""Turn GeoSAM2 face labels into a viewable / cuttable GLB in the source asset's frame.

    finetune\run_ft.bat geosam2_to_glb.py --mesh <geosam2 .glb> --labels <.npy> --ref <original.glb>
                                          --out parts.glb [--labels_json labels.json]

GeoSAM2 exports its mesh Z-up, centred and scaled to unit max extent, with labels as vertex colours
(which Blender's glTF importer does not wire into the material, so bpy renders them white). This
writes one sub-mesh per label with a flat PBR material instead, rotated back to Y-up and rescaled to
the reference GLB's bounding box, so it renders with data_toolkit/render_cond_view.py exactly like a
SegviGen output and can be split into parts the same way. Sub-meshes are named after the part
(labels.json ids) or "label_<k>"; unlabelled faces (0 / 999) become one grey "unlabelled" mesh.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import GREY  # noqa: E402

PALETTE = [(228, 26, 28), (55, 126, 184), (77, 175, 74), (152, 78, 163), (255, 127, 0), (255, 255, 51),
           (166, 86, 40), (247, 129, 191), (0, 158, 115), (86, 180, 233), (213, 94, 0), (204, 121, 167),
           (120, 94, 240), (230, 159, 0), (0, 114, 178), (176, 122, 161), (60, 180, 75), (145, 30, 180),
           (245, 130, 48), (70, 240, 240)]


def load_mesh(path: str) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene")
    meshes = [m for m in loaded.dump() if isinstance(m, trimesh.Trimesh)]
    return meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)


def to_reference_frame(mesh: trimesh.Trimesh, ref: trimesh.Trimesh) -> trimesh.Trimesh:
    """Undo GeoSAM2's normalisation: try both x-axis quarter turns, keep the one whose extents match."""
    ref_ext = ref.extents / ref.extents.max()
    best = None
    for deg in (-90, 90, 0):
        m = mesh.copy()
        m.apply_transform(trimesh.transformations.rotation_matrix(np.radians(deg), [1, 0, 0]))
        ext = m.extents / m.extents.max()
        err = float(np.abs(ext - ref_ext).sum())
        if best is None or err < best[0]:
            best = (err, m)
    m = best[1]
    m.apply_translation(-m.bounds.mean(0))
    m.apply_scale(ref.extents.max() / m.extents.max())
    m.apply_translation(ref.bounds.mean(0))
    return m


def build_scene(mesh: trimesh.Trimesh, labels: np.ndarray, names: dict[int, str], unlabelled: set[int]) -> trimesh.Scene:
    scene = trimesh.Scene()
    ids = [int(k) for k in np.unique(labels)]
    k_named = 0
    for k in ids:
        faces = np.nonzero(labels == k)[0]
        sub = mesh.submesh([faces], append=True)
        if k in unlabelled:
            name, color = "unlabelled", GREY
        else:
            name = names.get(k, f"label_{k}")
            color = PALETTE[k_named % len(PALETTE)]
            k_named += 1
        mat = trimesh.visual.material.PBRMaterial(baseColorFactor=[*color, 255], metallicFactor=0.0, roughnessFactor=0.9)
        sub.visual = trimesh.visual.TextureVisuals(material=mat)
        scene.add_geometry(sub, node_name=f"{name}_{k}", geom_name=f"{name}_{k}")
    return scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--ref", required=True, help="Original asset GLB (frame / scale target)")
    ap.add_argument("--labels_json", default=None)
    ap.add_argument("--unlabelled", type=int, nargs="*", default=[0, 999])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mesh = load_mesh(args.mesh)
    labels = np.rint(np.load(args.labels)).astype(np.int64)
    if len(labels) != len(mesh.faces):
        raise SystemExit(f"{len(labels)} labels for {len(mesh.faces)} faces")
    names = {}
    if args.labels_json:
        with open(args.labels_json, "r", encoding="utf-8") as f:
            names = {int(k): v for k, v in json.load(f).get("ids", {}).items()}
    mesh = to_reference_frame(mesh, load_mesh(args.ref))
    scene = build_scene(mesh, labels, names, set(args.unlabelled))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    scene.export(args.out)
    print(f"saved {args.out}: {len(scene.geometry)} parts")


if __name__ == "__main__":
    main()
