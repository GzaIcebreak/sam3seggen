"""Regenerate open part instances as closed solids with HoloPart.

Runs in the HoloPart venv and imports nothing from pipeline. The input glb is the
per-instance open surfaces X-Part already wrote (open_instances.glb): one node per
instance, names ``00_head``. Output keeps those names so hybrid_complete can pair them.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from xpart_complete import load_part_nodes

DEFAULT_HOLOPART_ROOT = os.environ.get(
    "SEGVIGEN_HOLOPART_ROOT", "/root/autodl-tmp/HoloPart")
DEFAULT_HOLOPART_WEIGHTS = os.environ.get(
    "SEGVIGEN_HOLOPART_WEIGHTS",
    "/root/autodl-tmp/HoloPart/pretrained_weights/HoloPart")


def _cumesh_clean(mesh):
    mesh.remove_duplicate_faces()
    mesh.repair_non_manifold_edges()
    mesh.remove_small_connected_components(1e-5)
    mesh.fill_holes(max_hole_perimeter=3e-2)


def cumesh_simplify(mesh, n_faces):
    """GPU quadric simplification with TRELLIS.2's clean-up (o_voxel.postprocess.to_glb).

    Takes HoloPart's normalised DMC mesh (about [-1, 1]), so the absolute clean-up
    tolerances mean the same thing for every part.
    """
    import torch
    import cumesh

    welded = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.faces), process=False)
    welded.merge_vertices()
    cm = cumesh.CuMesh()
    cm.init(torch.tensor(np.asarray(welded.vertices), dtype=torch.float32, device="cuda"),
            torch.tensor(np.asarray(welded.faces), dtype=torch.int32, device="cuda"))
    if cm.num_faces > 3 * n_faces:
        cm.simplify(3 * n_faces)
    _cumesh_clean(cm)
    cm.simplify(n_faces)
    _cumesh_clean(cm)
    vertices, faces = cm.read()
    return trimesh.Trimesh(vertices.cpu().numpy(), faces.cpu().numpy())


def instance_sort_key(name):
    prefix = str(name).split("_", 1)[0]
    return (int(prefix), name) if prefix.isdigit() else (10**9, name)


def write_ordered(parts_glb, dest):
    """Re-export so HoloPart's dict iteration matches 00, 01, 02, ..."""
    scene = trimesh.Scene()
    names = []
    for name, mesh in sorted(load_part_nodes(parts_glb), key=lambda item: instance_sort_key(item[0])):
        scene.add_geometry(mesh, geom_name=name)
        names.append(name)
    if not names:
        raise SystemExit(f"no mesh found in {parts_glb}")
    scene.export(dest)
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parts", required=True,
                        help="Open instances glb (one node per instance)")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--holopart_root", default=DEFAULT_HOLOPART_ROOT)
    parser.add_argument("--weights", default=DEFAULT_HOLOPART_WEIGHTS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_chunks", type=int, default=100000,
                        help="decoder query points per call; 20000 made 1178 calls on the robot (17 s), 100000 makes 235 (10 s) for +0.8 GB")
    parser.add_argument("--max_faces", type=int, default=200000,
                        help="HoloPart's own script decimates every solid to 10,000 faces; "
                             "that flattens a 25%-of-the-model breastplate to a smooth shell. "
                             "Its marching cubes run at 505^3, so keep up to this many.")
    parser.add_argument("--simplify", choices=("cumesh", "pymeshlab"), default="cumesh",
                        help="cumesh: GPU, ~0.3 s per solid; pymeshlab: HoloPart's own, ~15 s")
    parser.add_argument("--only", action="append", default=[], metavar="NODE",
                        help="Generate only these instances (repeatable). Every instance "
                             "still describes the whole shape HoloPart conditions on.")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    ordered = os.path.join(out_dir, "holopart_input.glb")
    names = write_ordered(os.path.abspath(args.parts), ordered)

    holopart_root = os.path.abspath(args.holopart_root)
    if holopart_root not in sys.path:
        sys.path.insert(0, holopart_root)
    os.chdir(holopart_root)

    import torch
    from holopart.pipelines.pipeline_holopart import HoloPartPipeline
    import scripts.inference_holopart as inference_holopart
    from scripts.inference_holopart import prepare_data, run_holopart

    # run_holopart calls simplify_mesh(mesh, 10000) from its module globals; raise the cap.
    original_simplify = inference_holopart.simplify_mesh

    def simplify(mesh, n_faces):
        n_faces = max(int(n_faces), args.max_faces)
        if args.simplify == "cumesh":
            try:
                return cumesh_simplify(mesh, n_faces)
            except (ImportError, RuntimeError) as exc:
                print(f"[holopart] cumesh simplify failed ({exc}); falling back to pymeshlab")
        return original_simplify(mesh, n_faces)

    inference_holopart.simplify_mesh = simplify

    weights = os.path.abspath(args.weights)
    if not os.path.isdir(weights):
        raise SystemExit(f"HoloPart weights not found: {weights}")
    print(f"[holopart] {len(names)} instances from {os.path.basename(args.parts)}")
    pipe = HoloPartPipeline.from_pretrained(weights).to("cuda", torch.float16)
    batch = prepare_data(ordered, device="cuda")
    if args.only:
        missing = [name for name in args.only if name not in names]
        if missing:
            raise SystemExit(f"--only instance(s) not in {args.parts}: {missing}")
        # Slice after prepare_data: whole_cond is sampled from all instances, so the
        # context keeps every part while only the chosen ones pay for a diffusion run.
        picked = [names.index(name) for name in args.only]
        index = torch.tensor(picked, device="cuda")
        for key in ("whole_cond", "part_cond", "part_local_cond"):
            batch[key] = batch[key].index_select(0, index)
        for key in ("part_id_list", "part_center_list", "part_scale_list"):
            batch[key] = [batch[key][i] for i in picked]
        names = [names[i] for i in picked]
        print(f"[holopart] generating {len(names)} of the instances: {names}")
    def draw(sub_batch):
        scene = run_holopart(
            pipe,
            batch=sub_batch,
            batch_size=args.batch_size,
            seed=args.seed,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            num_chunks=args.num_chunks,
            device="cuda",
        )
        return list(scene.geometry.values())

    def slice_batch(indices):
        index = torch.tensor(indices, device="cuda")
        sub = dict(batch)
        for key in ("whole_cond", "part_cond", "part_local_cond"):
            sub[key] = batch[key].index_select(0, index)
        for key in ("part_id_list", "part_center_list", "part_scale_list"):
            sub[key] = [batch[key][i] for i in indices]
        return sub

    try:
        meshes = draw(batch)
        if len(meshes) != len(names):
            raise RuntimeError(f"HoloPart returned {len(meshes)} solids for {len(names)} instances")
        results = list(zip(names, meshes))
    except Exception as exc:
        # One instance can take the whole batch down (an empty occupancy grid raised
        # inside flash_extract_geometry on a shovel); draw them one at a time and skip
        # the ones that fail -- the score policy falls back to X-Part or the open surface.
        print(f"[holopart] batch failed ({type(exc).__name__}: {str(exc)[:120]}); "
              "drawing the instances one at a time")
        torch.cuda.empty_cache()
        results = []
        for i, name in enumerate(names):
            try:
                single = draw(slice_batch([i]))
                if len(single) == 1:
                    results.append((name, single[0]))
                else:
                    print(f"  {name}: HoloPart returned {len(single)} solids; skipped")
            except Exception as inner:
                print(f"  {name}: HoloPart failed ({type(inner).__name__}: {str(inner)[:100]}); skipped")
            torch.cuda.empty_cache()
        if not results:
            raise SystemExit("HoloPart produced no solid for any instance")
    out = trimesh.Scene()
    for name, mesh in results:
        out.add_geometry(mesh, geom_name=name)
        ext = np.asarray(mesh.bounds[1] - mesh.bounds[0])
        print(f"  {name:<18} faces={len(mesh.faces)} ext={np.round(ext, 3)}")
    dest = os.path.join(out_dir, "holopart_instances.glb")
    out.export(dest)
    print(f"saved {dest}")


if __name__ == "__main__":
    main()
