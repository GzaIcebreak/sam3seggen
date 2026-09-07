"""Decimate a GLB to at most --max_faces triangles with Blender's Decimate (collapse) modifier, keeping
UVs and materials, so GeoSAM2 (per-face sampling + Python post-processing, ~2 GB RAM per 100k faces on
its worst path) can take multi-million-face assets.

    finetune\run_ft.bat decimate_glb.py --glb E:\...\飞机.glb --out E:\...\plane.glb --max_faces 200000
"""
import argparse
import os
import time

import bpy


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def total_faces(objs):
    n = 0
    for o in objs:
        o.data.calc_loop_triangles()
        n += len(o.data.loop_triangles)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_faces", type=int, default=200000)
    args = ap.parse_args()

    t0 = time.time()
    clear_scene()
    bpy.ops.import_scene.gltf(filepath=args.glb)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    n = total_faces(meshes)
    print(f"{os.path.basename(args.glb)}: {n} triangles in {len(meshes)} mesh(es)", flush=True)
    if n > args.max_faces:
        ratio = args.max_faces / n
        for o in meshes:
            bpy.context.view_layer.objects.active = o
            mod = o.modifiers.new("dec", "DECIMATE")
            mod.decimate_type = "COLLAPSE"
            mod.ratio = ratio
            mod.use_collapse_triangulate = True
            bpy.ops.object.modifier_apply(modifier=mod.name)
        print(f"  decimated with ratio {ratio:.4f} -> {total_faces(meshes)} triangles", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=args.out, export_format="GLB", export_yup=True, export_apply=True)
    print(f"saved {args.out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
