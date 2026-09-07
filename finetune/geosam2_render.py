"""Render objects in GeoSAM2's 12-view convention using the `bpy` module (no Blender binary needed).

GeoSAM2 ships `geosam2_render.py` as a `blender -b -P` script. This wrapper imports that file as a
module and re-runs its `process()` body with two changes needed here:
  * Blender 4.1+ removed `Mesh.use_auto_smooth`; the equivalent is the "smooth by angle" operator.
  * The engine is selectable (EEVEE needs a GL context which the bpy module may not have; Cycles
    on CUDA always works) via --engine, default tries EEVEE and falls back to Cycles.
Everything else (cameras, normalisation, depth/normal passes, meta.json) is GeoSAM2's own code, so
the output is exactly what its `inference.py` expects:
  <out>/<obj>/color_0000..0011.webp, depth_*.exr, normal_*.webp, meta.json, mesh.glb

Run with finetune\run_ft.bat (bpy lives in the main venv):
  finetune\run_ft.bat finetune\geosam2_render.py --objects_file E:\...\pv_hard.txt --out E:\...\geosam2\renders
  finetune\run_ft.bat finetune\geosam2_render.py --glb E:\...\小狗.glb --name dog --out ...
"""
import argparse
import glob
import json
import math
import os
import shutil
import sys
import time

GEOSAM2_ROOT = os.environ.get("GEOSAM2_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "GeoSAM2"))
sys.path.insert(0, GEOSAM2_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Vector  # noqa: E402
import importlib.util  # noqa: E402


def _load_geosam2_script():
    """GeoSAM2's render script shares this file's name, so load it explicitly by path."""
    path = os.path.join(GEOSAM2_ROOT, "geosam2_render.py")
    spec = importlib.util.spec_from_file_location("geosam2_render_upstream", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gr = _load_geosam2_script()  # only its helpers are used


def init_engine(engine: str):
    scene = bpy.context.scene
    # Blender 4.x defaults to AgX which renders the bundled GeoSAM2 samples' mid-grey much darker;
    # the shipped example renders match the Standard transform.
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    if engine == "EEVEE":
        gr.eevee_init()
        return
    scene.render.engine = "CYCLES"
    scene.cycles.samples = gr.RENDER_SAMPLES
    scene.cycles.use_denoising = True
    scene.render.use_high_quality_normals = True
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "CUDA"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type == "CUDA"
    scene.cycles.device = "GPU"


def smooth_by_angle(mesh_objects, angle_deg: float = 30.0):
    """Blender < 4.1: legacy auto smooth; >= 4.1: the operator that replaced it."""
    for obj in mesh_objects:
        if hasattr(obj.data, "use_auto_smooth"):
            obj.data.use_auto_smooth = True
            obj.data.auto_smooth_angle = np.deg2rad(angle_deg)
            continue
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        try:
            bpy.ops.object.shade_smooth_by_angle(angle=math.radians(angle_deg))
        except Exception:
            bpy.ops.object.shade_smooth()


def process(filepath: str, types: str, output_path: str, engine: str) -> bool:
    """GeoSAM2's process() with the two compatibility changes described in the module docstring."""
    init_engine(engine)
    gr.clear_scene()
    gr.import_models(filepath, types)
    gr.reset_keyframes()

    bpy.ops.object.select_by_type(type="MESH")
    os.makedirs(output_path, exist_ok=True)
    shutil.copy(filepath, os.path.join(output_path, f"mesh.{types}"))

    mesh_objects = [o for o in bpy.context.scene.objects
                    if o.type == "MESH" and o.visible_get() and not o.hide_get()]
    smooth_by_angle(mesh_objects)
    for obj in bpy.data.objects:
        if obj.animation_data is not None:
            obj.animation_data_clear()

    gr.clear_normal_map()
    gr.change_material_blend_show_transparent(False)

    root_object, bbox_size, scale, mesh_offset = gr.normalize_scene(1.0, mesh_objects)
    bpy.context.view_layer.update()
    root_object.rotation_euler[2] = math.radians(int(os.environ.get("FORCE_ROTATION", 0)))
    bbox_center = Vector((0, 0, 0))

    bpy.ops.object.camera_add(location=(0, 0, 0))
    bpy.context.scene.camera = bpy.context.object

    default_camera_lens, default_camera_senser_width = 50, 36
    distance = default_camera_lens / default_camera_senser_width * \
        math.sqrt(bbox_size.x ** 2 + bbox_size.y ** 2 + bbox_size.z ** 2)
    gr.set_global_light(env_light=0.5)
    camera_angle_x = 2.0 * math.atan(default_camera_senser_width / 2 / default_camera_lens)
    out_data = {
        "camera_angle_x": camera_angle_x, "camera_lens": default_camera_lens,
        "sensor_width": default_camera_senser_width, "env_texture": "null",
        "bbox_size": list(bbox_size), "scaling_factor": scale, "translation": list(mesh_offset),
        "transforms": [],
    }
    parent_matrix = gr.rotation_matrix(0, 0, 0, 0)
    for camera_location in gr.get_solid_points_on_sphere(bbox_center, distance):
        rot = (bbox_center - camera_location).to_track_quat("-Z", "Y").to_euler()
        cam_matrix = gr.build_transformation_mat(camera_location, rot)
        cam_matrix = gr.listify_matrix(parent_matrix) @ cam_matrix
        params = {"camera_type": "PERSP", "camera_lens": default_camera_lens,
                  "camera_sensor_width": default_camera_senser_width}
        gr.add_camera_pose(cam_matrix, params)
        out_data["transforms"].append(gr.listify_matrix(cam_matrix))

    gr.set_color_output(output_dir=output_path)
    gr.render()

    gr.change_material_blend_mode()
    gr.set_color_output(output_dir=output_path, file_prefix="render_opaque_")
    gr.enable_depth_output(output_dir=output_path)
    gr.enable_normals_output(output_dir=output_path)
    gr.render()

    with open(os.path.join(output_path, gr.META_FILENAME), "w") as f:
        json.dump(out_data, f, indent=4)
    for file in glob.glob(os.path.join(output_path, "render_opaque_*")):
        os.remove(file)
    return True


def complete(out_dir: str) -> bool:
    need = [f"color_{i:04d}.webp" for i in range(12)] + [f"depth_{i:04d}.exr" for i in range(12)] + \
           [f"normal_{i:04d}.webp" for i in range(12)] + ["meta.json", "mesh.glb"]
    return all(os.path.exists(os.path.join(out_dir, n)) for n in need)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", default=None, help="pv dataset root; objects are <root>/<id>/input.glb")
    ap.add_argument("--objects_file", default=None)
    ap.add_argument("--objects", nargs="*", default=None)
    ap.add_argument("--glb", nargs="*", default=None, help="Stand-alone GLB files (external assets)")
    ap.add_argument("--name", nargs="*", default=None, help="Output names for --glb (default: file stem)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine", choices=["auto", "EEVEE", "CYCLES"], default="auto")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    jobs: list[tuple[str, str]] = []
    if args.dataset_root:
        import common
        for oid in common.object_names(args.dataset_root, args.objects, args.objects_file):
            jobs.append((oid, os.path.join(args.dataset_root, oid, "input.glb")))
    for i, g in enumerate(args.glb or []):
        name = (args.name or [])[i] if args.name and i < len(args.name) else os.path.splitext(os.path.basename(g))[0]
        jobs.append((name, g))

    engine = args.engine
    for name, glb in jobs:
        out_dir = os.path.join(args.out, name)
        if complete(out_dir) and not args.force:
            print(f"[skip] {name}", flush=True)
            continue
        t0 = time.time()
        tried = ["EEVEE", "CYCLES"] if engine == "auto" else [engine]
        ok = False
        for eng in tried:
            try:
                ok = process(glb, "glb", out_dir, eng)
                if ok and complete(out_dir):
                    engine = eng if engine == "auto" else engine  # keep the engine that worked
                    break
                ok = False
            except Exception as e:  # EEVEE without a GL context raises here
                print(f"[{name}] {eng} failed: {type(e).__name__}: {e}", flush=True)
                ok = False
        print(f"[{'ok' if ok else 'FAIL'}] {name} engine={eng} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
