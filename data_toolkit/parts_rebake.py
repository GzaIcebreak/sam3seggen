"""Split a SegviGen result into parts and bake the source model's texture onto them.

SegviGen encodes the segmentation as flat colours in the output texture, so parts are
recovered by clustering per-face base colour. Each part then gets a fresh UV layout from
Blender's smart project, and the original model's base colour is baked onto it with a
selected-to-active bake, which restores the real material instead of the part colour.

All parts are exported together as a single glb (one node/mesh/texture per part) rather
than one glb per part, so the split result stays a single file to move around; `parts.json`
lists each part's node name, label, colour, face count and area for downstream lookup.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import trimesh

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def load_single_mesh(path):
    scene = trimesh.load(path, force="scene")
    meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh) and len(g.faces)]
    if not meshes:
        raise SystemExit(f"no mesh found in {path}")
    return meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)


def face_base_colors(mesh):
    """Base colour per face, read at the face's UV centroid."""
    visual = mesh.visual
    if not isinstance(visual, trimesh.visual.TextureVisuals) or visual.uv is None:
        raise SystemExit("segmentation glb has no UV texture to read part colours from")
    image = visual.material.baseColorTexture
    if image is None:
        raise SystemExit("segmentation glb has no baseColorTexture")
    uv = np.asarray(visual.uv, dtype=np.float64)
    centroids = uv[mesh.faces].mean(axis=1)
    return np.asarray(trimesh.visual.color.uv_to_color(centroids, image))[:, :3].astype(np.int16)


def palette_from_legend(path):
    """Part colours the 2D map asked for, so faces can be matched instead of clustered."""
    with open(path, "r", encoding="utf-8") as f:
        legend = json.load(f)
    colors = np.array([entry["color"] for entry in legend], dtype=np.float64)
    return colors, [entry["prompt"] for entry in legend]


def assign_to_palette(colors, centers):
    return np.argmin(np.linalg.norm(colors[:, None, :] - centers[None], axis=2), axis=1)


def cluster_parts(colors, areas, color_tol):
    """Greedy colour clustering, seeded by surface area so speckle cannot create parts."""
    quantized = (colors // 8).astype(np.int32)
    keys, inverse = np.unique(quantized, axis=0, return_inverse=True)
    weights = np.bincount(inverse, weights=areas, minlength=len(keys))

    centers = []
    for index in np.argsort(-weights):
        candidate = colors[inverse == index].mean(axis=0)
        if all(np.linalg.norm(candidate - c) > color_tol for c in centers):
            centers.append(candidate)
    centers = np.asarray(centers, dtype=np.float64)
    labels = np.argmin(np.linalg.norm(colors[:, None, :] - centers[None], axis=2), axis=1)
    return labels, centers


def smooth_labels(mesh, labels, n_labels, iterations, self_weight=2):
    """Majority vote over face neighbours.

    Texture filtering and to_glb's seam inpainting blend colours where two parts meet,
    so the thin bands along those seams cluster as colours of their own. They lose the
    vote to the solid regions on either side.
    """
    adjacency = np.asarray(mesh.face_adjacency)
    if len(adjacency) == 0:
        return labels
    left, right = adjacency[:, 0], adjacency[:, 1]
    rows = np.arange(len(labels))
    for _ in range(iterations):
        votes = np.zeros((len(labels), n_labels), dtype=np.int32)
        np.add.at(votes, (left, labels[right]), 1)
        np.add.at(votes, (right, labels[left]), 1)
        votes[rows, labels] += self_weight
        labels = votes.argmax(axis=1)
    return labels


def drop_small_parts(labels, centers, areas, min_area_ratio, names=None):
    """Fold parts below the area threshold into the nearest surviving colour."""
    total_area = float(areas.sum())
    present = [i for i in range(len(centers)) if (labels == i).any()]
    keep = [i for i in present if areas[labels == i].sum() >= min_area_ratio * total_area]
    if not keep:
        keep = [max(present, key=lambda i: areas[labels == i].sum())]

    kept_centers = centers[keep]
    remap = np.argmin(np.linalg.norm(centers[:, None, :] - kept_centers[None], axis=2), axis=1)
    kept_names = [names[i] for i in keep] if names is not None else None
    return remap[labels], kept_centers, kept_names


def part_geometries(mesh, labels, centers, names=None):
    parts = []
    for label in range(len(centers)):
        selection = labels == label
        if not selection.any():
            continue
        faces = mesh.faces[selection]
        used, remapped = np.unique(faces, return_inverse=True)
        parts.append({
            "label": int(label),
            "name": names[label] if names is not None else None,
            "part_color": [int(round(c)) for c in centers[label]],
            "vertices": np.asarray(mesh.vertices)[used],
            "faces": remapped.reshape(faces.shape),
            "area": float(mesh.area_faces[selection].sum()),
        })
    return parts


def _reset_bpy_scene():
    import bpy

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.images):
        for item in list(block):
            block.remove(item, do_unlink=True)


def _setup_cycles_bake(samples, cage_extrusion, max_ray_distance, margin=2):
    import bpy

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = False
    try:
        scene.cycles.device = "GPU"
        bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
        bpy.context.preferences.addons["cycles"].preferences.get_devices()
    except Exception:
        scene.cycles.device = "CPU"
    scene.render.bake.use_selected_to_active = True
    scene.render.bake.use_pass_direct = False
    scene.render.bake.use_pass_indirect = False
    scene.render.bake.use_pass_color = True
    scene.render.bake.cage_extrusion = cage_extrusion
    scene.render.bake.max_ray_distance = max_ray_distance
    # Smart Project packs many small islands close together (see uv_margin); Blender's
    # default 16px bake margin is an "extend" fill that then bleeds each island's edge
    # colour across into its neighbours, showing up as confetti-like noise once islands
    # are this small. A couple of pixels is enough to hide seams without cross-talk.
    scene.render.bake.margin = margin
    scene.render.bake.margin_type = "EXTEND"


def _import_aligned_source(source_glb):
    """Bring the original glb into the same frame as a to_glb / process_glb_to_vxz mesh."""
    import bpy
    from mathutils import Matrix, Vector

    bpy.ops.import_scene.gltf(filepath=os.path.abspath(source_glb))
    source_objects = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not source_objects:
        raise SystemExit(f"no mesh imported from {source_glb}")

    # to_glb rotates the voxelised model about X, and the glTF importer applies the same
    # kind of rotation to both files, so undoing it here puts the source model back into
    # the segmentation's frame. The normalisation mirrors process_glb_to_vxz.
    align = Matrix.Rotation(math.radians(-90), 4, "X")
    for obj in source_objects:
        obj.matrix_world = align @ obj.matrix_world
    bpy.context.view_layer.update()

    corners = np.array([
        np.array(obj.matrix_world @ Vector(corner)) for obj in source_objects for corner in obj.bound_box
    ])
    center = (corners.min(axis=0) + corners.max(axis=0)) / 2
    scale = 0.99999 / float((corners.max(axis=0) - corners.min(axis=0)).max())
    normalize = Matrix.Diagonal((scale, scale, scale, 1.0)) @ Matrix.Translation(-center)
    for obj in source_objects:
        obj.matrix_world = normalize @ obj.matrix_world
    bpy.context.view_layer.update()
    return source_objects


def _undo_to_glb_rotation(objs):
    """Rotate baked parts back into the source model's own upright orientation.

    to_glb (o_voxel.postprocess) bakes an axis swap into every vertex it emits, so the
    segmentation glb's "up" ends up on a different glTF axis than the original model's
    (see _import_aligned_source, which applies the matching -90 deg X align to bring the
    source *into* that rotated frame for baking). Once the bake is done there is no reason
    to keep that rotation: undo it here so the exported glb stands upright the same way
    the source model and its transforms.json camera do.
    """
    import bpy
    from mathutils import Matrix

    undo = Matrix.Rotation(math.radians(90), 4, "X")
    for obj in objs:
        obj.matrix_world = undo @ obj.matrix_world
    bpy.context.view_layer.update()


def _gltf_to_blender(vertices):
    vertices = np.asarray(vertices, dtype=np.float64)
    return np.stack([vertices[:, 0], -vertices[:, 2], vertices[:, 1]], axis=1)


def _make_part_object(name, vertices, faces):
    import bpy

    mesh_data = bpy.data.meshes.new(name)
    mesh_data.from_pydata(_gltf_to_blender(vertices).tolist(), [], np.asarray(faces).tolist())
    mesh_data.update()
    obj = bpy.data.objects.new(name, mesh_data)
    bpy.context.collection.objects.link(obj)
    return obj


def _smart_project(obj, uv_angle_limit, uv_margin):
    import bpy

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.shade_smooth()
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(uv_angle_limit), island_margin=uv_margin)
    bpy.ops.object.mode_set(mode="OBJECT")


def _assign_bake_image(obj, name, texture_size):
    import bpy

    image = bpy.data.images.new(f"{name}_basecolor", texture_size, texture_size)
    material = bpy.data.materials.new(f"{name}_mat")
    material.use_nodes = True
    tex_node = material.node_tree.nodes.new("ShaderNodeTexImage")
    tex_node.image = image
    material.node_tree.nodes.active = tex_node
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled is not None:
        material.node_tree.links.new(tex_node.outputs["Color"], principled.inputs["Base Color"])
    obj.data.materials.append(material)
    return image


def _emit_from_base_color(source_objects):
    """Metallic gold and similar PBR leaves Diffuse almost black; bake the albedo via Emission."""
    import bpy

    originals = []
    for obj in source_objects:
        for slot in obj.material_slots:
            mat = slot.material
            if mat is None or not mat.use_nodes:
                continue
            tree = mat.node_tree
            output = next((n for n in tree.nodes if n.type == "OUTPUT_MATERIAL"), None)
            principled = next((n for n in tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
            if output is None or principled is None:
                continue
            emit = tree.nodes.new("ShaderNodeEmission")
            color_links = [l for l in tree.links if l.to_node == principled and l.to_socket.name == "Base Color"]
            if color_links:
                tree.links.new(color_links[0].from_socket, emit.inputs["Color"])
            else:
                emit.inputs["Color"].default_value = principled.inputs["Base Color"].default_value
            surface = next((l for l in tree.links if l.to_node == output and l.to_socket.name == "Surface"), None)
            originals.append((tree, output, surface.from_socket if surface else None, emit))
            if surface is not None:
                tree.links.remove(surface)
            tree.links.new(emit.outputs["Emission"], output.inputs["Surface"])
    return originals


def _restore_surface_links(originals):
    for tree, output, from_socket, emit in originals:
        existing = next((l for l in tree.links if l.to_node == output and l.to_socket.name == "Surface"), None)
        if existing is not None:
            tree.links.remove(existing)
        if from_socket is not None:
            tree.links.new(from_socket, output.inputs["Surface"])
        tree.nodes.remove(emit)


def _bake_selected_to_active(source_objects, dest_obj):
    import bpy

    bpy.ops.object.select_all(action="DESELECT")
    for obj in source_objects:
        if obj != dest_obj:
            obj.select_set(True)
    dest_obj.select_set(True)
    bpy.context.view_layer.objects.active = dest_obj
    originals = _emit_from_base_color(source_objects)
    try:
        bpy.ops.object.bake(type="EMIT")
    finally:
        _restore_surface_links(originals)


def blender_reuv_and_bake(mesh, source_glb, texture_size=2048, uv_angle_limit=66.0, uv_margin=0.003,
                          cage_extrusion=0.02, max_ray_distance=0.05, samples=16, margin=2):
    """Re-unwrap a cleaned mesh and bake the original model's albedo onto the new UVs.

    Call this after drop_offbody_components: the leftover xatlas packing is from the
    pre-cleanup mesh, and the SegviGen colours are only a part label. A fresh Smart
    Project plus a selected-to-active bake restores the source texture.
    """
    import tempfile
    import bpy

    source_glb = os.path.abspath(source_glb)
    _setup_cycles_bake(samples, cage_extrusion, max_ray_distance, margin)
    _reset_bpy_scene()
    source_objects = _import_aligned_source(source_glb)

    dest = _make_part_object("rebake", mesh.vertices, mesh.faces)
    _smart_project(dest, uv_angle_limit, uv_margin)
    image = _assign_bake_image(dest, "rebake", texture_size)
    _bake_selected_to_active(source_objects, dest)
    image.pack()
    _undo_to_glb_rotation([dest])

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "rebake.glb")
        bpy.ops.object.select_all(action="DESELECT")
        dest.select_set(True)
        bpy.context.view_layer.objects.active = dest
        # load_single_mesh reads the raw mesh buffer straight from scene.geometry, which
        # skips any per-node transform glTF export would otherwise store separately, so
        # the undo-rotation above has to be baked into the vertices themselves here.
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        bpy.ops.export_scene.gltf(filepath=out, export_format="GLB", use_selection=True)
        baked = load_single_mesh(out)
    print(f"Blender re-UV + bake: {len(mesh.faces)} faces, {texture_size}px from {source_glb}")
    return baked


def bake_parts(parts, source_glb, out_dir, texture_size, uv_angle_limit, uv_margin,
               cage_extrusion, max_ray_distance, samples, margin=2, combined_name="parts.glb",
               save_textures=False):
    """Re-UV and bake every part, then export them all as one glb (one node per part).

    Each part keeps its own mesh/material/UV/texture, so downstream tools can still tell
    them apart by node name, but the whole assembly loads and moves as a single file
    instead of one glb per part.
    """
    import bpy

    _setup_cycles_bake(samples, cage_extrusion, max_ray_distance, margin)
    _reset_bpy_scene()
    source_objects = _import_aligned_source(source_glb)

    # A selected-to-active bake only finds the source surface if the two overlap, so make
    # the assumed alignment visible rather than silently baking a blank texture.
    from mathutils import Vector

    source_bounds = np.array([
        np.array(obj.matrix_world @ Vector(corner)) for obj in source_objects for corner in obj.bound_box
    ])
    part_points = np.concatenate([np.asarray(p["vertices"]) for p in parts])
    part_bounds = np.stack([part_points[:, 0], -part_points[:, 2], part_points[:, 1]], axis=1)
    print(f"aligned source bounds {source_bounds.min(axis=0).round(3)} .. {source_bounds.max(axis=0).round(3)}")
    print(f"parts bounds          {part_bounds.min(axis=0).round(3)} .. {part_bounds.max(axis=0).round(3)}")

    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    part_objects = []
    for part in parts:
        suffix = "".join(c if c.isalnum() else "_" for c in part["name"]) if part["name"] else ""
        name = f"part_{part['label']:02d}" + (f"_{suffix}" if suffix else "")
        part_obj = _make_part_object(name, part["vertices"], part["faces"])
        _smart_project(part_obj, uv_angle_limit, uv_margin)
        image = _assign_bake_image(part_obj, name, texture_size)
        _bake_selected_to_active(source_objects, part_obj)
        image.pack()
        if save_textures:
            texture_path = os.path.join(out_dir, f"{name}_basecolor.png")
            image.filepath_raw = texture_path
            image.file_format = "PNG"
            image.save()

        part_objects.append(part_obj)
        print(f"  {name}: {len(part['faces'])} faces baked")
        manifest.append({
            "label": part["label"],
            "name": part["name"],
            "node": name,
            "part_color": part["part_color"],
            "faces": int(len(part["faces"])),
            "area": part["area"],
        })

    _undo_to_glb_rotation(part_objects)

    combined_path = os.path.join(out_dir, combined_name)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in part_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = part_objects[0]
    bpy.ops.export_scene.gltf(filepath=combined_path, export_format="GLB", use_selection=True)
    print(f"combined {len(part_objects)} parts -> {combined_path}")

    with open(os.path.join(out_dir, "parts.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def export_parts(seg_glb, source_glb, out_dir, palette=None, texture_size=2048, color_tol=40.0,
                 min_area_ratio=0.01, smooth_iterations=3, uv_angle_limit=66.0, uv_margin=0.003,
                 cage_extrusion=0.02, max_ray_distance=0.05, samples=16, margin=2,
                 combined_name="parts.glb", save_textures=False):
    seg_glb = os.path.abspath(seg_glb)
    source_glb = os.path.abspath(source_glb)
    out_dir = os.path.abspath(out_dir)

    mesh = load_single_mesh(seg_glb)
    areas = np.asarray(mesh.area_faces)
    colors = face_base_colors(mesh)
    if palette:
        centers, names = palette_from_legend(os.path.abspath(palette))
        labels = assign_to_palette(colors, centers)
    else:
        labels, centers = cluster_parts(colors, areas, color_tol)
        names = None
    labels = smooth_labels(mesh, labels, len(centers), smooth_iterations)
    labels, centers, names = drop_small_parts(labels, centers, areas, min_area_ratio, names)
    parts = part_geometries(mesh, labels, centers, names)

    print(f"{len(parts)} parts from {len(mesh.faces)} faces")
    for part in parts:
        label = part["name"] or "?"
        print(f"  part_{part['label']:02d} {label} colour={part['part_color']} faces={len(part['faces'])}")

    return bake_parts(parts, source_glb, out_dir, texture_size, uv_angle_limit, uv_margin,
                      cage_extrusion, max_ray_distance, samples, margin,
                      combined_name=combined_name, save_textures=save_textures)


def main():
    parser = argparse.ArgumentParser(description="Split SegviGen output into parts and rebake source texture")
    parser.add_argument("--seg_glb", required=True, help="SegviGen output glb (part colours)")
    parser.add_argument("--source_glb", required=True, help="Original model whose texture is baked back")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--palette", default=None,
                        help="SAM3 *_legend.json; match faces to its colours instead of clustering")
    parser.add_argument("--texture_size", type=int, default=2048)
    parser.add_argument("--color_tol", type=float, default=40.0, help="RGB distance separating two parts")
    parser.add_argument("--min_area_ratio", type=float, default=0.01, help="Discard parts below this area share")
    parser.add_argument("--smooth_iterations", type=int, default=3, help="Neighbour vote passes over seam bands")
    parser.add_argument("--uv_angle_limit", type=float, default=66.0)
    parser.add_argument("--uv_margin", type=float, default=0.003)
    parser.add_argument("--cage_extrusion", type=float, default=0.02)
    parser.add_argument("--max_ray_distance", type=float, default=0.05)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--margin", type=int, default=2,
                        help="Bake edge-extend margin in px; keep small since Smart Project "
                             "packs many tiny islands close together (large margins bleed "
                             "between neighbours and show up as speckle noise).")
    parser.add_argument(
        "--blender_reuv",
        action="store_true",
        help="Skip part split: Smart Project the whole mesh and bake the source albedo back.",
    )
    parser.add_argument("--combined_name", default="parts.glb",
                        help="Filename of the single glb holding every part (one node each).")
    parser.add_argument("--save_textures", action="store_true",
                        help="Also dump each part's baked texture as a standalone PNG "
                             "(they're always packed inside the combined glb regardless).")
    args = parser.parse_args()

    if args.blender_reuv:
        os.makedirs(os.path.abspath(args.out_dir), exist_ok=True)
        baked = blender_reuv_and_bake(
            load_single_mesh(os.path.abspath(args.seg_glb)),
            os.path.abspath(args.source_glb),
            texture_size=args.texture_size,
            uv_angle_limit=args.uv_angle_limit,
            uv_margin=args.uv_margin,
            cage_extrusion=args.cage_extrusion,
            max_ray_distance=args.max_ray_distance,
            samples=args.samples,
            margin=args.margin,
        )
        out = os.path.join(os.path.abspath(args.out_dir), "rebake.glb")
        baked.export(out)
        print(f"saved {out}")
        return

    export_parts(
        args.seg_glb, args.source_glb, args.out_dir,
        palette=args.palette,
        texture_size=args.texture_size,
        color_tol=args.color_tol,
        min_area_ratio=args.min_area_ratio,
        smooth_iterations=args.smooth_iterations,
        uv_angle_limit=args.uv_angle_limit,
        uv_margin=args.uv_margin,
        cage_extrusion=args.cage_extrusion,
        max_ray_distance=args.max_ray_distance,
        samples=args.samples,
        margin=args.margin,
        combined_name=args.combined_name,
        save_textures=args.save_textures,
    )


if __name__ == "__main__":
    main()
