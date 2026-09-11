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

from data_toolkit.project_2d import face_map_labels, label_mesh_from_map


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
    """Concept colours the 2D map asked for, plus the part each concept belongs to."""
    with open(path, "r", encoding="utf-8") as f:
        legend = json.load(f)
    colors = np.array([entry["color"] for entry in legend], dtype=np.float64)
    concepts = [entry["prompt"] for entry in legend]
    parts = [entry.get("part", entry["prompt"]) for entry in legend]
    return colors, concepts, parts


def merge_labels_by_part(labels, part_names_per_label):
    """Collapse several concept ids that share an output part name into one label."""
    unique = list(dict.fromkeys(part_names_per_label))
    index = {name: i for i, name in enumerate(unique)}
    remap = np.array([index[name] for name in part_names_per_label], dtype=np.int64)
    return remap[np.asarray(labels)], unique


def merge_centers_by_part(centers, part_names_per_label, unique_parts):
    merged = []
    for part in unique_parts:
        first = next(i for i, name in enumerate(part_names_per_label) if name == part)
        merged.append(centers[first])
    return np.asarray(merged, dtype=np.float64)


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


def welded_face_adjacency(mesh):
    """Face adjacency across real geometric edges, ignoring UV seams.

    glTF stores a separate vertex per UV corner, so mesh.face_adjacency only sees the
    fraction of edges whose two faces happen to share vertex indices -- on a SegviGen
    output that leaves every part shattered into hundreds of "components", which makes
    both the neighbour vote and the island cleanup below nearly no-ops. Re-index the faces
    by vertex position first so neighbours are actually recognised. The mesh itself is left
    alone: welding it would collapse the UVs the bake needs.
    """
    _, inverse = np.unique(np.asarray(mesh.vertices).round(6), axis=0, return_inverse=True)
    faces = inverse.reshape(-1)[np.asarray(mesh.faces)]
    # Pair faces through shared welded edges by hand: face indices must keep lining up
    # with labels/areas, and a face that welding degenerates to a sliver simply ends up
    # with fewer (or no) neighbours instead of being dropped.
    corners = np.stack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=1).reshape(-1, 2)
    owner = np.repeat(np.arange(len(faces)), 3)
    keep = corners[:, 0] != corners[:, 1]
    corners, owner = np.sort(corners[keep], axis=1), owner[keep]
    order = np.lexsort((corners[:, 1], corners[:, 0]))
    corners, owner = corners[order], owner[order]
    new_edge = np.ones(len(corners), dtype=bool)
    new_edge[1:] = np.any(corners[1:] != corners[:-1], axis=1)
    edge_id = np.cumsum(new_edge) - 1
    counts = np.bincount(edge_id)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    pairs = []
    two = counts == 2
    pairs.append(np.stack([owner[starts[two]], owner[starts[two] + 1]], axis=1))
    for e in np.nonzero(counts > 2)[0]:
        members = owner[starts[e]:starts[e] + counts[e]]
        ii, jj = np.triu_indices(len(members), k=1)
        pairs.append(np.stack([members[ii], members[jj]], axis=1))
    adjacency = np.concatenate(pairs, axis=0) if pairs else np.zeros((0, 2), dtype=np.int64)
    return adjacency[adjacency[:, 0] != adjacency[:, 1]].astype(np.int64)


def smooth_labels(adjacency, labels, n_labels, iterations, self_weight=2):
    """Majority vote over face neighbours.

    Texture filtering and to_glb's seam inpainting blend colours where two parts meet,
    so the thin bands along those seams cluster as colours of their own. They lose the
    vote to the solid regions on either side.
    """
    adjacency = np.asarray(adjacency)
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


def reassign_label_islands(adjacency, labels, areas, min_island_ratio=0.2):
    """Hand a part's small detached patches over to the part surrounding them.

    Neighbour voting in smooth_labels only cleans up seam-width noise; a whole boot that
    got the staff's colour survives it, because inside that patch every neighbour agrees.
    Such a patch is always disconnected from the rest of its part, so compare each
    connected patch against the largest patch of the same part and give away the small
    ones, which mostly ends the "one stray limb in the wrong part" failure.
    """
    from trimesh.graph import connected_components

    adjacency = np.asarray(adjacency)
    if len(adjacency) == 0 or min_island_ratio <= 0:
        return labels

    labels = labels.copy()
    same_part = labels[adjacency[:, 0]] == labels[adjacency[:, 1]]
    patches = connected_components(adjacency[same_part], nodes=np.arange(len(labels)))
    by_part = {}
    for patch in patches:
        by_part.setdefault(labels[patch[0]], []).append(patch)

    for part, part_patches in by_part.items():
        biggest = max(areas[patch].sum() for patch in part_patches)
        for patch in part_patches:
            if areas[patch].sum() >= min_island_ratio * biggest:
                continue
            member = np.zeros(len(labels), dtype=bool)
            member[patch] = True
            # Faces across the patch border, weighted by area so a long thin contact with
            # one part cannot outvote a broad one.
            crossing = adjacency[member[adjacency[:, 0]] != member[adjacency[:, 1]]]
            outside = np.where(member[crossing[:, 0]], crossing[:, 1], crossing[:, 0])
            if len(outside) == 0:
                continue
            weights = np.bincount(labels[outside], weights=areas[outside])
            winner = int(weights.argmax())
            if winner != part:
                labels[patch] = winner
    return labels


def absorb_small_fragments(adjacency, labels, min_faces=100, iterations=3):
    """Hand same-label patches under `min_faces` faces to their majority neighbour.

    This is split.py's topology fix and nothing more: a blob big enough to be a real
    part is never touched, whatever it is connected to, so the split follows SegviGen's
    colouring instead of second-guessing it. Small islands that texture filtering or
    seam inpainting produced are the only thing that moves.
    """
    from trimesh.graph import connected_components

    adjacency = np.asarray(adjacency)
    if len(adjacency) == 0 or min_faces <= 0:
        return labels

    labels = labels.copy()
    for _ in range(iterations):
        same = labels[adjacency[:, 0]] == labels[adjacency[:, 1]]
        patches = connected_components(adjacency[same], nodes=np.arange(len(labels)))
        small = [p for p in patches if len(p) < min_faces]
        if not small:
            break
        changed = False
        for patch in small:
            member = np.zeros(len(labels), dtype=bool)
            member[patch] = True
            crossing = adjacency[member[adjacency[:, 0]] != member[adjacency[:, 1]]]
            outside = np.where(member[crossing[:, 0]], crossing[:, 1], crossing[:, 0])
            if len(outside) == 0:
                continue
            winner = int(np.bincount(labels[outside]).argmax())
            if winner != labels[patch[0]]:
                labels[patch] = winner
                changed = True
        if not changed:
            break
    return labels


def weld_pieces_to_map(adjacency, labels, seen, min_visible_share=0.25, min_visible_faces=20,
                       min_agreement=0.6):
    """Rename whole same-label pieces after the guide map, never cutting inside one.

    `seen` is the palette index the 2D map paints on each face (-1 when hidden or on
    background). A piece changes label only when enough of it faces the camera and the
    visible faces clearly agree on another label; hidden pieces and split votes keep
    SegviGen's colour. This is the "weld" between the stain split (which trusts every
    colour) and the refine pipeline (which rewrites faces pixel by pixel and fragments them).

    Returns (labels, moved_faces, moved_pieces).
    """
    from trimesh.graph import connected_components

    adjacency = np.asarray(adjacency)
    labels = labels.copy()
    seen = np.asarray(seen)
    if len(adjacency) == 0 or not (seen >= 0).any():
        return labels, 0, 0

    same = labels[adjacency[:, 0]] == labels[adjacency[:, 1]]
    pieces = connected_components(adjacency[same], nodes=np.arange(len(labels)))
    moved_faces = moved_pieces = 0
    for piece in pieces:
        votes = seen[piece]
        votes = votes[votes >= 0]
        if len(votes) < min_visible_faces or len(votes) < min_visible_share * len(piece):
            continue
        counts = np.bincount(votes)
        winner = int(counts.argmax())
        if winner == labels[piece[0]] or counts[winner] < min_agreement * len(votes):
            continue
        labels[piece] = winner
        moved_faces += len(piece)
        moved_pieces += 1
    return labels, moved_faces, moved_pieces


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


def _undo_to_glb_rotation_np(points):
    """Same fix as _undo_to_glb_rotation, without going through Blender.

    to_glb bakes a -90 deg X rotation into every vertex it emits (see the comment in
    _import_aligned_source); the no-bake split path never touches Blender, so undo it
    directly on the raw glTF-space vertices instead.
    """
    points = np.asarray(points, dtype=np.float64)
    return np.stack([points[:, 0], -points[:, 2], points[:, 1]], axis=1)


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


def _import_source(source_glb, align_to_seg=True):
    """Import the original glb. `align_to_seg` puts it in the to_glb / remesh frame.

    X-Part writes solids in the source model's own frame, so baking those must skip the
    align -- otherwise the cage looks at a unit-cube, rotated copy of the source and
    the generated parts miss it entirely.
    """
    import bpy
    from mathutils import Matrix, Vector

    bpy.ops.import_scene.gltf(filepath=os.path.abspath(source_glb))
    source_objects = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not source_objects:
        raise SystemExit(f"no mesh imported from {source_glb}")
    if not align_to_seg:
        return source_objects

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


def _import_aligned_source(source_glb):
    return _import_source(source_glb, align_to_seg=True)


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
               save_textures=False, source_frame=False):
    """Re-UV and bake every part, then export them all as one glb (one node per part).

    Each part keeps its own mesh/material/UV/texture, so downstream tools can still tell
    them apart by node name, but the whole assembly loads and moves as a single file
    instead of one glb per part.
    """
    import bpy

    _setup_cycles_bake(samples, cage_extrusion, max_ray_distance, margin)
    _reset_bpy_scene()
    source_objects = _import_source(source_glb, align_to_seg=not source_frame)

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

    if not source_frame:
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


LABEL_COLORS = np.array(
    [
        [220, 40, 40], [40, 90, 230], [30, 180, 70], [240, 210, 30],
        [40, 200, 210], [230, 70, 180], [140, 50, 200], [240, 130, 30],
        [20, 120, 120], [180, 180, 40], [80, 40, 160], [40, 160, 40],
    ],
    dtype=np.float64,
)


def load_face_labels(labels_path, names_path, n_faces):
    """Per-face part labels decided elsewhere, e.g. by lifting multi-view SAM3 masks.

    Colours here are only a display placeholder: the labels already say which part each
    face belongs to, so nothing downstream has to recover that from a colour again.
    """
    labels = np.load(os.path.abspath(labels_path))
    if len(labels) != n_faces:
        raise ValueError(f"labels cover {len(labels)} faces but the mesh has {n_faces}")
    with open(os.path.abspath(names_path), "r", encoding="utf-8") as handle:
        names = json.load(handle)
    if labels.max(initial=-1) >= len(names):
        raise ValueError(f"label {int(labels.max())} has no name in {names_path}")
    centers = LABEL_COLORS[np.arange(len(names)) % len(LABEL_COLORS)]
    return labels.astype(np.int64), list(names), centers


SPLIT_MODES = ("stain", "weld", "refine")


def _face_labels(mesh, palette=None, color_tol=40.0, min_area_ratio=0.01,
                 smooth_iterations=3, min_island_ratio=0.2,
                 two_d_map=None, transforms=None, azimuth=0.0,
                 labels_npy=None, label_names=None,
                 split_mode="stain", min_fragment_faces=100,
                 weld_min_visible=0.25, weld_min_agreement=0.6):
    """One part label per face, plus the part colours and names: (labels, centers, names).

    split_mode "stain" trusts SegviGen's colouring: nearest palette colour per face, then
    only fragments under `min_fragment_faces` faces move (split.py's behaviour). "weld"
    keeps every cut of the stain but renames whole same-colour pieces when the 2D guide
    map clearly paints them as another part (weld_pieces_to_map). "refine" is the older
    pipeline that overwrites visible faces from the 2D map, votes seam bands and
    reassigns whole detached islands -- it repairs merged touching parts but also
    redraws boundaries SegviGen already got right.
    """
    if split_mode not in SPLIT_MODES:
        raise ValueError(f"split_mode must be one of {SPLIT_MODES}, got {split_mode!r}")
    areas = np.asarray(mesh.area_faces)
    if labels_npy:
        labels, names, centers = load_face_labels(labels_npy, label_names, len(mesh.faces))
        print(f"using {len(names)} precomputed face labels: {names}")
        return labels, centers, names

    colors = face_base_colors(mesh)
    part_names_for_concepts = None
    projected = False
    if palette:
        centers, names, part_names_for_concepts = palette_from_legend(os.path.abspath(palette))
        labels = assign_to_palette(colors, centers)
        if split_mode == "refine" and two_d_map and transforms:
            labels = label_mesh_from_map(
                mesh, labels, os.path.abspath(two_d_map), centers,
                os.path.abspath(transforms), azimuth,
            )
            projected = True
            print(f"overwrote visible-face labels from {two_d_map}")
    else:
        labels, centers = cluster_parts(colors, areas, color_tol)
        names = None
    adjacency = welded_face_adjacency(mesh)
    if split_mode in ("stain", "weld"):
        if split_mode == "weld" and palette and two_d_map and transforms:
            seen = face_map_labels(
                mesh, os.path.abspath(two_d_map), centers,
                os.path.abspath(transforms), azimuth,
            )
            labels, moved_faces, moved_pieces = weld_pieces_to_map(
                adjacency, labels, seen, min_visible_share=weld_min_visible,
                min_agreement=weld_min_agreement,
            )
            print(f"weld: {moved_pieces} pieces / {moved_faces} faces renamed after {two_d_map}")
        elif split_mode == "weld":
            print("weld: no palette/2D map/transforms given, falling back to the stain split")
        before = labels.copy()
        labels = absorb_small_fragments(adjacency, labels, min_fragment_faces)
        print(f"{split_mode} split: absorbed {int((before != labels).sum())} fragment faces "
              f"(<{min_fragment_faces} faces), boundaries otherwise as coloured")
    else:
        if not projected:
            labels = smooth_labels(adjacency, labels, len(centers), smooth_iterations)
        labels = reassign_label_islands(adjacency, labels, areas, min_island_ratio)
    if part_names_for_concepts is not None:
        labels, names = merge_labels_by_part(labels, part_names_for_concepts)
        centers = merge_centers_by_part(centers, part_names_for_concepts, names)
    else:
        labels, centers, names = drop_small_parts(labels, centers, areas, min_area_ratio, names)
    return labels, centers, names


def _split_labels(mesh, **kwargs):
    """part_geometries of _face_labels; see there for the options."""
    labels, centers, names = _face_labels(mesh, **kwargs)
    return part_geometries(mesh, labels, centers, names)


def export_parts(seg_glb, source_glb, out_dir, palette=None, texture_size=2048, color_tol=40.0,
                 min_area_ratio=0.01, smooth_iterations=3, uv_angle_limit=66.0, uv_margin=0.003,
                 cage_extrusion=0.02, max_ray_distance=0.05, samples=16, margin=2,
                 combined_name="parts.glb", save_textures=False, min_island_ratio=0.2,
                 two_d_map=None, transforms=None, azimuth=0.0,
                 labels_npy=None, label_names=None,
                 split_mode="stain", min_fragment_faces=100,
                 weld_min_visible=0.25, weld_min_agreement=0.6):
    seg_glb = os.path.abspath(seg_glb)
    source_glb = os.path.abspath(source_glb)
    out_dir = os.path.abspath(out_dir)

    mesh = load_single_mesh(seg_glb)
    parts = _split_labels(
        mesh, palette=palette, color_tol=color_tol, min_area_ratio=min_area_ratio,
        smooth_iterations=smooth_iterations, min_island_ratio=min_island_ratio,
        two_d_map=two_d_map, transforms=transforms, azimuth=azimuth,
        labels_npy=labels_npy, label_names=label_names,
        split_mode=split_mode, min_fragment_faces=min_fragment_faces,
        weld_min_visible=weld_min_visible, weld_min_agreement=weld_min_agreement,
    )

    print(f"{len(parts)} parts from {len(mesh.faces)} faces")
    for part in parts:
        label = part["name"] or "?"
        print(f"  part_{part['label']:02d} {label} colour={part['part_color']} faces={len(part['faces'])}")

    return bake_parts(parts, source_glb, out_dir, texture_size, uv_angle_limit, uv_margin,
                      cage_extrusion, max_ray_distance, samples, margin,
                      combined_name=combined_name, save_textures=save_textures)


def _short_part_name(name):
    """00_part_00_head -> head, so the bake does not prefix the prefix."""
    import re

    stripped = re.sub(r"^(?:\d+_)+", "", name)
    stripped = re.sub(r"^(?:part_\d+_)+", "", stripped)
    return stripped or name


def completed_part_geometries(completed_glb):
    """One bake-parts dict per node of a closed-parts glb already in the source frame."""
    scene = trimesh.load(completed_glb, force="scene")
    parts = []
    for label, name in enumerate(scene.geometry):
        mesh = scene.geometry[name].copy()
        if name in scene.graph.geometry_nodes:
            transform, _ = scene.graph.get(scene.graph.geometry_nodes[name][0])
            mesh.apply_transform(transform)
        parts.append({
            "label": int(label),
            "name": _short_part_name(name),
            "part_color": [int(c) for c in LABEL_COLORS[label % len(LABEL_COLORS)]],
            "vertices": np.asarray(mesh.vertices),
            "faces": np.asarray(mesh.faces),
            "area": float(mesh.area),
        })
    if not parts:
        raise SystemExit(f"no mesh found in {completed_glb}")
    return parts


def bake_completed(completed_glb, source_glb, out_dir, texture_size=2048,
                   uv_angle_limit=66.0, uv_margin=0.003, cage_extrusion=0.05,
                   max_ray_distance=0.15, samples=16, margin=2,
                   combined_name="xpart_parts.glb", save_textures=False):
    """Re-UV X-Part's closed solids and bake the source albedo onto them.

    The cage is looser than the split bake: a generated solid only approximates the
    source surface, and the default 0.02/0.05 pair leaves most of it unhit.
    """
    parts = completed_part_geometries(completed_glb)
    print(f"{len(parts)} closed solids from {os.path.basename(completed_glb)}")
    for part in parts:
        print(f"  {part['name']}: {len(part['faces'])} faces")
    return bake_parts(
        parts, source_glb, out_dir, texture_size, uv_angle_limit, uv_margin,
        cage_extrusion, max_ray_distance, samples, margin,
        combined_name=combined_name, save_textures=save_textures, source_frame=True,
    )


def export_parts_no_bake(seg_glb, out_dir, palette=None, color_tol=40.0, min_area_ratio=0.01,
                          smooth_iterations=3, combined_name="parts.glb", min_island_ratio=0.2,
                          two_d_map=None, transforms=None, azimuth=0.0,
                          labels_npy=None, label_names=None,
                          split_mode="stain", min_fragment_faces=100,
                          weld_min_visible=0.25, weld_min_agreement=0.6):
    """Split into parts without Blender: no re-UV, no texture bake, no bpy dependency.

    Each part keeps its own flat "part colour" (the same colour it was assigned in the
    2D/clustering map) as a placeholder vertex colour instead of the source model's real
    albedo. Much faster than export_parts and works in any plain Python env with trimesh,
    at the cost of not looking like the original material.
    """
    seg_glb = os.path.abspath(seg_glb)
    out_dir = os.path.abspath(out_dir)

    mesh = load_single_mesh(seg_glb)
    parts = _split_labels(
        mesh, palette=palette, color_tol=color_tol, min_area_ratio=min_area_ratio,
        smooth_iterations=smooth_iterations, min_island_ratio=min_island_ratio,
        two_d_map=two_d_map, transforms=transforms, azimuth=azimuth,
        labels_npy=labels_npy, label_names=label_names,
        split_mode=split_mode, min_fragment_faces=min_fragment_faces,
        weld_min_visible=weld_min_visible, weld_min_agreement=weld_min_agreement,
    )

    print(f"{len(parts)} parts from {len(mesh.faces)} faces (no texture bake)")
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    scene = trimesh.Scene()
    for part in parts:
        suffix = "".join(c if c.isalnum() else "_" for c in part["name"]) if part["name"] else ""
        name = f"part_{part['label']:02d}" + (f"_{suffix}" if suffix else "")
        vertices = _undo_to_glb_rotation_np(part["vertices"])
        rgba = np.array(part["part_color"] + [255], dtype=np.uint8)
        vertex_colors = np.tile(rgba, (len(vertices), 1))
        part_mesh = trimesh.Trimesh(
            vertices=vertices, faces=part["faces"], vertex_colors=vertex_colors, process=False,
        )
        scene.add_geometry(part_mesh, node_name=name, geom_name=name)
        print(f"  {name}: {len(part['faces'])} faces colour={part['part_color']}")
        manifest.append({
            "label": part["label"],
            "name": part["name"],
            "node": name,
            "part_color": part["part_color"],
            "faces": int(len(part["faces"])),
            "area": part["area"],
        })

    combined_path = os.path.join(out_dir, combined_name)
    scene.export(combined_path)
    print(f"combined {len(parts)} parts -> {combined_path}")

    with open(os.path.join(out_dir, "parts.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Split SegviGen output into parts and rebake source texture")
    parser.add_argument("--seg_glb", default=None, help="SegviGen output glb (part colours)")
    parser.add_argument("--completed", default=None,
                        help="Closed parts already in the source frame (X-Part output). "
                             "Skip the split; just re-UV and bake the source albedo onto them.")
    parser.add_argument("--source_glb", default=None,
                        help="Original model whose texture is baked back (required unless --no_bake)")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--no_bake", action="store_true",
                        help="Split into parts without Blender/bpy: no re-UV, no texture bake, "
                             "each part just keeps a flat placeholder colour. Fast, no --source_glb needed.")
    parser.add_argument("--palette", default=None,
                        help="SAM3 *_legend.json; match faces to its colours instead of clustering")
    parser.add_argument("--texture_size", type=int, default=2048)
    parser.add_argument("--color_tol", type=float, default=40.0, help="RGB distance separating two parts")
    parser.add_argument("--min_area_ratio", type=float, default=0.01, help="Discard parts below this area share")
    parser.add_argument("--smooth_iterations", type=int, default=3, help="Neighbour vote passes over seam bands")
    parser.add_argument("--split_mode", choices=list(SPLIT_MODES), default="stain",
                        help="'stain' (default) cuts exactly along SegviGen's colouring and only "
                             "absorbs fragments under --min_fragment_faces (split.py's rule). "
                             "'weld' keeps those cuts but renames whole same-colour pieces the "
                             "--two_d_map clearly paints as another part. 'refine' overwrites "
                             "visible faces from the map pixel by pixel, votes seam bands and "
                             "reassigns detached islands.")
    parser.add_argument("--min_fragment_faces", type=int, default=100,
                        help="--split_mode stain/weld: same-colour patches under this many faces go "
                             "to their majority neighbour; larger patches are never touched.")
    parser.add_argument("--weld_min_visible", type=float, default=0.25,
                        help="--split_mode weld: a piece is only renamed when at least this share of "
                             "its faces (and 20 faces) is visible in the map.")
    parser.add_argument("--weld_min_agreement", type=float, default=0.6,
                        help="--split_mode weld: ...and at least this share of those visible faces "
                             "agree on the new label.")
    parser.add_argument("--min_island_ratio", type=float, default=0.2,
                        help="--split_mode refine: give a part's detached patch to the surrounding "
                             "part when it is smaller than this share of that part's largest "
                             "patch. 0 disables.")
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
    parser.add_argument("--two_d_map", default=None,
                        help="SAM3 colour map; visible faces are labelled from it.")
    parser.add_argument("--transforms", default=None,
                        help="transforms.json used to render the conditioning view.")
    parser.add_argument("--azimuth", type=float, default=0.0,
                        help="Camera orbit used for the conditioning view, in degrees.")
    parser.add_argument("--labels", default=None,
                        help="npy of one part label per face, e.g. from data_toolkit/lift_sam3.py. "
                             "Bypasses colour clustering and the 2D map entirely.")
    parser.add_argument("--label_names", default=None,
                        help="JSON list naming each label index used by --labels.")
    args = parser.parse_args()
    if not args.no_bake and args.source_glb is None:
        parser.error("--source_glb is required unless --no_bake is set")
    if bool(args.labels) != bool(args.label_names):
        parser.error("--labels and --label_names must be given together")
    if args.completed and args.seg_glb:
        parser.error("pass either --completed or --seg_glb, not both")
    if not args.completed and not args.seg_glb and not args.blender_reuv:
        parser.error("one of --seg_glb, --completed or --blender_reuv is required")

    if args.completed:
        bake_completed(
            os.path.abspath(args.completed), os.path.abspath(args.source_glb),
            os.path.abspath(args.out_dir),
            texture_size=args.texture_size,
            uv_angle_limit=args.uv_angle_limit,
            uv_margin=args.uv_margin,
            cage_extrusion=args.cage_extrusion,
            max_ray_distance=args.max_ray_distance,
            samples=args.samples,
            margin=args.margin,
            combined_name=args.combined_name,
            save_textures=args.save_textures,
        )
        return

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

    if args.no_bake:
        export_parts_no_bake(
            args.seg_glb, args.out_dir,
            palette=args.palette,
            color_tol=args.color_tol,
            min_area_ratio=args.min_area_ratio,
            smooth_iterations=args.smooth_iterations,
            combined_name=args.combined_name,
            min_island_ratio=args.min_island_ratio,
            two_d_map=args.two_d_map,
            transforms=args.transforms,
            azimuth=args.azimuth,
            labels_npy=args.labels,
            label_names=args.label_names,
            split_mode=args.split_mode,
            min_fragment_faces=args.min_fragment_faces,
            weld_min_visible=args.weld_min_visible,
            weld_min_agreement=args.weld_min_agreement,
        )
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
        min_island_ratio=args.min_island_ratio,
        two_d_map=args.two_d_map,
        transforms=args.transforms,
        azimuth=args.azimuth,
        labels_npy=args.labels,
        label_names=args.label_names,
        split_mode=args.split_mode,
        min_fragment_faces=args.min_fragment_faces,
        weld_min_visible=args.weld_min_visible,
        weld_min_agreement=args.weld_min_agreement,
    )


if __name__ == "__main__":
    main()
    # bpy's pip package can access-violate during interpreter teardown after a Cycles
    # bake (harmless -- happens after every output file is already written), which
    # would otherwise make callers checking the exit code think this run failed.
    sys.stdout.flush()
    os._exit(0)
