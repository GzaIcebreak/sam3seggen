"""Assign semantic names to an unguided segmentation's parts by multi-view SAM3 voting.

SegviGen's unguided full segmentation produces clean but *anonymous* parts, while
SAM3 knows names but cannot touch geometry. Voting per *part* (not per face, as
lift_sam3.py does) joins the two: over a fixed view grid, each part inherits the
concept whose SAM3 masks covered most of the pixels that part owns, and parts that
land on the same concept are merged into a single output object.

The geometric atoms stay exactly what the unguided segmentation produced -- voting
only ever *merges* whole parts, never re-cuts a boundary, so a torn edge is
impossible by construction.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import trimesh

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_toolkit.lift_sam3 import (
    accumulate_votes,
    load_cameras,
    load_masks,
    part_votes_from_concepts,
    view_weights,
)
from data_toolkit.multiview import (
    gltf_to_blender,
    normalize_to_unit_cube,
    rasterize_face_ids,
)


def load_parts(parts_glb):
    """Load a combined parts glb; return (scene, concatenated mesh, owner per face, node names)."""
    scene = trimesh.load(parts_glb, force="scene")
    names = list(scene.geometry.keys())
    verts, faces, owner = [], [], []
    for index, name in enumerate(names):
        geometry = scene.geometry[name]
        transform = scene.graph.get(scene.graph.geometry_nodes[name][0])[0]
        points = trimesh.transform_points(np.asarray(geometry.vertices), transform)
        faces.append(np.asarray(geometry.faces) + sum(len(v) for v in verts))
        verts.append(points)
        owner.append(np.full(len(geometry.faces), index, dtype=np.int64))
    mesh = trimesh.Trimesh(
        vertices=np.concatenate(verts), faces=np.concatenate(faces), process=False
    )
    return scene, mesh, np.concatenate(owner), names


def vote_parts(mesh, owner, mask_set, cameras, camera_angle_x, resolution):
    """[parts, names] vote table: how much of each part each semantic name's masks covered.

    Face-level votes are summed over the faces of each part, so one stray triangle can
    never outvote the rest of its object -- the granularity decision stays with the
    unguided segmentation, SAM3 only supplies names.
    """
    vertices, _ = normalize_to_unit_cube(gltf_to_blender(np.asarray(mesh.vertices)))
    faces = np.asarray(mesh.faces)
    face_ids = rasterize_face_ids(vertices, faces, cameras, camera_angle_x, resolution)
    centroids = vertices[faces].mean(axis=1)
    normals = gltf_to_blender(np.asarray(mesh.face_normals, dtype=np.float64))
    weights = view_weights(centroids, normals, cameras)

    votes_faces, _ = accumulate_votes(face_ids, mask_set, weights, len(faces), supersample=1)
    n_parts = int(owner.max()) + 1
    votes = np.zeros((n_parts, votes_faces.shape[1]), dtype=np.float64)
    np.add.at(votes, owner, votes_faces)
    votes, names = part_votes_from_concepts(votes, mask_set.owners)

    # Raw pixel ownership per part, to normalise votes into a coverage fraction:
    # a 30k-face shell catching 25 bleeding mask pixels must not outvote anything.
    pixels_owned = np.zeros(n_parts, dtype=np.int64)
    for view in range(len(face_ids)):
        flat = face_ids[view].ravel()
        hit = flat > 0
        if hit.any():
            np.add.at(pixels_owned, owner[flat[hit] - 1], 1)
    return votes, names, pixels_owned


def assign_parts(votes, names, pixels_owned, unassigned_to=None, min_cover=0.25):
    """Winner-take-all per part, gated by coverage; one name (or None) per part.

    A part earns a name only when that name's masks covered at least `min_cover`
    of the pixels the part owns across the view grid. Winner-take-all alone is too
    weak: mask edges bleeding a few pixels onto a huge neighbour otherwise hand it
    a name it never meaningfully wore.
    """
    votes = np.asarray(votes, dtype=np.float64)
    pixels_owned = np.asarray(pixels_owned, dtype=np.float64)
    assignment = []
    for row, owned in zip(votes, pixels_owned):
        best = int(row.argmax())
        if owned <= 0 or row[best] < min_cover * owned:
            assignment.append(None)
        else:
            assignment.append(names[best])
    if unassigned_to:
        assignment = [name if name is not None else unassigned_to for name in assignment]
    return assignment


# Texture slots that parts_rebake's bake produces; each gets its own atlas page.
_TEXTURE_SLOTS = ("baseColorTexture", "normalTexture", "metallicRoughnessTexture", "emissiveTexture")
# Neutral fill for a slot a member does not have (flat-shaded / colour-only parts).
# MR fill encodes roughness=1, metallic=0 in the glTF G/B channels.
_SLOT_FILL = {
    "baseColorTexture": (255, 255, 255),
    "normalTexture": (128, 128, 255),
    "metallicRoughnessTexture": (255, 255, 0),
    "emissiveTexture": (0, 0, 0),
}


def _atlas_merge(meshes):
    """Merge textured meshes into ONE mesh, packing their textures into an atlas.

    Plain trimesh.util.concatenate cannot do this: each part carries its own baked
    texture, so concatenating naively drops every material but the first. Instead each
    member's textures are blitted into one tile of a grid atlas (per texture slot) and
    its UVs are remapped into that tile. Lossless -- no rebake, no resampling beyond
    the paste itself.
    """
    import math

    from PIL import Image, ImageDraw

    count = len(meshes)
    cols = math.ceil(math.sqrt(count))
    rows = math.ceil(count / cols)

    # Vertices/faces/uv per member, with node transforms already applied by caller.
    tile_size = 0
    member_uvs = []
    for mesh in meshes:
        visual = mesh.visual
        uv = getattr(visual, "uv", None)
        if uv is None:
            # Colour-only part: synthesise degenerate UVs pointing at its tile; the
            # tile is filled with the flat colour below.
            uv = np.zeros((len(mesh.vertices), 2), dtype=np.float64)
        member_uvs.append(np.asarray(uv, dtype=np.float64))
        material = getattr(visual, "material", None)
        texture = getattr(material, "baseColorTexture", None) if material else None
        if texture is not None:
            tile_size = max(tile_size, texture.size[0], texture.size[1])
        elif getattr(visual, "kind", None) == "vertex" or hasattr(visual, "vertex_colors"):
            tile_size = max(tile_size, 1)
    tile_size = max(tile_size, 64)  # at least a few px so solid tiles sample cleanly

    # Only build atlas pages for slots at least one member actually has -- attaching a
    # fill-only page would override the material factors (e.g. a white MR page reads
    # as metallic=1 and renders black).
    member_materials = [getattr(mesh.visual, "material", None) for mesh in meshes]
    active_slots = [
        slot for slot in _TEXTURE_SLOTS
        if slot == "baseColorTexture"
        or any(getattr(m, slot, None) is not None for m in member_materials if m)
    ]
    atlases = {
        slot: Image.new("RGB", (cols * tile_size, rows * tile_size), _SLOT_FILL[slot])
        for slot in active_slots
    }
    for index, mesh in enumerate(meshes):
        tile_x, tile_y = index % cols, index // cols
        # trimesh keeps glTF UVs v-flipped in memory (v=0 at the image BOTTOM), so a
        # uv remapped to tile row `tile_y` samples pixels from atlas row
        # (rows-1-tile_y); paste the tile there.
        box = (tile_x * tile_size, (rows - 1 - tile_y) * tile_size)
        material = member_materials[index]
        for slot, atlas in atlases.items():
            texture = getattr(material, slot, None) if material else None
            if texture is not None:
                atlas.paste(texture.convert("RGB").resize((tile_size, tile_size)),
                            (box[0], box[1]))
            elif slot == "baseColorTexture":
                # Flat-colour member: fill its tile with its own colour.
                colour = (255, 255, 255)
                if hasattr(mesh.visual, "vertex_colors") and len(mesh.visual.vertex_colors):
                    colour = tuple(int(c) for c in mesh.visual.vertex_colors[0][:3])
                elif hasattr(mesh.visual, "face_colors") and len(mesh.visual.face_colors):
                    colour = tuple(int(c) for c in mesh.visual.face_colors[0][:3])
                draw = ImageDraw.Draw(atlas)
                draw.rectangle([box[0], box[1], box[0] + tile_size, box[1] + tile_size],
                               fill=colour)
        member_uvs[index] = (member_uvs[index] + np.array([tile_x, tile_y])) / [cols, rows]

    material_kwargs = {"baseColorTexture": atlases["baseColorTexture"]}
    if "normalTexture" in atlases:
        material_kwargs["normalTexture"] = atlases["normalTexture"]
    if "metallicRoughnessTexture" in atlases:
        material_kwargs["metallicRoughnessTexture"] = atlases["metallicRoughnessTexture"]
    if "emissiveTexture" in atlases:
        material_kwargs["emissiveTexture"] = atlases["emissiveTexture"]
        material_kwargs["emissiveFactor"] = [1.0, 1.0, 1.0]
    # No MR page: carry the members' scalar factors so the merged mesh shades the
    # same as the source parts (trimesh defaults to metallicFactor=1, i.e. black).
    first_material = next((m for m in member_materials if m is not None), None)
    if "metallicRoughnessTexture" not in atlases:
        material_kwargs["metallicFactor"] = float(
            getattr(first_material, "metallicFactor", 0.0) or 0.0)
        material_kwargs["roughnessFactor"] = float(
            getattr(first_material, "roughnessFactor", 1.0) or 1.0)
    material = trimesh.visual.material.PBRMaterial(**material_kwargs)
    out = trimesh.util.concatenate([
        trimesh.Trimesh(vertices=np.asarray(m.vertices), faces=np.asarray(m.faces),
                        process=False)
        for m in meshes
    ])
    out.visual = trimesh.visual.TextureVisuals(
        uv=np.concatenate(member_uvs), material=material
    )
    return out


def merge_parts(parts_glb, assignment, part_order, out_glb, strict=True):
    """Group the part nodes of `parts_glb` by their assigned name and export one glb.

    Parts that share a name are merged into a single mesh (textures packed into an
    atlas, see _atlas_merge), so every output object is exactly one node.
    """
    scene = trimesh.load(parts_glb, force="scene")
    node_names = list(scene.geometry.keys())
    if len(node_names) != len(assignment):
        raise ValueError(f"{len(assignment)} assignments for {len(node_names)} part nodes")

    out = trimesh.Scene()
    manifest = []
    for label, name in enumerate(part_order):
        members = [node for node, a in zip(node_names, assignment) if a == name]
        if not members:
            if strict:
                raise ValueError(f"requested part '{name}' claimed no segmented part")
            continue
        base = f"part_{label:02d}_{name}"
        meshes = []
        for member in members:
            node = scene.graph.geometry_nodes[member][0]
            transform, _ = scene.graph.get(node)
            geometry = scene.geometry[member].copy()
            geometry.apply_transform(transform)
            meshes.append(geometry)
        merged = meshes[0] if len(meshes) == 1 else _atlas_merge(meshes)
        out.add_geometry(merged, node_name=base, geom_name=base)
        manifest.append({
            "label": label,
            "name": name,
            "nodes": [base],
            "members": members,
            "faces": int(len(merged.faces)),
            "area": float(merged.area),
        })

    unnamed = [node for node, a in zip(node_names, assignment) if a not in part_order]
    if unnamed and strict:
        raise ValueError(f"parts left over with no requested name: {unnamed}")

    out_glb = os.path.abspath(out_glb)
    os.makedirs(os.path.dirname(out_glb) or ".", exist_ok=True)
    out.export(out_glb)
    manifest_path = os.path.join(os.path.dirname(out_glb), "parts.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"merged {len(node_names)} parts -> {len(manifest)} objects: {out_glb}")
    for row in manifest:
        print(f"  {row['name']}: {len(row['members'])} parts merged, {row['faces']} faces")
    return manifest
