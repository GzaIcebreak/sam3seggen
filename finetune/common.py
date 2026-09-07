"""Shared building blocks for SegviGen 2D-map fine-tuning data.

Sample layout (one directory per object):

    <obj>/input.glb                 whole textured object
    <obj>/parts/<k>.glb             one GLB per part, natural-sorted order = part index
    <obj>/names.json                optional list of part names aligned with parts/
    <obj>/input.vxz                 512^3 voxelisation of input.glb
    <obj>/ids.vxz                   voxelisation of the parts painted with ID colours
    <obj>/ids_meta.json             {"n_parts", "id_colors", "aabb"}
    <obj>/voxel_part.npy            part index of every ids.vxz voxel (-1 = undecodable)
    <obj>/shape_slat.pth            shape latent on the common coords
    <obj>/input_tex_slat.pth        input texture latent on the common coords
    <obj>/common_coords.pth         latent coords shared by input and every variant
    <obj>/views/<view>/ids.npy      int16 part-index raster (-1 = background)
    <obj>/views/<view>/render.png   textured bpy render (path A only)
    <obj>/variants/<name>/map.png   2D condition map fed to DINOv3
    <obj>/variants/<name>/cond.pth  {"cond", "neg_cond"}
    <obj>/variants/<name>/output_tex_slat.pth   target latent on the common coords
    <obj>/variants/<name>/meta.json {"kind", "view", "groups", "grey_parts", "colors", ...}

Colours: every voxel of a part gets that part's group colour; parts SAM3 (or the
synthetic corruption) never saw are painted GREY in both 2D and 3D, so the model
learns "grey means unassigned" instead of guessing a colour it cannot know.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_TOOLKIT = os.path.join(ROOT, "data_toolkit")
for _p in (ROOT, DATA_TOOLKIT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import trimesh
from PIL import Image

GREY = (150, 150, 150)
WHITE = (255, 255, 255)
LABEL_BG = -1
LABEL_GREY = -2
TRANSFORMS = os.path.join(DATA_TOOLKIT, "transforms.json")
TRELLIS_DIR = os.path.join(ROOT, "microsoft", "TRELLIS.2-4B")
RESOLUTION = 512


def natural_key(name: str):
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r"(\d+)", name)]


class ObjectDir:
    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self.input_glb = os.path.join(self.path, "input.glb")
        self.parts_dir = os.path.join(self.path, "parts")
        self.names_json = os.path.join(self.path, "names.json")
        self.input_vxz = os.path.join(self.path, "input.vxz")
        self.ids_vxz = os.path.join(self.path, "ids.vxz")
        self.ids_meta = os.path.join(self.path, "ids_meta.json")
        self.voxel_part = os.path.join(self.path, "voxel_part.npy")
        self.shape_slat = os.path.join(self.path, "shape_slat.pth")
        self.input_tex_slat = os.path.join(self.path, "input_tex_slat.pth")
        self.common_coords = os.path.join(self.path, "common_coords.pth")
        self.views_dir = os.path.join(self.path, "views")
        self.variants_dir = os.path.join(self.path, "variants")

    def part_files(self) -> list[str]:
        files = [f for f in os.listdir(self.parts_dir) if f.lower().endswith(".glb")]
        return [os.path.join(self.parts_dir, f) for f in sorted(files, key=natural_key)]

    def names(self) -> list[str] | None:
        if not os.path.exists(self.names_json):
            return None
        with open(self.names_json, "r", encoding="utf-8") as f:
            return json.load(f)

    def view_dir(self, tag: str) -> str:
        return os.path.join(self.views_dir, tag)

    def variant_dir(self, name: str) -> str:
        return os.path.join(self.variants_dir, name)

    def is_prepared(self) -> bool:
        return all(os.path.exists(p) for p in (
            self.input_vxz, self.ids_vxz, self.ids_meta, self.voxel_part,
            self.shape_slat, self.input_tex_slat, self.common_coords))


# --------------------------------------------------------------------------- objects

def prepare_object_from_glb(glb_path: str, out_dir: str, names: list[str] | None = None,
                            min_faces: int = 1) -> ObjectDir:
    """Split a multi-geometry GLB into the sample layout. Names default to geometry names."""
    obj = ObjectDir(out_dir)
    os.makedirs(obj.parts_dir, exist_ok=True)
    shutil.copyfile(glb_path, obj.input_glb)
    scene = trimesh.load(glb_path, force="scene")
    kept_names = []
    idx = 0
    for geom_name, geometry in scene.geometry.items():
        if not isinstance(geometry, trimesh.Trimesh) or len(geometry.faces) < min_faces:
            continue
        # Bake the node transform so the part lands where it sits in the assembly.
        transforms = [scene.graph[node][0] for node in scene.graph.nodes_geometry
                      if scene.graph[node][1] == geom_name]
        part = geometry.copy()
        if transforms:
            part.apply_transform(transforms[0])
        part_scene = trimesh.Scene()
        part_scene.add_geometry(part, node_name=f"part_{idx}", geom_name=f"geom_{idx}")
        part_scene.export(os.path.join(obj.parts_dir, f"{idx}.glb"))
        kept_names.append(geom_name)
        idx += 1
    if idx == 0:
        raise ValueError(f"{glb_path}: no mesh geometry")
    with open(obj.names_json, "w", encoding="utf-8") as f:
        json.dump(names if names is not None else kept_names, f, ensure_ascii=False, indent=2)
    return obj


def load_single_mesh(path: str) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        meshes = [m for m in loaded.dump() if isinstance(m, trimesh.Trimesh) and len(m.vertices) > 0]
        return trimesh.util.concatenate(meshes)
    return loaded


def load_parts(obj: ObjectDir) -> list[trimesh.Trimesh]:
    return [load_single_mesh(p) for p in obj.part_files()]


def scene_aabb(glb_path: str) -> np.ndarray:
    return np.asarray(trimesh.load(glb_path, force="scene").bounding_box.bounds, dtype=np.float64)


# --------------------------------------------------------------------------- ID colours

_ID_LEVELS = (0, 85, 170, 255)


def id_colors(n_parts: int) -> list[tuple[int, int, int]]:
    """Up to 64 colours pairwise >= 85 apart, so nearest-colour decoding survives voxel blending."""
    if n_parts > len(_ID_LEVELS) ** 3:
        raise ValueError(f"{n_parts} parts exceed the {len(_ID_LEVELS) ** 3}-colour ID palette")
    colors = [(r, g, b) for r in _ID_LEVELS for g in _ID_LEVELS for b in _ID_LEVELS]
    # Keep white for last: the voxeliser's background/default is white-ish.
    colors.sort(key=lambda c: (c == (255, 255, 255), c))
    return colors[:n_parts]


def decode_nearest(rgb_uint8: np.ndarray, palette: list[tuple[int, int, int]], max_dist: float = 60.0) -> np.ndarray:
    pal = np.asarray(palette, dtype=np.float32)
    x = rgb_uint8.astype(np.float32)
    d2 = ((x[:, None, :] - pal[None, :, :]) ** 2).sum(-1)
    idx = d2.argmin(1)
    idx[np.sqrt(d2[np.arange(len(idx)), idx]) > max_dist] = -1
    return idx.astype(np.int32)


def set_mesh_solid_pbr(mesh: trimesh.Trimesh, rgb: tuple[int, int, int]) -> trimesh.Trimesh:
    from trimesh.visual.material import PBRMaterial
    rgba = np.array([*rgb, 255], dtype=np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=np.tile(rgba, (len(mesh.vertices), 1)))
    f = [c / 255.0 for c in rgb]
    mesh.visual.material = PBRMaterial(baseColorFactor=[*f, 1.0], metallicFactor=0.0,
                                       roughnessFactor=1.0, emissiveFactor=f)
    return mesh


def write_colored_glb(parts: list[trimesh.Trimesh], colors: list[tuple[int, int, int]], path: str) -> None:
    scene = trimesh.Scene()
    for i, (mesh, color) in enumerate(zip(parts, colors)):
        scene.add_geometry(set_mesh_solid_pbr(mesh.copy(), color), node_name=f"part_{i}", geom_name=f"geom_{i}")
    scene.export(path)


# --------------------------------------------------------------------------- voxels

def voxelize(glb_path: str, vxz_path: str) -> None:
    from glb_to_vxz import glb_to_vxz
    glb_to_vxz(glb_path, vxz_path)


def read_vxz(path: str):
    import o_voxel
    coords, data = o_voxel.io.read(path)
    return coords, data


def prepare_object(obj: ObjectDir, encoders, force: bool = False, aabb_tol: float = 2e-3) -> dict:
    """Voxelise input + ID-painted parts, decode voxel->part, encode the shared latents."""
    if obj.is_prepared() and not force:
        with open(obj.ids_meta, "r", encoding="utf-8") as f:
            return json.load(f)
    parts = load_parts(obj)
    union = trimesh.util.concatenate(parts)
    aabb_in = scene_aabb(obj.input_glb)
    aabb_parts = np.asarray(union.bounds, dtype=np.float64)
    extent = float((aabb_in[1] - aabb_in[0]).max())
    if np.abs(aabb_in - aabb_parts).max() > aabb_tol * extent:
        raise ValueError(f"{obj.path}: parts AABB differs from input AABB by "
                         f"{np.abs(aabb_in - aabb_parts).max() / extent:.4f} of extent; voxel grids would misalign")

    colors = id_colors(len(parts))
    ids_glb = os.path.join(obj.path, "ids.glb")
    write_colored_glb(parts, colors, ids_glb)
    if force or not os.path.exists(obj.input_vxz):
        voxelize(obj.input_glb, obj.input_vxz)
    voxelize(ids_glb, obj.ids_vxz)

    coords, data = read_vxz(obj.ids_vxz)
    voxel_part = decode_nearest(data["base_color"].numpy(), colors)
    np.save(obj.voxel_part, voxel_part)

    shape_encoder, tex_encoder = encoders
    from vxz_to_slat import vxz_to_latent_slat, get_common_coords, get_slat_by_common_coords
    in_shape, in_tex = vxz_to_latent_slat(shape_encoder, tex_encoder, obj.input_vxz)
    out_shape, out_tex = vxz_to_latent_slat(shape_encoder, tex_encoder, obj.ids_vxz)
    common = get_common_coords(in_shape, in_tex, out_shape, out_tex)
    save_slat(get_slat_by_common_coords(in_shape, common), obj.shape_slat)
    save_slat(get_slat_by_common_coords(in_tex, common), obj.input_tex_slat)
    torch.save(common.cpu(), obj.common_coords)

    meta = {
        "n_parts": len(parts),
        "id_colors": [list(c) for c in colors],
        "aabb": aabb_in.tolist(),
        "n_voxels": int(coords.shape[0]),
        "undecoded_voxels": int((voxel_part < 0).sum()),
        "n_latent": int(common.shape[0]),
        "part_faces": [int(len(p.faces)) for p in parts],
    }
    with open(obj.ids_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def recolor_voxels(data: dict, voxel_part: np.ndarray, part_color: dict[int, tuple[int, int, int] | None]) -> dict:
    """New attribute dict whose base_color follows part_color; None / missing parts -> GREY."""
    base = np.full((len(voxel_part), 3), GREY, dtype=np.uint8)
    for part, color in part_color.items():
        if color is None:
            continue
        base[voxel_part == part] = np.asarray(color, dtype=np.uint8)
    out = {k: v for k, v in data.items()}
    out["base_color"] = torch.from_numpy(base)
    return out


# --------------------------------------------------------------------------- latents

def load_encoders():
    from trellis2 import models
    shape_encoder = models.from_pretrained(os.path.join(TRELLIS_DIR, "ckpts", "shape_enc_next_dc_f16c32_fp16")).cuda().eval()
    tex_encoder = models.from_pretrained(os.path.join(TRELLIS_DIR, "ckpts", "tex_enc_next_dc_f16c32_fp16")).cuda().eval()
    return shape_encoder, tex_encoder


@torch.no_grad()
def encode_tex(tex_encoder, coords: torch.Tensor, data: dict):
    import trellis2.modules.sparse as sp
    coords = torch.cat([torch.zeros(coords.shape[0], 1, dtype=torch.int32), coords], dim=1).cuda()
    attr = torch.cat([data["base_color"] / 255, data["metallic"] / 255,
                      data["roughness"] / 255, data["alpha"] / 255], dim=-1).float().cuda() * 2 - 1
    return tex_encoder(sp.SparseTensor(attr, coords))


def restrict(slat, common_coords: torch.Tensor):
    from vxz_to_slat import get_slat_by_common_coords
    return get_slat_by_common_coords(slat, common_coords.cuda())


def save_slat(slat, path: str) -> None:
    torch.save({"feats": slat.feats.detach().cpu(), "coords": slat.coords.detach().cpu()}, path)


def load_slat(path: str):
    return torch.load(path, map_location="cpu")


def load_normalization() -> dict:
    with open(os.path.join(TRELLIS_DIR, "pipeline.json"), "r") as f:
        args = json.load(f)["args"]
    return {
        "shape_mean": torch.tensor(args["shape_slat_normalization"]["mean"]),
        "shape_std": torch.tensor(args["shape_slat_normalization"]["std"]),
        "tex_mean": torch.tensor(args["tex_slat_normalization"]["mean"]),
        "tex_std": torch.tensor(args["tex_slat_normalization"]["std"]),
    }


# --------------------------------------------------------------------------- camera / raster

def camera_from_transforms(transforms_json: str, azimuth_deg: float = 0.0):
    """world_to_cam (Blender Z-up world) and fov_x, mirroring bpy_render.set_camera_from_matrix."""
    with open(transforms_json, "r") as f:
        entry = json.load(f)[0]
    cam_to_world = np.asarray(entry["transform_matrix"], dtype=np.float64)
    if azimuth_deg:
        a = math.radians(azimuth_deg)
        orbit = np.array([[math.cos(a), -math.sin(a), 0, 0],
                          [math.sin(a), math.cos(a), 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]], dtype=np.float64)
        cam_to_world = orbit @ cam_to_world
    # TRACK_TO re-aims the camera at the origin; rebuild the rotation from the position.
    eye = cam_to_world[:3, 3]
    forward = -eye / (np.linalg.norm(eye) + 1e-12)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = true_up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = eye
    return np.linalg.inv(c2w), float(entry["camera_angle_x"])


def projection_matrix(fov_x: float, width: int, height: int, z_near: float = 0.01, z_far: float = 100.0) -> np.ndarray:
    aspect = width / height
    f = 1.0 / math.tan(fov_x / 2.0) * aspect  # square images: fov_y == fov_x
    return np.array([
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (z_far + z_near) / (z_near - z_far), (2.0 * z_far * z_near) / (z_near - z_far)],
        [0.0, 0.0, -1.0, 0.0],
    ], dtype=np.float64)


_GX = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)  # glTF Y-up -> Blender Z-up

_glctx = None


def render_part_ids(parts: list[trimesh.Trimesh], aabb: np.ndarray, azimuth_deg: float,
                    transforms_json: str = TRANSFORMS, resolution: int = RESOLUTION) -> np.ndarray:
    """Exact per-pixel part index (-1 background) from the same camera bpy uses."""
    global _glctx
    import nvdiffrast.torch as nr
    if _glctx is None:
        _glctx = nr.RasterizeCudaContext()
    verts, faces, face_part = [], [], []
    offset = 0
    for i, mesh in enumerate(parts):
        verts.append(np.asarray(mesh.vertices, dtype=np.float64))
        faces.append(np.asarray(mesh.faces, dtype=np.int64) + offset)
        face_part.append(np.full(len(mesh.faces), i, dtype=np.int32))
        offset += len(mesh.vertices)
    V = np.concatenate(verts)
    F = np.concatenate(faces)
    face_part = np.concatenate(face_part)
    center = (aabb[0] + aabb[1]) / 2.0
    scale = 1.0 / float((aabb[1] - aabb[0]).max())
    V = (V - center) * scale

    world_to_cam, fov_x = camera_from_transforms(transforms_json, azimuth_deg)
    P = projection_matrix(fov_x, resolution, resolution)
    M = P @ world_to_cam @ _GX
    pos = np.concatenate([V, np.ones((len(V), 1))], axis=1) @ M.T
    pos_t = torch.from_numpy(pos.astype(np.float32)).cuda().unsqueeze(0).contiguous()
    F_t = torch.from_numpy(F.astype(np.int32)).cuda()
    rast, _ = nr.rasterize(_glctx, pos_t, F_t, resolution=[resolution, resolution])
    tri = rast[0, ..., 3].long().cpu().numpy()  # 0 = background, else triangle index + 1
    labels = np.full(tri.shape, LABEL_BG, dtype=np.int16)
    hit = tri > 0
    labels[hit] = face_part[tri[hit] - 1]
    return labels[::-1].copy()


# --------------------------------------------------------------------------- 2D maps

def paint_map(labels: np.ndarray, label_color: dict[int, tuple[int, int, int]]) -> Image.Image:
    """labels: part index >= 0, LABEL_BG, LABEL_GREY. Unlisted parts fall back to GREY."""
    canvas = np.full((*labels.shape, 3), WHITE, dtype=np.uint8)
    fg = labels != LABEL_BG
    canvas[fg] = GREY
    for label, color in label_color.items():
        canvas[labels == label] = np.asarray(color, dtype=np.uint8)
    return Image.fromarray(canvas, mode="RGB")


def random_palette(rng: np.random.Generator, n: int, min_dist: float = 60.0, tries: int = 4000) -> list[tuple[int, int, int]]:
    """Random colours far from each other, from GREY and from WHITE (training used random RGB)."""
    chosen: list[np.ndarray] = [np.asarray(GREY, dtype=np.float32), np.asarray(WHITE, dtype=np.float32)]
    out = []
    for _ in range(n):
        for _ in range(tries):
            c = rng.integers(0, 256, size=3).astype(np.float32)
            if all(np.linalg.norm(c - o) >= min_dist for o in chosen):
                break
        else:
            raise RuntimeError("could not sample a separated palette")
        chosen.append(c)
        out.append(tuple(int(v) for v in c))
    return out


# --------------------------------------------------------------------------- condition

def load_cond_models():
    from trellis2.pipelines.rembg import BiRefNet
    from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor
    # BiRefNet.cuda() returns None, so the calls cannot be chained.
    rembg = BiRefNet(model_name="briaai/RMBG-2.0")
    rembg.cuda()
    dino = DinoV3FeatureExtractor(model_name="facebook/dinov3-vitl16-pretrain-lvd1689m")
    dino.cuda()
    return rembg, dino


def map_to_cond(cond_models, image_path: str, save_path: str) -> None:
    from img_to_cond import img_to_cond
    rembg, dino = cond_models
    img_to_cond(rembg, dino, image_path, save_path)


# --------------------------------------------------------------------------- variants

def write_variant(obj: ObjectDir, name: str, labels: np.ndarray, groups: list[list[int]],
                  grey_parts: list[int], colors: list[tuple[int, int, int]], kind: str, view: str,
                  encoders, cond_models, extra: dict | None = None, force: bool = False,
                  target_from: str | None = None, mask_2d: list[int] | None = None) -> str:
    """Materialise one (2D map, 3D target, cond) triple.

    groups[g] lists the parts sharing colors[g]; grey_parts are painted GREY in 3D;
    `labels` is the already-corrupted 2D label raster (part index / LABEL_BG / LABEL_GREY).
    `target_from` names a sibling variant with the identical (groups, grey, colors): its
    output_tex_slat.pth is copied instead of re-encoded (paired views share one 3D target).
    `mask_2d` parts are painted GREY in the 2D map only (v4 "partial" variants): they keep their
    colour in 3D and in the legend, so only the legend says what colour they are.
    """
    vdir = obj.variant_dir(name)
    done = os.path.join(vdir, "meta.json")
    if os.path.exists(done) and not force:
        return vdir
    os.makedirs(vdir, exist_ok=True)

    part_color: dict[int, tuple[int, int, int] | None] = {p: None for p in grey_parts}
    label_color: dict[int, tuple[int, int, int]] = {}
    for g, members in enumerate(groups):
        for p in members:
            part_color[p] = colors[g]
            label_color[p] = colors[g]
    for p in list(grey_parts) + list(mask_2d or []):
        label_color.pop(p, None)
    label_color[LABEL_GREY] = GREY

    map_path = os.path.join(vdir, "map.png")
    paint_map(labels, label_color).save(map_path)
    map_to_cond(cond_models, map_path, os.path.join(vdir, "cond.pth"))

    src = os.path.join(obj.variant_dir(target_from), "output_tex_slat.pth") if target_from else None
    if src and os.path.exists(src):
        shutil.copyfile(src, os.path.join(vdir, "output_tex_slat.pth"))
    else:
        coords, data = read_vxz(obj.ids_vxz)
        voxel_part = np.load(obj.voxel_part)
        recolored = recolor_voxels(data, voxel_part, part_color)
        _, tex_encoder = encoders
        out_tex = encode_tex(tex_encoder, coords, recolored)
        common = torch.load(obj.common_coords)
        save_slat(restrict(out_tex, common), os.path.join(vdir, "output_tex_slat.pth"))

    meta = {
        "kind": kind, "view": view, "groups": groups, "grey_parts": grey_parts,
        "colors": [list(c) for c in colors],
        "grey_pixels": int((labels == LABEL_GREY).sum()),
        "fg_pixels": int((labels != LABEL_BG).sum()),
    }
    if mask_2d:
        meta["masked_parts"] = list(mask_2d)
        meta["masked_pixels"] = int(np.isin(labels, list(mask_2d)).sum())
    if extra:
        meta.update(extra)
    with open(done, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return vdir


def object_names(dataset_root: str, objects: list[str] | None, objects_file: str | None) -> list[str]:
    if objects:
        return list(objects)
    if objects_file:
        with open(objects_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return sorted(d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d)))


def list_variants(dataset_root: str) -> list[tuple[ObjectDir, str, dict]]:
    out = []
    for name in sorted(os.listdir(dataset_root)):
        obj = ObjectDir(os.path.join(dataset_root, name))
        if not obj.is_prepared() or not os.path.isdir(obj.variants_dir):
            continue
        for vname in sorted(os.listdir(obj.variants_dir)):
            meta_path = os.path.join(obj.variant_dir(vname), "meta.json")
            if not os.path.exists(meta_path):
                continue
            with open(meta_path, "r", encoding="utf-8") as f:
                out.append((obj, vname, json.load(f)))
    return out
