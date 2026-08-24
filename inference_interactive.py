import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch
import trimesh
import o_voxel
import o_voxel.postprocess
import numpy as np
import torch.nn as nn
import trellis2.modules.sparse as sp

from PIL import Image
from tqdm import tqdm
from trellis2 import models
from types import MethodType
from collections import OrderedDict
from torch.nn import functional as F
from trellis2.pipelines.rembg import BiRefNet
from trellis2.modules.utils import manual_cast
from trellis2.representations import MeshWithVoxel
from data_toolkit.bpy_render import render_from_transforms
from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor

class Sampler:
    def _inference_model(self, model, x_t, tex_slat, shape_slat, input_points, coords_len_list, t, cond):
        t = torch.tensor([t*1000] * x_t.shape[0], dtype=torch.float32).cuda()
        return model(x_t, tex_slat, shape_slat, t, cond, input_points, coords_len_list)

    def guidance_inference_model(self, model, x_t, tex_slat, shape_slat, input_points, coords_len_list, t, cond_dict, guidance_strength, guidance_rescale=0.0):
        if guidance_strength == 1:
            return self._inference_model(model, x_t, tex_slat, shape_slat, input_points, coords_len_list, t, cond_dict['cond'])
        elif guidance_strength == 0:
            return self._inference_model(model, x_t, tex_slat, shape_slat, input_points, coords_len_list, t, cond_dict['neg_cond'])
        else:
            pred_pos = self._inference_model(model, x_t, tex_slat, shape_slat, input_points, coords_len_list, t, cond_dict['cond'])
            pred_neg = self._inference_model(model, x_t, tex_slat, shape_slat, input_points, coords_len_list, t, cond_dict['neg_cond'])
            pred = guidance_strength * pred_pos + (1 - guidance_strength) * pred_neg
            if guidance_rescale > 0:
                x_0_pos = self._pred_to_xstart(x_t, t, pred_pos)
                x_0_cfg = self._pred_to_xstart(x_t, t, pred)
                std_pos = x_0_pos.std(dim=list(range(1, x_0_pos.ndim)), keepdim=True)
                std_cfg = x_0_cfg.std(dim=list(range(1, x_0_cfg.ndim)), keepdim=True)
                x_0_rescaled = x_0_cfg * (std_pos / std_cfg)
                x_0 = guidance_rescale * x_0_rescaled + (1 - guidance_rescale) * x_0_cfg
                pred = self._xstart_to_pred(x_t, t, x_0)
            return pred

    def interval_inference_model(self, model, x_t, tex_slat, shape_slat, input_points, coords_len_list, t, cond_dict, sampler_params):
        guidance_strength = sampler_params['guidance_strength']
        guidance_interval = sampler_params['guidance_interval']
        guidance_rescale = sampler_params['guidance_rescale']
        if guidance_interval[0] <= t <= guidance_interval[1]:
            return self.guidance_inference_model(model, x_t, tex_slat, shape_slat, input_points, coords_len_list, t, cond_dict, guidance_strength, guidance_rescale)
        else:
            return self.guidance_inference_model(model, x_t, tex_slat, shape_slat, input_points, coords_len_list, t, cond_dict, 1, guidance_rescale)

    @torch.no_grad()
    def sample_once(self, model, x_t, tex_slat, shape_slat, input_points, coords_len_list, t, t_prev, cond_dict, sampler_params):
        pred_v = self.interval_inference_model(model, x_t, tex_slat, shape_slat, input_points, coords_len_list, t, cond_dict, sampler_params)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return pred_x_prev

    @torch.no_grad()
    def sample(self, model, noise, tex_slat, shape_slat, input_points, coords_len_list, cond_dict, sampler_params):
        sample = noise
        steps = sampler_params['steps']
        rescale_t = sampler_params['rescale_t']
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_seq = t_seq.tolist()
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        for t, t_prev in tqdm(t_pairs, desc="Sampling"):
            sample = self.sample_once(model, sample, tex_slat, shape_slat, input_points, coords_len_list, t, t_prev, cond_dict, sampler_params)
        return sample

def flow_forward(self, x, t, cond, concat_cond, point_embeds, coords_len_list):
    # x.feats: [N, 32]
    x = sp.sparse_cat([x, concat_cond], dim=-1)
    if isinstance(cond, list):
        cond = sp.VarLenTensor.from_tensor_list(cond)
    # x.feats: [N, 64]
    h = self.input_layer(x)
    # h.feats: [N, 1536]
    h = manual_cast(h, self.dtype)
    t_emb = self.t_embedder(t)
    t_emb = self.adaLN_modulation(t_emb)
    t_emb = manual_cast(t_emb, self.dtype)
    cond = manual_cast(cond, self.dtype)
    point_embeds = manual_cast(point_embeds, self.dtype)

    h_feats_list = []
    h_coords_list = []
    begin = 0
    for i, coords_len in enumerate(coords_len_list):
        end = begin + 2 * coords_len
        h_feats_list.append(h.feats[begin:end])
        h_coords_list.append(h.coords[begin:end])
        h_feats_list.append(point_embeds.feats[i*10:(i+1)*10])
        h_coords_list.append(point_embeds.coords[i*10:(i+1)*10])
        begin = end + 10
    h = sp.SparseTensor(torch.cat(h_feats_list), torch.cat(h_coords_list))

    for block in self.blocks:
        h = block(h, t_emb, cond)

    h_feats_list = []
    h_coords_list = []
    begin = 0
    for i, coords_len in enumerate(coords_len_list):
        end = begin + 2 * coords_len
        h_feats_list.append(h.feats[begin:end])
        h_coords_list.append(h.coords[begin:end])
        begin = end
    h = sp.SparseTensor(torch.cat(h_feats_list), torch.cat(h_coords_list))

    h = manual_cast(h, x.dtype)
    h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
    # h.feats: [N, 1536]
    h = self.out_layer(h)
    # h.feats: [N, 32]
    return h

class Gen3DSeg(nn.Module):
    def __init__(self, flow_model):
        super().__init__()
        self.flow_model = flow_model
        self.seg_embeddings = nn.Embedding(1, 1536)
        
    def get_positional_encoding(self, input_points):
        point_feats_embed = torch.zeros((10, 1536), dtype=torch.float32).to(input_points['point_slats'].feats.device)
        labels = input_points['point_labels'].squeeze(-1)
        point_feats_embed[labels == 1] = self.seg_embeddings.weight
        return sp.SparseTensor(point_feats_embed, input_points['point_slats'].coords)

    def forward(self, x_t, tex_slats, shape_slats, t, cond, input_points, coords_len_list):
        input_tex_feats_list = []
        input_tex_coords_list = []
        shape_feats_list = []
        shape_coords_list = []
        begin = 0
        for coords_len in coords_len_list:
            end = begin + coords_len
            input_tex_feats_list.append(x_t.feats[begin:end])
            input_tex_feats_list.append(tex_slats.feats[begin:end])
            input_tex_coords_list.append(x_t.coords[begin:end])
            input_tex_coords_list.append(tex_slats.coords[begin:end])
            shape_feats_list.append(shape_slats.feats[begin:end])
            shape_feats_list.append(shape_slats.feats[begin:end])
            shape_coords_list.append(shape_slats.coords[begin:end])
            shape_coords_list.append(shape_slats.coords[begin:end])
            begin = end
        x_t = sp.SparseTensor(torch.cat(input_tex_feats_list), torch.cat(input_tex_coords_list))
        shape_slats = sp.SparseTensor(torch.cat(shape_feats_list), torch.cat(shape_coords_list))

        point_embeds = self.get_positional_encoding(input_points)
        output_tex_slats = self.flow_model(x_t, t, cond, shape_slats, point_embeds, coords_len_list)
        
        output_tex_feats_list = []
        output_tex_coords_list = []
        begin = 0
        for coords_len in coords_len_list:
            end = begin + coords_len
            output_tex_feats_list.append(output_tex_slats.feats[begin:end])
            output_tex_coords_list.append(output_tex_slats.coords[begin:end])
            begin = begin + 2 * coords_len
        output_tex_slat = sp.SparseTensor(torch.cat(output_tex_feats_list), torch.cat(output_tex_coords_list))
        return output_tex_slat

def make_texture_square_pow2(img: Image.Image, target_size=None):
    w, h = img.size
    max_side = max(w, h)
    pow2 = 1
    while pow2 < max_side:
        pow2 *= 2
    if target_size is not None:
        pow2 = target_size
    pow2 = min(pow2, 2048)
    return img.resize((pow2, pow2), Image.BILINEAR)

def _ensure_pbr_texture_visuals(geom):
    visual = getattr(geom, "visual", None)
    if visual is None:
        return
    if isinstance(visual, trimesh.visual.color.ColorVisuals):
        geom.visual = visual.to_texture()
        visual = geom.visual
    mat = getattr(visual, "material", None)
    if isinstance(mat, trimesh.visual.material.SimpleMaterial):
        geom.visual.material = mat.to_pbr()


def preprocess_scene_textures(asset):
    if not isinstance(asset, trimesh.Scene):
        return asset
    TEX_KEYS = ["baseColorTexture", "normalTexture", "metallicRoughnessTexture", "emissiveTexture", "occlusionTexture"]
    for geom in asset.geometry.values():
        if isinstance(geom, trimesh.Trimesh):
            _ensure_pbr_texture_visuals(geom)
        visual = getattr(geom, "visual", None)
        mat = getattr(visual, "material", None)
        if mat is None:
            continue
        for key in TEX_KEYS:
            if not hasattr(mat, key):
                continue
            tex = getattr(mat, key)
            if tex is None:
                continue
            if isinstance(tex, Image.Image):
                setattr(mat, key, make_texture_square_pow2(tex))
            elif hasattr(tex, "image") and tex.image is not None:
                img = tex.image
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img)
                tex.image = make_texture_square_pow2(img)
        if hasattr(mat, "image") and mat.image is not None:
            img = mat.image
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            mat.image = make_texture_square_pow2(img)
    return asset

def process_glb_to_vxz(glb_path, vxz_path):
    asset = trimesh.load(glb_path, force='scene')
    asset = preprocess_scene_textures(asset)
    aabb = asset.bounding_box.bounds
    center = (aabb[0] + aabb[1]) / 2
    scale = 0.99999 / (aabb[1] - aabb[0]).max()
    asset.apply_translation(-center)
    asset.apply_scale(scale)
    mesh = asset.to_mesh()
    vertices = torch.from_numpy(mesh.vertices).float()
    faces = torch.from_numpy(mesh.faces).long()

    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices, faces, grid_size=512, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2, timing=False
    )
    vid = o_voxel.serialize.encode_seq(voxel_indices)
    mapping = torch.argsort(vid)
    voxel_indices = voxel_indices[mapping]
    dual_vertices = dual_vertices[mapping]
    intersected = intersected[mapping]

    voxel_indices_mat, attributes = o_voxel.convert.textured_mesh_to_volumetric_attr(
        asset, grid_size=512, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], timing=False
    )
    vid_mat = o_voxel.serialize.encode_seq(voxel_indices_mat)
    mapping_mat = torch.argsort(vid_mat)
    attributes = {k: v[mapping_mat] for k, v in attributes.items()}

    dual_vertices = dual_vertices * 512 - voxel_indices
    dual_vertices = (torch.clamp(dual_vertices, 0, 1) * 255).type(torch.uint8)
    intersected = (intersected[:, 0:1] + 2 * intersected[:, 1:2] + 4 * intersected[:, 2:3]).type(torch.uint8)

    attributes['dual_vertices'] = dual_vertices
    attributes['intersected'] = intersected
    o_voxel.io.write(vxz_path, voxel_indices, attributes)

def vxz_to_latent_slat(shape_encoder, shape_decoder, tex_encoder, vxz_path):
    coords, data = o_voxel.io.read(vxz_path)
    coords = torch.cat([torch.zeros(coords.shape[0], 1, dtype=torch.int32), coords], dim=1).cuda()
    vertices = (data['dual_vertices'].cuda() / 255)
    intersected = torch.cat([data['intersected'] % 2, data['intersected'] // 2 % 2, data['intersected'] // 4 % 2], dim=-1).bool().cuda()
    vertices_sparse = sp.SparseTensor(vertices, coords)
    intersected_sparse = sp.SparseTensor(intersected.float(), coords)
    with torch.no_grad():
        shape_slat = shape_encoder(vertices_sparse, intersected_sparse)
        shape_slat = sp.SparseTensor(shape_slat.feats.cuda(), shape_slat.coords.cuda())
        shape_decoder.set_resolution(512)
        meshes, subs = shape_decoder(shape_slat, return_subs=True)
    
    base_color = (data['base_color'] / 255)
    metallic = (data['metallic'] / 255)
    roughness = (data['roughness'] / 255)
    alpha = (data['alpha'] / 255)
    attr = torch.cat([base_color, metallic, roughness, alpha], dim=-1).float().cuda() * 2 - 1
    with torch.no_grad():
        tex_slat = tex_encoder(sp.SparseTensor(attr, coords))
    return shape_slat, meshes, subs, tex_slat

def preprocess_image(rembg_model, input):
    if input.mode != "RGB":
        bg = Image.new("RGB", input.size, (255, 255, 255))
        bg.paste(input, mask=input.split()[3])
        input = bg
    has_alpha = False
    if input.mode == 'RGBA':
        alpha = np.array(input)[:, :, 3]
        if not np.all(alpha == 255):
            has_alpha = True
    max_size = max(input.size)
    scale = min(1, 1024 / max_size)
    if scale < 1:
        input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
    if has_alpha:
        output = input
    else:
        input = input.convert('RGB')
        output = rembg_model(input)
    output_np = np.array(output)
    alpha = output_np[:, :, 3]
    bbox = np.argwhere(alpha > 0.8 * 255)
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1)
    bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
    output = output.crop(bbox)  # type: ignore
    output = np.array(output).astype(np.float32) / 255
    output = output[:, :, :3] * output[:, :, 3:4]
    output = Image.fromarray((output * 255).astype(np.uint8))
    return output

def get_cond(image_cond_model, image):
    image_cond_model.image_size = 512
    cond = image_cond_model(image)
    neg_cond = torch.zeros_like(cond)
    return {'cond': cond, 'neg_cond': neg_cond}

def tex_slat_sample_single(gen3dseg, sampler, pipeline_args, shape_slat, input_tex_slat, cond_dict, input_points):
    device = shape_slat.feats.device
    shape_std = torch.tensor(pipeline_args['shape_slat_normalization']['std'])[None].to(device)
    shape_mean = torch.tensor(pipeline_args['shape_slat_normalization']['mean'])[None].to(device)
    tex_std = torch.tensor(pipeline_args['tex_slat_normalization']['std'])[None].to(device)
    tex_mean = torch.tensor(pipeline_args['tex_slat_normalization']['mean'])[None].to(device)
    shape_slat = ((shape_slat - shape_mean) / shape_std)
    input_tex_slat = ((input_tex_slat - tex_mean) / tex_std)
    coords_len_list = [shape_slat.coords.shape[0]]
    noise = sp.SparseTensor(torch.randn_like(input_tex_slat.feats), shape_slat.coords)
    output_tex_slat = sampler.sample(gen3dseg, noise, input_tex_slat, shape_slat, input_points, coords_len_list, cond_dict, pipeline_args['tex_slat_sampler']['params'])
    output_tex_slat = output_tex_slat * tex_std + tex_mean
    return output_tex_slat

def drop_offbody_components(mesh, aabb_limit=0.5, span_tolerance=0.98, max_drop_ratio=0.05):
    """Drop shells that the attribute-guided remesh extruded out to the aabb faces.

    to_glb's remesh is steered by the predicted attribute volume, so a colour condition
    that spills past the silhouette can extrude thin shells that stop exactly on the
    aabb. The input was normalised so that only its longest axis spans the full aabb;
    any other axis reaching an aabb face is therefore geometry that does not belong to
    the body. Bail out if the rule wants to delete a lot, which would mean the body
    itself landed on the aabb.
    """
    from trimesh.graph import connected_components

    faces = np.asarray(mesh.faces)
    if len(faces) == 0:
        return mesh
    extent = np.asarray(mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0))
    constrained = np.flatnonzero(extent < span_tolerance * extent.max())
    if constrained.size == 0:
        return mesh

    components = connected_components(mesh.face_adjacency, nodes=np.arange(len(faces)))
    dropped = []
    for component in components:
        vertices = np.asarray(mesh.vertices[faces[component].ravel()])[:, constrained]
        if (np.abs(vertices) >= aabb_limit).any():
            dropped.append(component)
    if not dropped:
        return mesh

    drop_faces = np.concatenate(dropped)
    if len(drop_faces) > max_drop_ratio * len(faces):
        print(f"Off-body cleanup skipped: {len(drop_faces)} of {len(faces)} faces flagged")
        return mesh
    keep = np.ones(len(faces), dtype=bool)
    keep[drop_faces] = False
    mesh.update_faces(keep)
    mesh.remove_unreferenced_vertices()
    print(f"Removed {len(dropped)} off-body components ({len(drop_faces)} faces)")
    return mesh

def slat_to_glb(meshes, tex_voxels, resolution=512):
    pbr_attr_layout = {
        'base_color': slice(0, 3),
        'metallic': slice(3, 4),
        'roughness': slice(4, 5),
        'alpha': slice(5, 6),
    }
    out_mesh = []
    for m, v in zip(meshes, tex_voxels):
        m.fill_holes()
        out_mesh.append(
            MeshWithVoxel(
                m.vertices, m.faces,
                origin = [-0.5, -0.5, -0.5],
                voxel_size = 1 / resolution,
                coords = v.coords[:, 1:],
                attrs = v.feats,
                voxel_shape = torch.Size([*v.shape, *v.spatial_shape]),
                layout=pbr_attr_layout
            )
        )
    mesh = out_mesh[0]
    mesh.simplify(10000000)
    # mesh.simplify(16777216) # nvdiffrast limit
    glb = o_voxel.postprocess.to_glb(
        vertices            =   mesh.vertices,
        faces               =   mesh.faces,
        attr_volume         =   mesh.attrs,
        coords              =   mesh.coords,
        attr_layout         =   mesh.layout,
        voxel_size          =   mesh.voxel_size,
        aabb                =   [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target   =   100000, # 1000000
        texture_size        =   4096,
        remesh              =   True,
        # remesh              =   False,
        remesh_band         =   1,
        remesh_project      =   0,
        verbose             =   True
    )
    return drop_offbody_components(glb)

def maybe_blender_reuv(mesh, item):
    if not item.get("blender_reuv"):
        return mesh
    from data_toolkit.parts_rebake import blender_reuv_and_bake
    print("-" * 100)
    print("Blender re-UV + bake ............")
    return blender_reuv_and_bake(
        mesh,
        item["glb"],
        texture_size=item.get("rebake_texture_size", 2048),
    )

POINT_SLOTS = 10


def to_latent_coords(tex_encoder, points):
    """Voxel seeds in [0, 511]^3 as the latent coordinates the point channel lives in.

    Seeds arrive spread widest-first, and only the first few survive the slot budget, so
    duplicates are dropped in place rather than with torch.unique -- sorting them by
    coordinate would quietly replace that spread with whichever corner sorts lowest.
    """
    coords = torch.tensor(points, dtype=torch.int32).cuda()
    coords = torch.cat([torch.zeros((coords.shape[0], 1), dtype=torch.int32).cuda(), coords], dim=1)
    encoded = tex_encoder(sp.SparseTensor(torch.zeros((coords.shape[0], 6), dtype=torch.float32).cuda(), coords)).coords
    seen, kept = set(), []
    for row in encoded.tolist():
        key = tuple(row)
        if key not in seen:
            seen.add(key)
            kept.append(row)
    return torch.tensor(kept, dtype=torch.int32, device=encoded.device)


def build_input_points(tex_encoder, positive, negative=(), negative_slots=0):
    """Wrap voxel seeds as the sparse point tensor the model was trained to condition on.

    There are only ten slots, and what fills them decides where the part stops. Seeds taken
    from the neighbouring parts and marked negative look like the obvious way to pin a
    boundary down, but they cost positive slots and measurably widen the result -- the staff
    grows from 4% of the surface to 25% -- so they are off unless asked for. Leftover slots
    sit at the origin marked negative, as in training.
    """
    positive_coords = to_latent_coords(tex_encoder, positive)
    negative_coords = to_latent_coords(tex_encoder, negative) if negative_slots and len(negative) else positive_coords[:0]
    # A negative that landed in the same latent voxel as a positive would contradict it.
    if len(negative_coords):
        clash = (negative_coords[:, None, :] == positive_coords[None, :, :]).all(dim=-1).any(dim=1)
        negative_coords = negative_coords[~clash]

    negative_coords = negative_coords[:negative_slots]
    positive_coords = positive_coords[: POINT_SLOTS - len(negative_coords)]
    negative_coords = negative_coords[:POINT_SLOTS - len(positive_coords)]

    coords = torch.cat([positive_coords, negative_coords], dim=0)
    labels = [[1]] * len(positive_coords) + [[0]] * len(negative_coords)
    spare = POINT_SLOTS - len(coords)
    if spare:
        coords = torch.cat([coords, torch.zeros((spare, 4), dtype=torch.int32).cuda()], dim=0)
        labels += [[0]] * spare
    print(f"  {len(positive_coords)} positive and {len(negative_coords)} negative latent points")
    return {
        'point_slats': sp.SparseTensor(coords, coords),
        'point_labels': torch.tensor(labels, dtype=torch.int32).cuda(),
    }


def paint_parts_volume(template, confidences, palette):
    """One attribute volume where every voxel carries the colour of its winning part.

    Meshing happens after decimation and a remesh that both follow the attributes, so two
    runs never produce the same topology and per-face masks cannot be lined up afterwards.
    The decoded volumes do share their coordinates, though, so the parts are combined here
    instead, and the result is a single palette-coloured volume -- the same form the
    non-interactive checkpoint emits, which the existing split-and-bake path already reads.
    """
    winner = torch.stack(confidences).argmax(dim=0)
    feats = template.feats.clone()
    colours = torch.tensor(palette, dtype=feats.dtype, device=feats.device) / 255.0
    feats[:, 0:3] = colours[winner]
    return sp.SparseTensor(feats, template.coords), winner


def inference(ckpt_path, item, parts):
    """Segment one part per entry in `parts`, a list of (name, voxel seeds).

    Voxelising the mesh, encoding it and embedding the conditioning image dominate the
    runtime and depend only on the input, so they happen once no matter how many parts are
    asked for. Only the sampling and the texture decode repeat.
    """
    print("-"*100)
    print("Loading model ............")
    with open("microsoft/TRELLIS.2-4B/pipeline.json", "r") as f:
        pipeline_config = json.load(f)
    pipeline_args = pipeline_config['args']
    tex_slat_flow_model = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16")
    tex_slat_flow_model.forward = MethodType(flow_forward, tex_slat_flow_model)

    gen3dseg = Gen3DSeg(tex_slat_flow_model)
    state_dict = torch.load(ckpt_path)['state_dict']
    state_dict = OrderedDict([(k.replace("gen3dseg.", ""), v) for k, v in state_dict.items()])
    gen3dseg.load_state_dict(state_dict)
    gen3dseg.eval()
    gen3dseg.cuda()
    sampler = Sampler()

    shape_encoder = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16").cuda().eval()
    tex_encoder = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16").cuda().eval()
    shape_decoder = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16").cuda().eval()
    tex_decoder = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16").cuda().eval()

    rembg_model = BiRefNet(model_name="briaai/RMBG-2.0")
    rembg_model.cuda()
    image_cond_model = DinoV3FeatureExtractor(model_name="facebook/dinov3-vitl16-pretrain-lvd1689m")
    image_cond_model.cuda()

    process_glb_to_vxz(item['glb'], item['input_vxz'])
    shape_slat, meshes, subs, tex_slat = vxz_to_latent_slat(shape_encoder, shape_decoder, tex_encoder, item['input_vxz'])

    print("-"*100)
    print("Getting cond ............")
    render_from_transforms(item['glb'], item['transforms'], item['img'])
    image = Image.open(item['img'])
    image = preprocess_image(rembg_model, image)
    cond = get_cond(image_cond_model, [image])

    region_confidence, volumes = {}, []
    for region, part, seeds in parts:
        print("-"*100)
        print(f"Sampling {region} from {len(seeds)} seed points ................")
        # Round-robin so every neighbouring region gets a negative slot; concatenating would
        # let the first one use them all up and leave the other boundaries unconstrained.
        others = [points for other, _, points in parts if other != region]
        negative = [points[i] for i in range(max(map(len, others))) for points in others if i < len(points)] if others else []
        input_points = build_input_points(tex_encoder, seeds, negative, item.get("negative_slots", 0))

        output_tex_slat = tex_slat_sample_single(gen3dseg, sampler, pipeline_args, shape_slat, tex_slat, cond, input_points)
        with torch.no_grad():
            tex_voxels = tex_decoder(output_tex_slat, guide_subs=subs) * 0.5 + 0.5

        confidence = tex_voxels[0].feats[:, 0:3].mean(dim=-1)
        print(f"  claimed {float((confidence > 0.5).float().mean()) * 100:.1f}% of voxels")
        region_confidence[region] = confidence
        volumes.append(tex_voxels)

    if item.get("export_dir"):
        for (region, _, _), tex_voxels in zip(parts, volumes):
            path = os.path.join(item["export_dir"], f"{region}.glb")
            maybe_blender_reuv(slat_to_glb(meshes, tex_voxels), item).export(path)
            print(f"wrote {path}")

    # A grouped part is the union of its regions, and a union of near-binary masks is a max.
    part_names, confidences = [], []
    for _, part, _ in parts:
        if part not in part_names:
            part_names.append(part)
    for part in part_names:
        members = [region for region, other, _ in parts if other == part]
        confidences.append(torch.stack([region_confidence[r] for r in members]).amax(dim=0))
        if len(members) > 1:
            print(f"{part}: union of {len(members)} regions -> "
                  f"{float((confidences[-1] > 0.5).float().mean()) * 100:.1f}% of voxels")
    parts = [(name, name, None) for name in part_names]

    if item.get("export_confidence"):
        # The voxel grid sits in the input glb's own coordinates, so these confidences can be
        # read back on the original mesh by looking up each face centroid -- no correspondence
        # between the original topology and the remesh below is needed.
        np.savez_compressed(
            item["export_confidence"],
            coords=volumes[0][0].coords[:, 1:].cpu().numpy().astype(np.int32),
            confidence=torch.stack(confidences, dim=1).float().cpu().numpy(),
            names=np.array(part_names),
        )
        print(f"wrote {item['export_confidence']}")

    print("-"*100)
    print("Combining parts and exporting glb ............")
    combined, winner = paint_parts_volume(volumes[0][0], confidences, item["palette"])
    maybe_blender_reuv(slat_to_glb(meshes, [combined]), item).export(item["export_glb"])
    for index, name in enumerate(part_names):
        print(f"  {name}: {float((winner == index).float().mean()) * 100:.1f}% of voxels")
    print(f"wrote {item['export_glb']}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to trained checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--glb",
        type=str,
        required=True,
        help="Input glb path.",
    )
    parser.add_argument(
        "--input_vxz",
        type=str,
        required=True,
        help="Intermediate vxz path.",
    )
    parser.add_argument(
        "--transforms",
        type=str,
        required=True,
        help="Path to transforms.json.",
    )
    parser.add_argument(
        "--img",
        type=str,
        required=True,
        help="Path to rendered input image.",
    )
    parser.add_argument(
        "--export_glb",
        type=str,
        required=True,
        help="Output glb path: one mesh with a flat palette colour per part.",
    )
    parser.add_argument(
        "--input_vxz_points",
        type=int,
        nargs="+",
        help="List of voxel coordinates in vxz space in [0-511]^3 space, as a flat list of ints: x1 y1 z1 x2 y2 z2 ...",
    )
    parser.add_argument(
        "--seed_points",
        type=str,
        help="JSON mapping part name to voxel seeds, as written by data_toolkit/part_seed_points.py. "
             "Every part is segmented in one process, which loads the model and encodes the mesh once.",
    )
    parser.add_argument(
        "--export_dir",
        type=str,
        help="Optional: also write each part's raw mask as its own glb here, for inspection.",
    )
    parser.add_argument(
        "--negative_slots",
        type=int,
        default=0,
        help="How many of the ten point slots to fill with seeds from the other parts, marked "
             "negative. They cost positive slots and in testing widened every part, so the "
             "default is none.",
    )
    parser.add_argument(
        "--blender_reuv",
        action="store_true",
        help="After dropping off-body components, Smart Project new UVs in Blender and bake the source albedo back.",
    )
    parser.add_argument(
        "--rebake_texture_size",
        type=int,
        default=2048,
        help="Bake resolution used with --blender_reuv.",
    )

    args = parser.parse_args()
    if args.seed_points:
        with open(os.path.abspath(args.seed_points), "r") as f:
            parts = [
                (region, entry["part"], entry["points"])
                for region, entry in json.load(f).items()
            ]
    else:
        if not args.input_vxz_points:
            parser.error("pass either --seed_points or --input_vxz_points")
        if len(args.input_vxz_points) % 3 != 0:
            parser.error("--input_vxz_points length must be a multiple of 3 (x y z per point).")
        parts = [("part", "part", [args.input_vxz_points[i : i + 3] for i in range(0, len(args.input_vxz_points), 3)])]

    from data_toolkit.parts_rebake import LABEL_COLORS
    distinct_parts = list(dict.fromkeys(part for _, part, _ in parts))
    export_glb = os.path.abspath(args.export_glb)
    os.makedirs(os.path.dirname(export_glb) or ".", exist_ok=True)
    item = {
        "glb": os.path.abspath(args.glb),
        "input_vxz": os.path.abspath(args.input_vxz),
        "transforms": os.path.abspath(args.transforms),
        "img": os.path.abspath(args.img),
        "export_glb": export_glb,
        "export_dir": os.path.abspath(args.export_dir) if args.export_dir else None,
        "export_confidence": os.path.splitext(export_glb)[0] + "_confidence.npz",
        "negative_slots": args.negative_slots,
        "palette": LABEL_COLORS[: len(distinct_parts)].tolist(),
        "blender_reuv": args.blender_reuv,
        "rebake_texture_size": args.rebake_texture_size,
    }
    if item["export_dir"]:
        os.makedirs(item["export_dir"], exist_ok=True)
    with open(os.path.splitext(export_glb)[0] + "_parts.json", "w") as f:
        json.dump(distinct_parts, f, indent=2)
    inference(os.path.abspath(args.ckpt_path), item, parts)
