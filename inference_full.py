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
from collections import OrderedDict
from trellis2.pipelines.rembg import BiRefNet
from trellis2.representations import MeshWithVoxel
from data_toolkit.bpy_render import render_from_transforms
from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor

class Sampler:
    def _inference_model(self, model, x_t, tex_slat, shape_slat, coords_len_list, t, cond):
        t = torch.tensor([t*1000] * x_t.shape[0], dtype=torch.float32).cuda()
        return model(x_t, tex_slat, shape_slat, t, cond, coords_len_list)

    def guidance_inference_model(self, model, x_t, tex_slat, shape_slat, coords_len_list, t, cond_dict, guidance_strength, guidance_rescale=0.0):
        if guidance_strength == 1:
            return self._inference_model(model, x_t, tex_slat, shape_slat, coords_len_list, t, cond_dict['cond'])
        elif guidance_strength == 0:
            return self._inference_model(model, x_t, tex_slat, shape_slat, coords_len_list, t, cond_dict['neg_cond'])
        else:
            pred_pos = self._inference_model(model, x_t, tex_slat, shape_slat, coords_len_list, t, cond_dict['cond'])
            pred_neg = self._inference_model(model, x_t, tex_slat, shape_slat, coords_len_list, t, cond_dict['neg_cond'])
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

    def interval_inference_model(self, model, x_t, tex_slat, shape_slat, coords_len_list, t, cond_dict, sampler_params):
        guidance_strength = sampler_params['guidance_strength']
        guidance_interval = sampler_params['guidance_interval']
        guidance_rescale = sampler_params['guidance_rescale']
        if guidance_interval[0] <= t <= guidance_interval[1]:
            return self.guidance_inference_model(model, x_t, tex_slat, shape_slat, coords_len_list, t, cond_dict, guidance_strength, guidance_rescale)
        else:
            return self.guidance_inference_model(model, x_t, tex_slat, shape_slat, coords_len_list, t, cond_dict, 1, guidance_rescale)

    @torch.no_grad()
    def sample_once(self, model, x_t, tex_slat, shape_slat, coords_len_list, t, t_prev, cond_dict, sampler_params):
        pred_v = self.interval_inference_model(model, x_t, tex_slat, shape_slat, coords_len_list, t, cond_dict, sampler_params)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return pred_x_prev

    @torch.no_grad()
    def sample(self, model, noise, tex_slat, shape_slat, coords_len_list, cond_dict, sampler_params):
        sample = noise
        steps = sampler_params['steps']
        rescale_t = sampler_params['rescale_t']
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_seq = t_seq.tolist()
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        for t, t_prev in tqdm(t_pairs, desc="Sampling"):
            sample = self.sample_once(model, sample, tex_slat, shape_slat, coords_len_list, t, t_prev, cond_dict, sampler_params)
        return sample

class Gen3DSeg(nn.Module):
    def __init__(self, flow_model):
        super().__init__()
        self.flow_model = flow_model

    def forward(self, x_t, tex_slats, shape_slats, t, cond, coords_len_list):
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

        output_tex_slats = self.flow_model(x_t, t, cond, shape_slats)
        
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


def load_legend_encoder(payload_path):
    """finetune/train.py payload (lora_*.pt) -> LegendEncoder, or None if it has no legend state."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetune"))
    from model import LegendEncoder
    payload = torch.load(payload_path, map_location="cpu")
    state = payload.get("legend")
    if not state:
        return None
    enc = LegendEncoder(text_dim=state["text_proj.weight"].shape[1])
    enc.load_state_dict(state)
    return enc.cuda().eval()


def read_legend(path):
    """legend json: sam3_to_2dmap's *_legend.json (rows with color + text_vec) or
    {"entries": [{"color": [r,g,b], "text_vec": [...]}, ...], "object_text_vec": [...] | null}."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    obj_vec = None
    if isinstance(data, dict):
        obj_vec = data.get("object_text_vec")
        rows = data.get("entries", [])
    else:
        rows = data
    text, rgb = [], []
    for row in rows:
        if row.get("text_vec") is None or row.get("prompt") == "<unassigned>":
            continue
        text.append(torch.tensor(row["text_vec"], dtype=torch.float32))
        rgb.append(torch.tensor(row["color"], dtype=torch.float32) / 255.0)
    return (torch.stack(text) if text else None, torch.stack(rgb) if rgb else None,
            torch.tensor(obj_vec, dtype=torch.float32) if obj_vec else None)


@torch.no_grad()
def get_cond_v3(image_cond_model, image, image2, legend_encoder, legend_path):
    """v3 context: main-view DINO tokens (+ partner view) (+ object token) (+ legend tokens).
    Returned as 1-element lists (variable-length context); neg_cond stays the zero image tokens."""
    image_cond_model.image_size = 512
    cond = image_cond_model([image])[0].float()
    cond2 = image_cond_model([image2])[0].float() if image2 is not None else None
    text = rgb = obj = None
    if legend_path:
        text, rgb, obj = read_legend(legend_path)
        text, rgb = (t.cuda() if t is not None else None for t in (text, rgb))
        obj = obj.cuda() if obj is not None else None
        n = 0 if text is None else text.shape[0]
        print(f"legend: {n} colour token(s){', object token' if obj is not None else ''}"
              f"{', second view' if cond2 is not None else ''}")
    full = legend_encoder(cond, cond2, text, rgb, obj)
    return {'cond': [full], 'neg_cond': [torch.zeros_like(cond)]}

def tex_slat_sample_single(gen3dseg, sampler, pipeline_args, shape_slat, input_tex_slat, cond_dict):
    device = shape_slat.feats.device
    shape_std = torch.tensor(pipeline_args['shape_slat_normalization']['std'])[None].to(device)
    shape_mean = torch.tensor(pipeline_args['shape_slat_normalization']['mean'])[None].to(device)
    tex_std = torch.tensor(pipeline_args['tex_slat_normalization']['std'])[None].to(device)
    tex_mean = torch.tensor(pipeline_args['tex_slat_normalization']['mean'])[None].to(device)
    shape_slat = ((shape_slat - shape_mean) / shape_std)
    input_tex_slat = ((input_tex_slat - tex_mean) / tex_std)
    coords_len_list = [shape_slat.coords.shape[0]]
    noise = sp.SparseTensor(torch.randn_like(input_tex_slat.feats), shape_slat.coords)
    output_tex_slat = sampler.sample(gen3dseg, noise, input_tex_slat, shape_slat, coords_len_list, cond_dict, pipeline_args['tex_slat_sampler']['params'])
    output_tex_slat = output_tex_slat * tex_std + tex_mean
    return output_tex_slat

def drop_offbody_components(mesh, aabb_limit=0.5, span_tolerance=0.98, max_drop_ratio=0.05):
    """Drop shells that the attribute-guided remesh extruded out to the aabb faces.

    to_glb's remesh is steered by the predicted attribute volume, so a colour condition
    that spills past the silhouette can extrude thin shells that stop exactly on the
    aabb. The input was normalised so that only its longest axis spans the full aabb;
    any other axis reaching an aabb face is therefore geometry that does not belong to
    the body. Besides detached shells, trim one-face-wide bridge chains whose leaf reaches
    an aabb face: these are connected to the body, so component cleanup cannot see them.
    Bail out if either rule wants to delete a lot, which would mean the body itself landed
    on the aabb.
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
    if dropped:
        drop_faces = np.concatenate(dropped)
        if len(drop_faces) > max_drop_ratio * len(faces):
            print(f"Off-body cleanup skipped: {len(drop_faces)} of {len(faces)} faces flagged")
            return mesh
        keep = np.ones(len(faces), dtype=bool)
        keep[drop_faces] = False
        mesh.update_faces(keep)
        mesh.remove_unreferenced_vertices()
        print(f"Removed {len(dropped)} off-body components ({len(drop_faces)} faces)")

    faces = np.asarray(mesh.faces)
    adjacency = np.asarray(mesh.face_adjacency)
    if len(adjacency) == 0:
        return mesh

    neighbours = [[] for _ in range(len(faces))]
    for left, right in adjacency:
        neighbours[int(left)].append(int(right))
        neighbours[int(right)].append(int(left))

    edge_lengths = np.asarray(mesh.edges_unique_length)
    edge_lengths = edge_lengths[np.isfinite(edge_lengths) & (edge_lengths > 0)]
    if not len(edge_lengths):
        return mesh
    aabb_tolerance = float(np.median(edge_lengths))
    vertices = np.asarray(mesh.vertices)
    face_vertices = vertices[faces][:, :, constrained]
    reaches_aabb = np.any(
        aabb_limit - np.abs(face_vertices) <= aabb_tolerance,
        axis=(1, 2),
    )

    chain_faces = set()
    for seed in np.flatnonzero(reaches_aabb):
        seed = int(seed)
        if len(neighbours[seed]) > 1 or seed in chain_faces:
            continue
        previous = None
        current = seed
        while len(neighbours[current]) <= 2:
            chain_faces.add(current)
            onward = [face for face in neighbours[current] if face != previous]
            if len(onward) != 1:
                break
            previous, current = current, onward[0]

    if not chain_faces:
        return mesh
    chain_faces = np.fromiter(sorted(chain_faces), dtype=np.int64)
    if len(chain_faces) > max_drop_ratio * len(faces):
        print(f"Off-body bridge cleanup skipped: {len(chain_faces)} of {len(faces)} faces flagged")
        return mesh
    keep = np.ones(len(faces), dtype=bool)
    keep[chain_faces] = False
    mesh.update_faces(keep)
    mesh.remove_unreferenced_vertices()
    print(f"Removed {len(chain_faces)} off-body bridge-chain faces")
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

def inference(ckpt_path, item):
    print("-"*100)
    print("Loading model ............")
    with open("microsoft/TRELLIS.2-4B/pipeline.json", "r") as f:
        pipeline_config = json.load(f)
    pipeline_args = pipeline_config['args']
    tex_slat_flow_model = models.from_pretrained("microsoft/TRELLIS.2-4B/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16")

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
    if not item['2d_map']:
        # transforms.json holds a single calibrated camera; without an azimuth offset that
        # camera can easily land on the model's back, and the conditioning view decides
        # which side of the model the part colours are inferred from.
        render_from_transforms(item['glb'], item['transforms'], item['img'],
                               azimuths=[item.get('azimuth', 0.0)])
    image = Image.open(item['img'])
    image = preprocess_image(rembg_model, image)
    legend_encoder = load_legend_encoder(item['legend_ckpt']) if item.get('legend_ckpt') else None
    if legend_encoder is not None:
        image2 = preprocess_image(rembg_model, Image.open(item['img2'])) if item.get('img2') else None
        cond = get_cond_v3(image_cond_model, image, image2, legend_encoder, item.get('legend'))
    else:
        if item.get('legend') or item.get('img2'):
            print("warning: --legend/--img2 ignored: no --legend_ckpt with a legend encoder given")
        cond = get_cond(image_cond_model, [image])

    print("-"*100)
    print("Sampling .................")
    output_tex_slat = tex_slat_sample_single(gen3dseg, sampler, pipeline_args, shape_slat, tex_slat, cond)
    with torch.no_grad():
        tex_voxels = tex_decoder(output_tex_slat, guide_subs=subs) * 0.5 + 0.5

    print("-"*100)
    print("Exporting glb ............")
    glb = maybe_blender_reuv(slat_to_glb(meshes, tex_voxels), item)
    glb.export(item['export_glb'])

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
        "--img",
        type=str,
        required=True,
        help="Render image or 2D guidance map path.",
    )
    parser.add_argument(
        "--export_glb",
        type=str,
        required=True,
        help="Output glb path.",
    )
    parser.add_argument(
        "--two_d_map",
        action="store_true",
        help="Use 2D guidance map instead of rendering from transforms.",
    )
    parser.add_argument(
        "--transforms",
        type=str,
        help="Path to transforms.json (required if not using --two_d_map).",
    )
    parser.add_argument(
        "--azimuth",
        type=float,
        default=0.0,
        help="Degrees to orbit transforms.json's camera around the up axis when rendering the "
             "conditioning view (ignored with --two_d_map). transforms.json's own camera is not "
             "necessarily in front of the model: for data_toolkit/transforms.json and monk.glb, "
             "0 looks at its back and ~135 is head-on front.",
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

    parser.add_argument("--legend_ckpt", type=str, default=None,
                        help="v3: finetune lora_*.pt payload holding the legend encoder weights "
                             "(the LoRA itself must already be merged into --ckpt_path).")
    parser.add_argument("--legend", type=str, default=None,
                        help="v3: legend json with per-colour text vectors (sam3_to_2dmap *_legend.json).")
    parser.add_argument("--img2", type=str, default=None,
                        help="v3: second 2D map (another view, same colours) used as extra context.")

    args = parser.parse_args()
    if (not args.two_d_map) and args.transforms is None:
        parser.error("--transforms is required unless --two_d_map is set.")
    item = {
        "2d_map": args.two_d_map,
        "glb": os.path.abspath(args.glb),
        "input_vxz": os.path.abspath(args.input_vxz),
        "img": os.path.abspath(args.img),
        "export_glb": os.path.abspath(args.export_glb),
        "blender_reuv": args.blender_reuv,
        "rebake_texture_size": args.rebake_texture_size,
        "azimuth": args.azimuth,
        "legend_ckpt": os.path.abspath(args.legend_ckpt) if args.legend_ckpt else None,
        "legend": os.path.abspath(args.legend) if args.legend else None,
        "img2": os.path.abspath(args.img2) if args.img2 else None,
    }
    if not args.two_d_map:
        item["transforms"] = os.path.abspath(args.transforms)
    inference(os.path.abspath(args.ckpt_path), item)
