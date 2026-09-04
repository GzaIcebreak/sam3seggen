"""Gen3DSeg (full-segmentation flavour) exactly as inference_full.py builds it, plus loading helpers."""
from __future__ import annotations

import os
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

import common
import trellis2.modules.sparse as sp
from trellis2 import models

FLOW_MODEL = os.path.join(common.TRELLIS_DIR, "ckpts", "slat_flow_imgshape2tex_dit_1_3B_512_bf16")
DEFAULT_CKPT = os.path.join(common.ROOT, "ckpt", "full_seg_w_2d_map.ckpt")


class Gen3DSeg(nn.Module):
    """Mirror of inference_full.Gen3DSeg: interleaves noisy target and input texture tokens."""

    def __init__(self, flow_model):
        super().__init__()
        self.flow_model = flow_model

    def forward(self, x_t, tex_slats, shape_slats, t, cond, coords_len_list):
        tex_feats, tex_coords, shape_feats, shape_coords = [], [], [], []
        begin = 0
        for n in coords_len_list:
            end = begin + n
            tex_feats += [x_t.feats[begin:end], tex_slats.feats[begin:end]]
            tex_coords += [x_t.coords[begin:end], tex_slats.coords[begin:end]]
            shape_feats += [shape_slats.feats[begin:end], shape_slats.feats[begin:end]]
            shape_coords += [shape_slats.coords[begin:end], shape_slats.coords[begin:end]]
            begin = end
        x_in = sp.SparseTensor(torch.cat(tex_feats), torch.cat(tex_coords))
        shape_in = sp.SparseTensor(torch.cat(shape_feats), torch.cat(shape_coords))
        out = self.flow_model(x_in, t, cond, shape_in)
        feats, coords = [], []
        begin = 0
        for n in coords_len_list:
            feats.append(out.feats[begin:begin + n])
            coords.append(out.coords[begin:begin + n])
            begin += 2 * n
        return sp.SparseTensor(torch.cat(feats), torch.cat(coords))


class LegendEncoder(nn.Module):
    """v3 conditioning: [main-view DINO tokens (+e_view0), partner-view tokens (+e_view1),
    object token, one legend token per colour group].

    legend token = LN(W_t * text_256 + MLP(rgb) + e_legend); object token = LN(W_t * text + e_obj).
    DINO tokens are already per-token layer-normed (rms 1), so the LN keeps the new tokens on the
    same scale. The DiT cross-attention takes a variable-length context per sample (list of
    tensors), so no padding is involved.
    """

    def __init__(self, text_dim: int = 256, cond_dim: int = 1024, hidden: int = 256):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, cond_dim)
        self.color_mlp = nn.Sequential(nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, cond_dim))
        self.e_view = nn.Parameter(torch.zeros(2, cond_dim))
        self.e_legend = nn.Parameter(torch.zeros(cond_dim))
        self.e_obj = nn.Parameter(torch.zeros(cond_dim))
        self.norm = nn.LayerNorm(cond_dim)
        nn.init.normal_(self.text_proj.weight, std=0.02)
        nn.init.zeros_(self.text_proj.bias)
        nn.init.normal_(self.e_view, std=0.02)

    def forward(self, cond: torch.Tensor, cond_partner: torch.Tensor | None,
                legend_text: torch.Tensor | None, legend_rgb: torch.Tensor | None,
                obj_text: torch.Tensor | None, use_view_emb: bool = True) -> torch.Tensor:
        """All inputs for ONE sample; returns [L, cond_dim]."""
        blocks = [cond + self.e_view[0] if use_view_emb else cond]
        if cond_partner is not None:
            blocks.append(cond_partner + self.e_view[1])
        if obj_text is not None:
            blocks.append(self.norm(self.text_proj(obj_text) + self.e_obj).unsqueeze(0))
        if legend_text is not None and legend_text.shape[0] > 0:
            tok = self.text_proj(legend_text) + self.color_mlp(legend_rgb) + self.e_legend
            blocks.append(self.norm(tok))
        return torch.cat(blocks, 0)


def load_gen3dseg(ckpt_path: str = DEFAULT_CKPT, device: str = "cuda") -> Gen3DSeg:
    flow = models.from_pretrained(FLOW_MODEL)
    model = Gen3DSeg(flow)
    state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    state = OrderedDict((k.replace("gen3dseg.", ""), v) for k, v in state.items())
    model.load_state_dict(state)
    return model.to(device)


def save_gen3dseg_ckpt(model: Gen3DSeg, path: str) -> None:
    """Same container inference_full.py reads: {'state_dict': {'gen3dseg.<k>': tensor}}."""
    state = OrderedDict((f"gen3dseg.{k}", v.detach().cpu()) for k, v in model.state_dict().items())
    torch.save({"state_dict": state}, path)


def set_gradient_checkpointing(flow_model, enabled: bool = True) -> None:
    for block in flow_model.blocks:
        block.use_checkpoint = enabled
