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

    v5: image token += W_tok * text_256(part under that patch). W_tok is zero-initialised and has
    no bias, so unnamed tokens (CLS, registers, background, unbound pixels) are never touched and
    the model equals the base at step 0. Delivering the name at the patch instead of in a global
    legend turns "which part is this?" from geometry recognition into a local look-up.
    """

    def __init__(self, text_dim: int = 256, cond_dim: int = 1024, hidden: int = 256):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, cond_dim)
        self.color_mlp = nn.Sequential(nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, cond_dim))
        self.tok_proj = nn.Linear(text_dim, cond_dim, bias=False)
        self.e_view = nn.Parameter(torch.zeros(2, cond_dim))
        self.e_legend = nn.Parameter(torch.zeros(cond_dim))
        self.e_obj = nn.Parameter(torch.zeros(cond_dim))
        self.norm = nn.LayerNorm(cond_dim)
        nn.init.normal_(self.text_proj.weight, std=0.02)
        nn.init.zeros_(self.text_proj.bias)
        nn.init.zeros_(self.tok_proj.weight)
        nn.init.normal_(self.e_view, std=0.02)

    def load_state_dict(self, state_dict, strict: bool = True):
        # v3/v4 payloads predate tok_proj; keep it at zero (= no per-token text) for them
        if "tok_proj.weight" not in state_dict:
            state_dict = dict(state_dict)
            state_dict["tok_proj.weight"] = torch.zeros_like(self.tok_proj.weight)
        return super().load_state_dict(state_dict, strict)

    def tok_gain(self) -> float:
        w = self.tok_proj.weight.detach()
        return float(w.norm() / w.shape[0] ** 0.5)

    def tokens(self, cond: torch.Tensor, cond_partner: torch.Tensor | None,
               legend_text: torch.Tensor | None, legend_rgb: torch.Tensor | None,
               obj_text: torch.Tensor | None, use_view_emb: bool = True,
               cond_text: torch.Tensor | None = None, partner_text: torch.Tensor | None = None):
        """ONE sample -> (image tokens [T(+T), C], legend tokens [1+G, C] or None).
        cond_text / partner_text: [T, text_dim] per-token names (v5), zero rows = no name."""
        main = cond + self.e_view[0] if use_view_emb else cond
        if cond_text is not None:
            main = main + self.tok_proj(cond_text)
        image = [main]
        if cond_partner is not None:
            second = cond_partner + self.e_view[1]
            if partner_text is not None:
                second = second + self.tok_proj(partner_text)
            image.append(second)
        legend = []
        if obj_text is not None:
            legend.append(self.norm(self.text_proj(obj_text) + self.e_obj).unsqueeze(0))
        if legend_text is not None and legend_text.shape[0] > 0:
            tok = self.text_proj(legend_text) + self.color_mlp(legend_rgb) + self.e_legend
            legend.append(self.norm(tok))
        return torch.cat(image, 0), (torch.cat(legend, 0) if legend else None)

    def forward(self, cond: torch.Tensor, cond_partner: torch.Tensor | None,
                legend_text: torch.Tensor | None, legend_rgb: torch.Tensor | None,
                obj_text: torch.Tensor | None, use_view_emb: bool = True,
                cond_text: torch.Tensor | None = None, partner_text: torch.Tensor | None = None) -> torch.Tensor:
        """v3 concat conditioning: all tokens in one context [L, cond_dim]."""
        image, legend = self.tokens(cond, cond_partner, legend_text, legend_rgb, obj_text, use_view_emb,
                                    cond_text, partner_text)
        return image if legend is None else torch.cat([image, legend], 0)


class LegendCrossAttention(nn.Module):
    """v4: decoupled (IP-Adapter style) cross-attention for the legend tokens.

    Wraps a block's cross_attn: h = inner(x, image_ctx) + to_out_l(Attn(q(x), K_l, V_l)) where
    K_l/V_l come from the legend tokens through this module's own to_kv (initialised from the image
    to_kv, trainable) and to_out_l is a ZERO-initialised linear (ControlNet-style "zero conv"), so
    the wrapped model equals the base model at step 0. A zero matrix rather than a scalar gate: a
    scalar only moves if the raw attention output happens to correlate with the needed direction and
    it starved everything upstream of gradient (pv_v4 gate-0 run: |tanh(gate)| stayed < 0.03 for 500
    steps), whereas a zero linear learns a useful read-out of whatever the attention produces and
    passes gradient upstream from the first update on. The query is shared with the image path.
    The legend context is set per forward on every wrapper (set_legend_context) because the DiT
    forward signature carries a single `cond`.
    """

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        base_kv = inner.to_kv.base if hasattr(inner.to_kv, "base") else inner.to_kv
        base_out = inner.to_out.base if hasattr(inner.to_out, "base") else inner.to_out
        # fp32 master weights (the torso is bf16, whose 8-bit mantissa would swallow 1e-4 updates)
        self.to_kv = nn.Linear(base_kv.in_features, base_kv.out_features, bias=base_kv.bias is not None)
        with torch.no_grad():
            self.to_kv.weight.copy_(base_kv.weight.float())
            if base_kv.bias is not None:
                self.to_kv.bias.copy_(base_kv.bias.float())
        self.to_out = nn.Linear(base_out.in_features, base_out.out_features, bias=True)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)
        self.legend_ctx = None  # VarLenTensor set by set_legend_context, None = base behaviour

    def gain(self) -> float:
        """Spectral-ish size of the legend read-out (0 at init)."""
        return float(self.to_out.weight.norm() / max(1, self.to_out.weight.shape[0]) ** 0.5)

    def forward(self, x, context):
        h = self.inner(x, context)
        ctx = self.legend_ctx
        if ctx is None:
            return h
        from trellis2.modules.sparse.attention.full_attn import sparse_scaled_dot_product_attention
        inner = self.inner
        q = inner._linear(inner.to_q, x)
        q = inner._reshape_chs(q, (inner.num_heads, -1))
        kv = ctx.replace(self.to_kv(ctx.feats.float()).to(ctx.dtype))
        kv = inner._fused_pre(kv, num_fused=2)
        if inner.qk_rms_norm:
            q = inner.q_rms_norm(q)
            k, v = kv.unbind(dim=-3)
            k = inner.k_rms_norm(k)
            h2 = sparse_scaled_dot_product_attention(q, k, v)
        else:
            h2 = sparse_scaled_dot_product_attention(q, kv)
        h2 = inner._reshape_chs(h2, (-1,))
        h2 = h2.replace(self.to_out(h2.feats.float()).to(h.dtype))
        return h + h2


def inject_legend_attention(flow_model) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Wrap every block's cross_attn (after LoRA injection).
    Returns (K/V params initialised from the image path, zero-initialised read-out params)."""
    kv_params, out_params = [], []
    for block in flow_model.blocks:
        if isinstance(block.cross_attn, LegendCrossAttention):
            continue
        wrapper = LegendCrossAttention(block.cross_attn)
        block.cross_attn = wrapper
        kv_params += list(wrapper.to_kv.parameters())
        out_params += list(wrapper.to_out.parameters())
    return kv_params, out_params


def set_legend_context(flow_model, legend: list[torch.Tensor | None] | None) -> None:
    """legend: per-sample [G_i, C] tokens (None entries -> one zero token, i.e. 'no legend').
    Passing None disables the legend path (pure base behaviour)."""
    ctx = None
    if legend is not None and any(t is not None for t in legend):
        dim = next(t for t in legend if t is not None).shape[-1]
        dev = next(t for t in legend if t is not None).device
        ctx = sp.VarLenTensor.from_tensor_list([t if t is not None else torch.zeros(1, dim, device=dev) for t in legend])
        ctx = ctx.replace(ctx.feats.to(flow_model.dtype))
    for block in flow_model.blocks:
        if isinstance(block.cross_attn, LegendCrossAttention):
            block.cross_attn.legend_ctx = ctx


def legend_attn_state_dict(model: nn.Module) -> dict:
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if (".cross_attn.to_kv." in k or ".cross_attn.to_out." in k) and ".inner." not in k}


def load_legend_attn_state(model: nn.Module, state: dict) -> None:
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise KeyError(f"legend-attention keys not in model: {result.unexpected_keys[:5]}")
    expected = set(legend_attn_state_dict(model))
    if expected != set(state):
        raise KeyError(f"legend-attention key mismatch: {len(expected - set(state))} missing, {len(set(state) - expected)} extra")


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
