"""Shared SAM3 helpers for the concept bank: prompt templates, text-embedding offsets, soft masks.

Lives in the SAM3 environment (.venv_holo): only torch / numpy / transformers.

A concept bank is a small file learned by concept_bank.py:
    {"E_0": [D], "names": [...], "E": [N, D], "template": "{name}", "base": "facebook/sam3"}
and is applied as   text_embeds.pooler_output += E_0 + E[name]   before the detector runs
(M2C, arXiv 2606.26711, adapted to a per-name + shared-offset bank). The SAM3 weights are
never touched; HF `Sam3Model.forward` accepts precomputed `text_embeds`.

v4 additions (all optional, absent in v3 files; see PLAN_concept_bank_v4.md):
    b       [N]     per-name bias on the detection logit (E)
    E_word  [W, D]  per-word offsets, name offset += sum of its words (C)
    tvec    [N, D]  unit text vectors of the names; out-of-vocabulary prompts borrow E from cosine
                    neighbours above `nn_cos` (C, inference-side fallback)
    ctx     [K, D]  learnable context tokens prepended to the text tokens (B)
    lr_down/lr_up   rank-r modulation pooled += pooled @ down @ up (B, optional)
G/H live outside the bank file: G is `--templates` (train sample / eval vote); H writes `decoder_lora.pt`.
`ConceptBank.apply` is the single place these are put on the text features; training and
`sam3_to_2dmap.segment_prompts` both go through it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

TEMPLATES = {
    "name": "{name}",
    "obj_name": "{object} {name}",
    "name_of_obj": "{name} of {object}",
    "the_name": "the {name}",
}


def fill_template(template: str, name: str, obj: str | None) -> str:
    if "{object}" in template and not obj:
        return name
    return template.format(name=name, object=obj or "")


def resolve_templates(spec: str) -> list[str]:
    """Comma-separated TEMPLATES keys or literal `{name}` / `{object}` strings."""
    out = []
    for t in spec.split(","):
        t = t.strip()
        if t:
            out.append(TEMPLATES.get(t, t))
    return out or [TEMPLATES["name"]]


def decoder_cross_attns(model, scope: str = "mask"):
    """SAM3 modules H can wrap: mask-decoder prompt cross-attn and/or DETR text cross-attn."""
    parts = {p.strip() for p in scope.split(",") if p.strip()}
    mods = []
    if "mask" in parts and getattr(model, "mask_decoder", None) is not None:
        mods.append(model.mask_decoder.prompt_cross_attn)
    if "text" in parts and getattr(model, "detr_decoder", None) is not None:
        for layer in model.detr_decoder.layers:
            mods.append(layer.text_cross_attn)
    return mods


def lora_targets(model, scope: str):
    """(parent module, attribute name) pairs to wrap. Scopes: mask / text (cross-attn q,k,v,o),
    embed (mask_embedder MLP = left side of the mask dot product), proj (instance_projection 1x1 conv =
    right side). See PLAN_concept_bank_v5.md section 2."""
    parts = {p.strip() for p in scope.split(",") if p.strip()}
    out = []
    for attn in decoder_cross_attns(model, scope):
        out += [(attn, n) for n in ("q_proj", "k_proj", "v_proj", "o_proj")]
    md = getattr(model, "mask_decoder", None)
    if md is not None:
        if "embed" in parts:
            out += [(md.mask_embedder.layers, str(i)) for i in range(len(md.mask_embedder.layers))]
        if "proj" in parts:
            out.append((md, "instance_projection"))
    return out


def inject_decoder_lora(model, r: int = 8, alpha: float | None = None, scope: str = "mask",
                        dropout: float = 0.0):
    """Wrap the chosen decoder linears / 1x1 convs with LoRA. Image encoder stays frozen."""
    from lora import LoRALinear, LoRAConv1x1
    alpha = float(alpha if alpha is not None else 2 * r)
    params = []
    n = 0
    for parent, name in lora_targets(model, scope):
        lin = getattr(parent, name, None)
        if isinstance(lin, torch.nn.Linear):
            wrapped = LoRALinear(lin, r, alpha, dropout)
        elif isinstance(lin, torch.nn.Conv2d) and lin.kernel_size == (1, 1):
            wrapped = LoRAConv1x1(lin, r, alpha)
        else:
            continue
        wrapped.to(lin.weight.device)
        setattr(parent, name, wrapped)
        params += [wrapped.lora_A, wrapped.lora_B]
        n += 1
    return params, n


def name_words(name: str) -> list[str]:
    return [w for w in name.lower().replace("-", " ").replace("_", " ").split() if w]


_OPTIONAL = ("b", "E_word", "tvec", "ctx", "lr_down", "lr_up", "bg")


@dataclass
class ConceptBank:
    E_0: torch.Tensor                      # [D]
    names: list[str]
    E: torch.Tensor                        # [N, D]
    template: str = "{name}"
    base: str = "facebook/sam3"
    meta: dict = field(default_factory=dict)
    b: torch.Tensor | None = None          # [N]   per-name detection-logit bias
    words: list[str] = field(default_factory=list)
    E_word: torch.Tensor | None = None     # [W, D]
    tvec: torch.Tensor | None = None       # [N, D] unit text vectors of `names` (for the OOV fallback)
    nn_cos: float = 0.0                    # OOV prompts borrow from names with cosine > nn_cos (0 = off)
    nn_temp: float = 0.05                  # softmax temperature over those cosines
    ctx: torch.Tensor | None = None        # [K, D] learnable context tokens
    lr_down: torch.Tensor | None = None    # [D, r]
    lr_up: torch.Tensor | None = None      # [r, D]
    bg: torch.Tensor | None = None         # []   v5 background logit of the per-pixel assignment softmax

    def index(self) -> dict[str, int]:
        return {n: i for i, n in enumerate(self.names)}

    def word_index(self) -> dict[str, int]:
        return {w: i for i, w in enumerate(self.words)}

    def parameters(self) -> list[torch.Tensor]:
        ps = [self.E_0, self.E]
        ps += [getattr(self, k) for k in ("b", "E_word", "ctx", "lr_down", "lr_up", "bg") if getattr(self, k) is not None]
        return ps

    def bg_logit(self) -> float:
        return float(self.bg.item()) if self.bg is not None else 0.0

    def detached(self) -> "ConceptBank":
        kw = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in self.__dict__.items()}
        return ConceptBank(**kw)

    # -- which rows of E / b a prompt uses: exact match, or (OOV) cosine neighbours ---------------
    def _rows(self, name: str, query_vec: torch.Tensor | None):
        i = self.index().get(name)
        if i is not None:
            return [i], None
        if self.nn_cos > 0 and self.tvec is not None and query_vec is not None:
            q = F.normalize(query_vec.float().to(self.tvec.device), dim=0)
            cos = self.tvec.float() @ q
            sel = torch.nonzero(cos > self.nn_cos).flatten()
            if sel.numel():
                w = torch.softmax(cos[sel] / self.nn_temp, dim=0)
                return sel.tolist(), w
        return [], None

    def offset(self, name: str, use_e0: bool = True, query_vec: torch.Tensor | None = None) -> torch.Tensor:
        off = self.E_0 if use_e0 else torch.zeros_like(self.E_0)
        rows, w = self._rows(name, query_vec)
        if rows:
            off = off + (self.E[rows[0]] if w is None else (w.to(self.E.dtype)[:, None] * self.E[rows]).sum(0))
        if self.E_word is not None and self.words:
            wi = self.word_index()
            ws = [wi[x] for x in name_words(name) if x in wi]
            if ws:
                off = off + self.E_word[ws].sum(0)
        return off

    def bias(self, name: str, query_vec: torch.Tensor | None = None) -> torch.Tensor:
        z = self.E_0.new_zeros(())
        if self.b is None:
            return z
        rows, w = self._rows(name, query_vec)
        if not rows:
            return z
        return self.b[rows[0]] if w is None else (w.to(self.b.dtype) * self.b[rows]).sum()

    def offsets(self, names: list[str], query_vecs: torch.Tensor | None = None, use_e0: bool = True) -> torch.Tensor:
        return torch.stack([self.offset(n, use_e0, None if query_vecs is None else query_vecs[k])
                            for k, n in enumerate(names)])

    def biases(self, names: list[str], query_vecs: torch.Tensor | None = None) -> torch.Tensor | None:
        if self.b is None:
            return None
        return torch.stack([self.bias(n, None if query_vecs is None else query_vecs[k]) for k, n in enumerate(names)])

    def apply(self, text_out, attention_mask, names: list[str], query_vecs: torch.Tensor | None = None,
              use_e0: bool = True):
        """Shift / extend the frozen text features for `names`. Returns (text_out', attention_mask', bias | None).

        `text_out.pooler_output` is [n, L, D]; the offset is broadcast over the L tokens, the optional
        low-rank term modulates every token, and the optional context tokens are prepended (mask padded
        with 1s). Differentiable w.r.t. the bank tensors; never mutates `text_out`."""
        pooled = text_out.pooler_output
        n = pooled.shape[0]
        off = self.offsets(names, query_vecs, use_e0)
        pooled = pooled + off.to(pooled.dtype).view(n, 1, -1)
        if self.lr_down is not None and self.lr_up is not None:
            pooled = pooled + ((pooled.float() @ self.lr_down.float()) @ self.lr_up.float()).to(pooled.dtype)
        am = attention_mask
        if self.ctx is not None and self.ctx.shape[0] > 0:
            k = self.ctx.shape[0]
            pooled = torch.cat([self.ctx.to(pooled.dtype).unsqueeze(0).expand(n, k, -1), pooled], 1)
            if am is not None:
                am = F.pad(am, (k, 0), value=1)
        shifted = text_out.__class__(**{k: v for k, v in text_out.items()})
        shifted.pooler_output = pooled
        return shifted, am, self.biases(names, query_vecs)

    def save(self, path: str) -> None:
        d = {"E_0": self.E_0.detach().cpu(), "names": list(self.names), "E": self.E.detach().cpu(),
             "template": self.template, "base": self.base, "meta": self.meta,
             "words": list(self.words), "nn_cos": self.nn_cos, "nn_temp": self.nn_temp}
        for k in _OPTIONAL:
            v = getattr(self, k)
            if v is not None:
                d[k] = v.detach().cpu()
        torch.save(d, path)

    @staticmethod
    def load(path: str, device: str = "cpu") -> "ConceptBank":
        d = torch.load(path, map_location=device, weights_only=False)
        kw = {k: d[k].to(device) for k in _OPTIONAL if d.get(k) is not None}
        return ConceptBank(E_0=d["E_0"].to(device), names=list(d["names"]), E=d["E"].to(device),
                           template=d.get("template", "{name}"), base=d.get("base", "facebook/sam3"),
                           meta=d.get("meta", {}), words=list(d.get("words", [])),
                           nn_cos=float(d.get("nn_cos", 0.0)), nn_temp=float(d.get("nn_temp", 0.05)), **kw)

    def describe(self) -> str:
        s = f"{len(self.names)} names, |E_0|={self.E_0.norm().item():.3f}"
        if self.b is not None:
            s += f", bias±{self.b.abs().mean().item():.2f}"
        if self.E_word is not None:
            s += f", {len(self.words)} words"
        if self.ctx is not None:
            s += f", ctx K={self.ctx.shape[0]}"
        if self.lr_down is not None:
            s += f", rank {self.lr_down.shape[1]}"
        if self.nn_cos > 0 and self.tvec is not None:
            s += f", nn fallback cos>{self.nn_cos:g}"
        if self.bg is not None:
            s += f", bg={self.bg.item():.2f}"
        return s


@torch.no_grad()
def encode_image(processor, model, image, device: str):
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    return model.get_vision_features(pixel_values=inputs["pixel_values"].to(device))


def text_features(processor, model, prompt: str, device: str):
    """Projected text features [1, L, D] plus attention mask; differentiable w.r.t. nothing (frozen)."""
    t = processor(text=prompt, return_tensors="pt")
    ids = t["input_ids"].to(device)
    am = t.get("attention_mask")
    am = am.to(device) if am is not None else None
    with torch.no_grad():
        out = model.get_text_features(input_ids=ids, attention_mask=am, return_dict=True)
    return out, am


def run_prompt(model, vision_embeds, text_out, attention_mask, offset: torch.Tensor | None = None):
    """Run the detector for one prompt. `offset` [D] is added to every text token (broadcast)."""
    pooled = text_out.pooler_output
    if offset is not None:
        pooled = pooled + offset.to(pooled.dtype).view(1, 1, -1)
    shifted = text_out.__class__(**{k: v for k, v in text_out.items()})   # never mutate the cached output
    shifted.pooler_output = pooled
    return model(vision_embeds=vision_embeds, text_embeds=shifted, attention_mask=attention_mask)


def expand_vision(vision_embeds, n: int):
    """View the single-image vision output as a batch of n identical images (no copy)."""
    if n == 1:
        return vision_embeds
    def ex(t):
        return t.expand(n, *t.shape[1:]) if torch.is_tensor(t) else t
    fields = {k: v for k, v in vision_embeds.items()}
    for k in ("fpn_hidden_states", "fpn_position_encoding"):
        if fields.get(k) is not None:
            fields[k] = tuple(ex(t) for t in fields[k])
    for k in ("last_hidden_state", "pooler_output"):
        if fields.get(k) is not None:
            fields[k] = ex(fields[k])
    return vision_embeds.__class__(**fields)


def text_features_batch(processor, model, prompts: list[str], device: str):
    """Padded batch of projected text features [N, L, D] + attention mask [N, L] (frozen encoder)."""
    t = processor(text=prompts, padding=True, return_tensors="pt")
    ids = t["input_ids"].to(device)
    am = t["attention_mask"].to(device)
    with torch.no_grad():
        out = model.get_text_features(input_ids=ids, attention_mask=am, return_dict=True)
    return out, am


def run_prompts(model, vision_embeds, text_out, attention_mask, offsets: torch.Tensor | None = None):
    """Batched detector run: N prompts on one image. `offsets` [N, D] (may require grad)."""
    n = text_out.pooler_output.shape[0]
    pooled = text_out.pooler_output
    if offsets is not None:
        pooled = pooled + offsets.to(pooled.dtype).view(n, 1, -1)
    shifted = text_out.__class__(**{k: v for k, v in text_out.items()})
    shifted.pooler_output = pooled
    return model(vision_embeds=expand_vision(vision_embeds, n), text_embeds=shifted, attention_mask=attention_mask)


def text_query_vecs(text_out, attention_mask) -> torch.Tensor:
    """Mean over valid tokens of the (un-shifted) projected text features, [N, D]; the OOV fallback key."""
    pooled = text_out.pooler_output.float()
    if attention_mask is None:
        return pooled.mean(1)
    m = attention_mask.float().unsqueeze(-1)
    return (pooled * m).sum(1) / m.sum(1).clamp(min=1)


def bank_forward(model, vision_embeds, text_out, attention_mask, bank: "ConceptBank | None", names: list[str],
                 query_vecs: torch.Tensor | None = None, use_e0: bool = True):
    """Batched detector run through `bank.apply`. Returns (outputs, bias | None); bias is [N]."""
    n = text_out.pooler_output.shape[0]
    bias = None
    if bank is not None:
        text_out, attention_mask, bias = bank.apply(text_out, attention_mask, names, query_vecs, use_e0)
    out = model(vision_embeds=expand_vision(vision_embeds, n), text_embeds=text_out, attention_mask=attention_mask)
    return out, bias


def batch_scores(outputs, bias: torch.Tensor | None = None) -> torch.Tensor:
    """[N, Q] final scores for a batched run; `bias` [N] shifts the detection logits per prompt."""
    logits = outputs.pred_logits
    if bias is not None:
        logits = logits + bias.to(logits.dtype).view(-1, 1)
    s = logits.sigmoid()
    if outputs.presence_logits is not None:
        s = s * outputs.presence_logits.sigmoid()
    return s


def batch_union_masks(outputs, size: tuple[int, int], threshold: float = 0.3, mask_threshold: float = 0.5,
                      bias: torch.Tensor | None = None) -> torch.Tensor:
    """Hard unions for a batched run, bool [N, H, W]."""
    scores = batch_scores(outputs, bias)
    n = scores.shape[0]
    out = torch.zeros((n, *size), dtype=torch.bool, device=scores.device)
    for i in range(n):
        keep = scores[i] > threshold
        if keep.any():
            m = F.interpolate(outputs.pred_masks[i][keep].sigmoid().unsqueeze(0), size=size,
                              mode="bilinear", align_corners=False)[0]
            out[i] = (m > mask_threshold).any(0)
    return out


def batch_soft_union(outputs, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Differentiable unions at native mask resolution, float [N, h, w]."""
    scores = batch_scores(outputs, bias)                             # [N, Q]
    m = outputs.pred_masks.sigmoid()                                 # [N, Q, h, w]
    p = (m * scores[:, :, None, None]).clamp(max=1 - 1e-6)
    return 1.0 - torch.exp(torch.log1p(-p).sum(1))


def name_logit_maps(outputs, bias: torch.Tensor | None = None,
                    log_weight: torch.Tensor | None = None) -> torch.Tensor:
    """v5: one pixel logit per prompt, float [N, h, w] at the mask head's resolution:
        L_n(p) = logsumexp_q [ mask_logit_{n,q}(p) + log score_{n,q} + log_weight_{n,q} ]
    Raw mask logits (the query . pixel dot product) rather than sigmoid(mask) and a log-score weight
    rather than a hard `score > t` gate, so the gradient reaches both ends of the dot product and a
    query that is not detected simply drops out of the competition. Multiple instances of one name
    (four legs) merge through the logsumexp.

    `log_weight` [N, Q] is the Mask RankGNN's contribution (lambda * log keep); None keeps the score
    ordering SAM3 assigned to each query on its own."""
    scores = batch_scores(outputs, bias).float().clamp(min=1e-6)      # [N, Q]
    L = outputs.pred_masks.float() + scores.log()[:, :, None, None]
    if log_weight is not None:
        L = L + log_weight.float()[:, :, None, None]
    return torch.logsumexp(L, dim=1)


def assign_probs(maps: torch.Tensor, bg_logit: float, size: tuple[int, int] | None = None,
                 temp: float = 1.0) -> torch.Tensor:
    """Per-pixel softmax over [prompts..., background], float [N + 1, H, W]."""
    if size is not None and tuple(maps.shape[-2:]) != tuple(size):
        maps = F.interpolate(maps[None], size=size, mode="bilinear", align_corners=False)[0]
    L = torch.cat([maps, torch.full_like(maps[:1], float(bg_logit))], 0) / temp
    return torch.softmax(L, dim=0)


def paint_argmax(maps: torch.Tensor, fg: torch.Tensor, bg_logit: float, tau: float = 0.0,
                 temp: float = 1.0) -> torch.Tensor:
    """v5 deployment operator: every silhouette pixel goes to the prompt with the highest assignment
    probability; -1 (grey) where the background wins or the winner's probability is below `tau`.
    Returns long [H, W] like `paint`."""
    n = maps.shape[0]
    out = torch.full(fg.shape, -1, dtype=torch.long, device=maps.device)
    if n == 0:
        return out
    p = assign_probs(maps, bg_logit, tuple(fg.shape), temp)
    conf, lab = p.max(0)
    keep = fg & (lab < n) & (conf >= tau)
    out[keep] = lab[keep]
    return out


def paint(unions: torch.Tensor, fg: torch.Tensor) -> torch.Tensor:
    """What sam3_to_2dmap.colorize does with one union mask per prompt: clip to the silhouette and paint
    the smallest masks first, each only where nothing was painted yet. Returns the per-pixel prompt
    index, long [H, W], -1 = unassigned (grey downstream)."""
    n = unions.shape[0]
    out = torch.full(fg.shape, -1, dtype=torch.long, device=unions.device)
    if n == 0:
        return out
    order = torch.argsort(unions.flatten(1).sum(1))
    for i in order.tolist():
        free = unions[i] & fg & (out < 0)
        out[free] = i
    return out


def instance_scores(outputs) -> torch.Tensor:
    """[Q] final per-query scores = sigmoid(logit) * sigmoid(presence), as HF post-processing does."""
    s = outputs.pred_logits.sigmoid()[0]
    if outputs.presence_logits is not None:
        s = s * outputs.presence_logits.sigmoid()[0]
    return s


def union_mask(outputs, size: tuple[int, int], threshold: float = 0.3, mask_threshold: float = 0.5) -> torch.Tensor:
    """Hard union of kept instances at `size`, bool [H, W]. Mirrors sam3_to_2dmap.segment_prompts."""
    scores = instance_scores(outputs)
    keep = scores > threshold
    if not keep.any():
        return torch.zeros(size, dtype=torch.bool, device=scores.device)
    m = outputs.pred_masks[0][keep].sigmoid().unsqueeze(0)
    m = F.interpolate(m, size=size, mode="bilinear", align_corners=False)[0]
    return (m > mask_threshold).any(0)


def soft_union(outputs, size: tuple[int, int], score_temp: float = 1.0) -> torch.Tensor:
    """Differentiable union: 1 - prod_q (1 - sigmoid(mask_q) * score_q), float [H, W]."""
    scores = instance_scores(outputs)
    if score_temp != 1.0:
        scores = scores.pow(score_temp)
    m = outputs.pred_masks[0].sigmoid().unsqueeze(0)
    m = F.interpolate(m, size=size, mode="bilinear", align_corners=False)[0]      # [Q, H, W]
    p = m * scores.view(-1, 1, 1)
    log_keep = torch.log1p(-p.clamp(max=1 - 1e-6)).sum(0)
    return 1.0 - torch.exp(log_keep)


# --------------------------------------------------------------------------- metrics

def iou(a: torch.Tensor, b: torch.Tensor) -> float:
    inter = (a & b).sum().item()
    union = (a | b).sum().item()
    return inter / union if union else 0.0


def _boundary(mask: torch.Tensor) -> torch.Tensor:
    m = mask.float()[None, None]
    eroded = -F.max_pool2d(-m, 3, stride=1, padding=1)
    return ((m - eroded) > 0)[0, 0]


def boundary_f1(pred: torch.Tensor, gt: torch.Tensor, tol: int = 5) -> float:
    """F1 of boundary pixels within `tol` px (Perazzi-style), 0 if either boundary is empty."""
    bp, bg = _boundary(pred), _boundary(gt)
    if bp.sum() == 0 or bg.sum() == 0:
        return 0.0
    k = 2 * tol + 1
    bp_d = F.max_pool2d(bp.float()[None, None], k, stride=1, padding=tol)[0, 0] > 0
    bg_d = F.max_pool2d(bg.float()[None, None], k, stride=1, padding=tol)[0, 0] > 0
    prec = (bp & bg_d).sum().item() / bp.sum().item()
    rec = (bg & bp_d).sum().item() / bg.sum().item()
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


# --------------------------------------------------------------------------- dataset access

def load_object_labels(obj_dir: str):
    """names.json, names_meta.json (object name + uncertain flags)."""
    import json
    with open(os.path.join(obj_dir, "names.json"), encoding="utf-8") as f:
        names = json.load(f)
    meta_path = os.path.join(obj_dir, "names_meta.json")
    obj_name, uncertain = None, [False] * len(names)
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            m = json.load(f)
        obj_name = (m.get("object") or "").strip() or None
        uncertain = list(m.get("uncertain", uncertain))
    return names, obj_name, uncertain


def gt_unions(ids: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    """name -> bool mask over all visible parts carrying that name (empty names skipped)."""
    out: dict[str, np.ndarray] = {}
    for p, n in enumerate(names):
        n = (n or "").strip()
        if not n:
            continue
        m = ids == p
        if not m.any():
            continue
        out[n] = out[n] | m if n in out else m
    return out


def vocabulary(dataset_root: str) -> dict[str, int]:
    import json
    from collections import Counter
    c: Counter = Counter()
    for o in os.listdir(dataset_root):
        p = os.path.join(dataset_root, o, "names.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                c.update(n.strip() for n in json.load(f) if n and n.strip())
    return dict(c)
