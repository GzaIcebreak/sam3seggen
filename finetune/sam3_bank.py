"""Shared SAM3 helpers for the concept bank: prompt templates, text-embedding offsets, soft masks.

Lives in the SAM3 environment (.venv_holo): only torch / numpy / transformers.

A concept bank is a small file learned by concept_bank.py:
    {"E_0": [D], "names": [...], "E": [N, D], "template": "{name}", "base": "facebook/sam3"}
and is applied as   text_embeds.pooler_output += E_0 + E[name]   before the detector runs
(M2C, arXiv 2606.26711, adapted to a per-name + shared-offset bank). The SAM3 weights are
never touched; HF `Sam3Model.forward` accepts precomputed `text_embeds`.
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
}


def fill_template(template: str, name: str, obj: str | None) -> str:
    if "{object}" in template and not obj:
        return name
    return template.format(name=name, object=obj or "")


@dataclass
class ConceptBank:
    E_0: torch.Tensor                      # [D]
    names: list[str]
    E: torch.Tensor                        # [N, D]
    template: str = "{name}"
    base: str = "facebook/sam3"
    meta: dict = field(default_factory=dict)

    def index(self) -> dict[str, int]:
        return {n: i for i, n in enumerate(self.names)}

    def offset(self, name: str, use_e0: bool = True) -> torch.Tensor:
        idx = self.index().get(name)
        off = self.E_0 if use_e0 else torch.zeros_like(self.E_0)
        if idx is not None:
            off = off + self.E[idx]
        return off

    def save(self, path: str) -> None:
        torch.save({"E_0": self.E_0.detach().cpu(), "names": list(self.names), "E": self.E.detach().cpu(),
                    "template": self.template, "base": self.base, "meta": self.meta}, path)

    @staticmethod
    def load(path: str, device: str = "cpu") -> "ConceptBank":
        d = torch.load(path, map_location=device)
        return ConceptBank(E_0=d["E_0"].to(device), names=list(d["names"]), E=d["E"].to(device),
                           template=d.get("template", "{name}"), base=d.get("base", "facebook/sam3"),
                           meta=d.get("meta", {}))


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


def batch_scores(outputs) -> torch.Tensor:
    """[N, Q] final scores for a batched run."""
    s = outputs.pred_logits.sigmoid()
    if outputs.presence_logits is not None:
        s = s * outputs.presence_logits.sigmoid()
    return s


def batch_union_masks(outputs, size: tuple[int, int], threshold: float = 0.3, mask_threshold: float = 0.5) -> torch.Tensor:
    """Hard unions for a batched run, bool [N, H, W]."""
    scores = batch_scores(outputs)
    n = scores.shape[0]
    out = torch.zeros((n, *size), dtype=torch.bool, device=scores.device)
    for i in range(n):
        keep = scores[i] > threshold
        if keep.any():
            m = F.interpolate(outputs.pred_masks[i][keep].sigmoid().unsqueeze(0), size=size,
                              mode="bilinear", align_corners=False)[0]
            out[i] = (m > mask_threshold).any(0)
    return out


def batch_soft_union(outputs) -> torch.Tensor:
    """Differentiable unions at native mask resolution, float [N, h, w]."""
    scores = batch_scores(outputs)                                   # [N, Q]
    m = outputs.pred_masks.sigmoid()                                 # [N, Q, h, w]
    p = (m * scores[:, :, None, None]).clamp(max=1 - 1e-6)
    return 1.0 - torch.exp(torch.log1p(-p).sum(1))


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
