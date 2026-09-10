"""Stage B: learn a SAM3 concept bank (shared offset E_0 + per-name offsets E) from GT part masks.

    <.venv_holo python> finetune/concept_bank.py --dataset_root E:\\data\\pv --out E:\\data\\concept_bank \\
        --template name --epochs 3 --holdout_file E:\\data\\pv_holdout_mix.txt

M2C (arXiv 2606.26711) with a two-level bank: every prompt's projected text features get
E_0 + E[name] added; SAM3 itself is frozen (no gradients into any weight). Positives are the
visible, non-uncertain part names of an image (target = union of the parts with that name);
negatives are frequent names from other objects (target = empty, presence = 0).

Loss per image (v3) = mean over positives of [BCE + Dice on the soft union]
                    + 0.5 * mean over negatives of BCE(soft union, 0)
                    + 0.5 * presence BCE over all prompts.

v4 options (PLAN_concept_bank_v4.md; every one is off by default so the defaults reproduce v3):
    A  --pixel_ce W --margin W   per-pixel softmax over the prompted names (+ background) with the GT
                                 part id as label, class-balanced; hinge margin against the negatives
    B  --ctx K --lowrank r       learnable context tokens / low-rank modulation of the text features
    C  --word_offsets --nn_cos   per-word offsets; OOV prompts borrow E from cosine neighbours at eval
    D  --online_hard             re-mine the negatives after every epoch from the bank's actual false hits
    E  --name_bias               per-name bias on the detection logit
    F  (always on)               colorize-style metrics: pixel_acc / pixel_wrong / pixel_unassigned /
                                 painted_part_rate, per-slice (--hard_file) and on extra sets (--extra_eval)
    G  --templates a,b,c         train: sample one template per image; eval: majority-vote the hard unions
    H  --decoder_lora r          LoRA on SAM3 mask-decoder (and optional DETR text) cross-attention;
                                 vision encoder stays frozen. Writes decoder_lora.pt next to bank.pt

v5 (PLAN_concept_bank_v5.md): --assign_ce W replaces the per-prompt objective by a per-pixel assignment
    CE over the name-level mask logits (logsumexp over queries of mask_logit + log score) plus a learnable
    background logit `bg`; BCE+Dice and the negatives' BCE default to 0, presence BCE stays. Every eval
    also reports the "argmax" block: the same pass painted by per-pixel argmax (sam3_bank.paint_argmax),
    which is the v5 deployment operator. --lora_scope embed,proj put LoRA on both ends of the mask dot product.

Writes <out>/bank.pt (loadable by sam3_bank.ConceptBank), <out>/split.json, <out>/log.jsonl,
and <out>/text_cache.pt: one pooled 256-d vector per vocabulary name and object name, with the
bank offsets applied, for the SegviGen legend tokens (Stage D).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import zlib
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import sam3_bank as sb
from sam3_to_2dmap import load_sam3


# --------------------------------------------------------------------------- data

def list_images(dataset_root: str, objects: list[str], azimuths: list[str]) -> list[tuple[str, str]]:
    out = []
    for o in objects:
        for az in azimuths:
            v = os.path.join(dataset_root, o, "views", az)
            if os.path.exists(os.path.join(v, "render.png")) and os.path.exists(os.path.join(v, "ids.npy")):
                out.append((o, az))
    return out


def list_objects(dataset_root: str) -> list[str]:
    return sorted(d for d in os.listdir(dataset_root)
                  if os.path.exists(os.path.join(dataset_root, d, "views", "az0", "ids.npy")))


def make_split(dataset_root: str, holdout_file: str | None, holdout_frac: float, seed: int):
    objs = list_objects(dataset_root)
    fixed = set()
    if holdout_file and os.path.exists(holdout_file):
        with open(holdout_file, encoding="utf-8") as f:
            fixed = {l.strip() for l in f if l.strip() and not l.startswith("#")} & set(objs)
    rng = random.Random(seed)
    rest = [o for o in objs if o not in fixed]
    rng.shuffle(rest)
    n_hold = max(0, int(round(holdout_frac * len(objs))) - len(fixed))
    holdout = sorted(fixed | set(rest[:n_hold]))
    train = sorted(set(objs) - set(holdout))
    return train, holdout


def read_id_file(path: str | None) -> set[str]:
    if not path or not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {l.strip() for l in f if l.strip() and not l.startswith("#")}


class ImageSample:
    __slots__ = ("root", "obj", "az", "image", "ids", "names", "obj_name", "uncertain", "gts", "pos_names", "bad")

    def __init__(self, dataset_root: str, obj: str, az: str, skip_uncertain: bool):
        v = os.path.join(dataset_root, obj, "views", az)
        self.root, self.obj, self.az = dataset_root, obj, az
        self.image = Image.open(os.path.join(v, "render.png"))
        self.ids = np.load(os.path.join(v, "ids.npy"))
        self.names, self.obj_name, self.uncertain = sb.load_object_labels(os.path.join(dataset_root, obj))
        self.gts = sb.gt_unions(self.ids, self.names)
        self.bad = set()
        if skip_uncertain:
            for p, n in enumerate(self.names):
                if p < len(self.uncertain) and self.uncertain[p] and n.strip():
                    self.bad.add(n.strip())
        self.pos_names = [n for n in self.gts if n not in self.bad]


# --------------------------------------------------------------------------- loss

def soft_gt(gt: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Area-downsampled GT so the loss lives at the mask head's resolution."""
    return F.interpolate(gt.float()[None, None], size=size, mode="area")[0, 0]


def bce_dice(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    p = pred.clamp(1e-6, 1 - 1e-6)
    bce = F.binary_cross_entropy(p, gt)
    inter = (p * gt).sum()
    dice = 1 - (2 * inter + eps) / (p.sum() + gt.sum() + eps)
    return bce + dice


def pixel_labels(s: ImageSample, pos_names: list[str], size: tuple[int, int], device: str) -> torch.Tensor:
    """Per-pixel class at the mask resolution: k for pos_names[k]; len(pos_names) for background and for
    parts that were not prompted (downstream they must stay grey); -1 (ignored) for uncertain names."""
    n_pos = len(pos_names)
    pi = {n: k for k, n in enumerate(pos_names)}
    lut = torch.full((max(1, len(s.names)),), n_pos, dtype=torch.long)
    for p, nm in enumerate(s.names):
        nm = (nm or "").strip()
        if nm in pi:
            lut[p] = pi[nm]
        elif nm in s.bad:
            lut[p] = -1
    ids = torch.from_numpy(s.ids.astype(np.int64)).to(device)
    small = F.interpolate(ids[None, None].float(), size=size, mode="nearest")[0, 0].long()
    lut = lut.to(device)
    return torch.where(small < 0, torch.full_like(small, n_pos), lut[small.clamp(min=0, max=len(lut) - 1)])


def competition_losses(union: torch.Tensor, label: torch.Tensor, n_pos: int, temp: float, bg_logit: float,
                       margin_m: float, mode: str = "all", balance: str = "inv") -> tuple[torch.Tensor, torch.Tensor]:
    """A: class-balanced per-pixel cross-entropy over [prompted names..., background] using logit(union)
    as the per-name score, plus a hinge margin: at every pixel of a positive part, every negative name
    must score below the true name by `margin_m`. Both are what colorize's overlay actually needs
    (a ranking between names), which BCE per prompt never asks for.

    mode "painted" restricts the CE to pixels where at least one name would be painted (union > 0.5),
    i.e. to the ranking question only; "not found" stays the BCE's job. balance "sqrt" weights classes by
    1/sqrt(count) instead of 1/count so a 30-pixel part cannot own the whole loss."""
    u = union.clamp(1e-4, 1 - 1e-4)
    logits = (torch.log(u) - torch.log1p(-u)) / temp                       # [N, h, w]
    n = logits.shape[0]
    L = torch.cat([logits, torch.full_like(logits[:1], bg_logit)], 0)     # [N + 1, h, w]; class N = background
    lab = torch.where(label == n_pos, torch.full_like(label, n), label)    # unprompted / background -> class N
    valid = lab >= 0
    if mode == "painted":
        valid = valid & (union.max(0).values > 0.5)
    ce_map = F.cross_entropy(L[None], lab.clamp(min=0)[None], reduction="none")[0]
    counts = torch.bincount(lab[valid], minlength=n + 1).float()
    if balance == "sqrt":
        counts = counts.sqrt()
    w = torch.zeros_like(ce_map)
    w[valid] = 1.0 / counts[lab[valid]]
    ce = (w * ce_map).sum() / w.sum().clamp(min=1e-6)
    margin = union.new_zeros(())
    if n > n_pos and margin_m > 0:
        fg = valid & (lab < n_pos)
        if bool(fg.any()):
            Lf = L[:, fg]                                                  # [N + 1, P]
            tgt = Lf.gather(0, lab[fg][None])                              # [1, P]
            margin = F.relu(margin_m + Lf[n_pos:n] - tgt).mean()
    return ce, margin


def assign_loss(maps: torch.Tensor, bg: torch.Tensor, label: torch.Tensor, n_pos: int, temp: float,
                syn: torch.Tensor | None, balance: str = "sqrt") -> torch.Tensor:
    """v5: per-pixel assignment cross-entropy over [prompted names..., background] using the name-level
    mask logits (sam3_bank.name_logit_maps) directly, no union in between.

    `label` from pixel_labels: k = pos_names[k], n_pos = background / unprompted part, -1 = ignored.
    Negatives (rows >= n_pos) are never a label, so the softmax pushes them down on every pixel: that is
    the direct penalty for "painting body onto torso" which BCE per prompt never sees.
    `syn` [N, N] bool: names that are near-synonyms of each other (text cosine > --syn_cos). Pixels of
    name k credit the logsumexp of k and its synonyms, so a body/torso labelling disagreement is not
    trained as an error. Classes are balanced by 1/sqrt(count) (or 1/count with balance="inv")."""
    n = maps.shape[0]
    L = torch.cat([maps, bg.to(maps.dtype).view(1, 1, 1).expand(1, *maps.shape[-2:])], 0) / temp   # [N + 1, h, w]
    lse = torch.logsumexp(L, dim=0)                                                             # [h, w]
    lab = torch.where(label == n_pos, torch.full_like(label, n), label)
    valid = lab >= 0
    if not bool(valid.any()):
        return maps.new_zeros(())
    if syn is not None and n_pos > 0 and bool(syn.any()):
        # target logit of name k = logsumexp over k and its synonyms (among all prompts)
        grp = syn.clone()
        grp.fill_diagonal_(True)
        masked = L[:n][None].expand(n, n, *L.shape[-2:]).masked_fill(~grp[:, :, None, None], float("-inf"))
        tgt_pos = torch.logsumexp(masked, dim=1)                                                # [N, h, w]
        Lt = torch.cat([tgt_pos, L[n:]], 0)
    else:
        Lt = L
    nll = lse - Lt.gather(0, lab.clamp(min=0)[None])[0]
    counts = torch.bincount(lab[valid], minlength=n + 1).float()
    if balance == "sqrt":
        counts = counts.sqrt()
    w = torch.zeros_like(nll)
    w[valid] = 1.0 / counts[lab[valid]]
    return (w * nll).sum() / w.sum().clamp(min=1e-6)


# --------------------------------------------------------------------------- eval

def _new_acc() -> dict:
    return {"iou": [], "bind": [], "fp": [], "hfp": [], "pb": [], "px": 0, "grey": 0,
            "cpx": 0, "cacc": 0, "cwrong": 0, "cpart": [], "images": 0}


def _summary(a: dict) -> dict:
    cpx = max(1, a["cpx"])
    return {"images": a["images"], "prompts": len(a["iou"]),
            "miou": float(np.mean(a["iou"])) if a["iou"] else 0.0,
            "bind_rate": float(np.mean(a["bind"])) if a["bind"] else 0.0,
            "false_positive_rate": float(np.mean(a["fp"])) if a["fp"] else 0.0,
            "false_positive_rate_hard": float(np.mean(a["hfp"])) if a["hfp"] else None,
            "part_bound_rate": float(np.mean(a["pb"])) if a["pb"] else 0.0,
            "grey_pixel_ratio": a["grey"] / max(1, a["px"]),
            # F: what the colorize overlay hands downstream, per GT part pixel
            "pixel_acc": a["cacc"] / cpx, "pixel_wrong": a["cwrong"] / cpx,
            "pixel_unassigned": 1.0 - (a["cacc"] + a["cwrong"]) / cpx,
            "painted_part_rate": float(np.mean(a["cpart"])) if a["cpart"] else 0.0}


def _new_aacc() -> dict:
    return {"cpx": 0, "cacc": 0, "cwrong": 0, "cpart": [], "fp": [], "hfp": [], "images": 0}


def _asummary(a: dict) -> dict:
    cpx = max(1, a["cpx"])
    return {"images": a["images"], "pixel_acc": a["cacc"] / cpx, "pixel_wrong": a["cwrong"] / cpx,
            "pixel_unassigned": 1.0 - (a["cacc"] + a["cwrong"]) / cpx,
            "painted_part_rate": float(np.mean(a["cpart"])) if a["cpart"] else 0.0,
            "false_positive_rate": float(np.mean(a["fp"])) if a["fp"] else 0.0,
            "false_positive_rate_hard": float(np.mean(a["hfp"])) if a["hfp"] else None}


@torch.no_grad()
def evaluate(processor, model, samples: list[ImageSample], template: str, bank, device: str,
             threshold: float, negatives: dict[str, list[str]], max_images: int | None = None,
             chunk: int = 12, hard_negatives: dict[str, list[str]] | None = None,
             thresholds: list[float] | None = None, slices: dict[str, set[str]] | None = None,
             fired: list | None = None, fired_t: float = 0.5,
             templates: list[str] | None = None,
             assign_taus: list[float] | None = None, assign_temp: float = 1.0,
             gate_rel: float = 0.0, gate_topk: int = 0, min_comp: float = 0.0) -> dict:
    """`negatives` (random names) give the comparable false_positive_rate; `hard_negatives`, when
    given, add false_positive_rate_hard on co-occurring / similar names.

    The main metrics use `threshold`; `thresholds` (extra score cut-offs, same forward pass) are
    reported under "sweep" so banks can be compared at equal false-positive rate: a bank that just
    makes everything more detectable looks good at 0.3 and no better once the threshold is raised.

    F metrics: the positive prompts' hard unions are painted the way sam3_to_2dmap.colorize does
    (smallest first, no overwriting, clipped to the silhouette) and compared per pixel with ids.npy:
    pixel_acc (right name), pixel_wrong (another prompt's name - the "mistaken identity" error),
    pixel_unassigned (grey), painted_part_rate (visible parts whose own name covers >= 50 % after the
    overlay). `slices` (name -> object ids) repeat everything on subsets, e.g. the hard list.

    v5 ("argmax" block, per tau in `assign_taus`): the same forward pass painted with sam3_bank.paint_argmax
    (per-pixel softmax over the positive prompts + background, PLAN_concept_bank_v5.md section 3); the
    false-positive rates there mean "the negative won at least one silhouette pixel" in a softmax that
    also contains the negatives. Independent of the score threshold.

    `gate_rel` / `gate_topk` (sam3_bank.query_gate) drop each prompt's weak queries before the logsumexp and
    `min_comp` (sam3_bank.clean_components) re-labels fragments under that share of the silhouette: the
    deployment painter's de-speckling, applied here so the numbers describe what is actually shipped."""
    ths = [threshold] + [t for t in (thresholds or []) if t != threshold]
    taus = list(assign_taus) if assign_taus else [0.0]
    slices = slices or {}
    tmpls = templates or [template]
    acc = {(sl, t): _new_acc() for sl in ["all", *slices] for t in ths}
    aacc = {(sl, tau): _new_aacc() for sl in ["all", *slices] for tau in taus}
    bg_logit = bank.bg_logit() if bank is not None else 0.0
    use_qv = bank is not None and bank.nn_cos > 0 and bank.tvec is not None
    for k, s in enumerate(samples):
        if max_images and k >= max_images:
            break
        if not s.gts:
            continue
        h, w = s.ids.shape
        vis = sb.encode_image(processor, model, s.image, device)
        names = list(s.gts.keys())
        negs = negatives.get(s.obj, [])
        hards = (hard_negatives or {}).get(s.obj, [])
        allp = names + negs + hards
        votes = {t: None for t in ths}
        maps_sum = None
        for tmpl in tmpls:
            chunks = {t: [] for t in ths}
            mchunks = []
            for k0 in range(0, len(allp), chunk):              # bounded memory: mask logits are [N, Q, h, w]
                part = allp[k0:k0 + chunk]
                prompts = [sb.fill_template(tmpl, n, s.obj_name) for n in part]
                tf, am = sb.text_features_batch(processor, model, prompts, device)
                qv = sb.text_query_vecs(tf, am) if use_qv else None
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    out, bias = sb.bank_forward(model, vis, tf, am, bank, part, qv)
                for t in ths:
                    chunks[t].append(sb.batch_union_masks(out, (h, w), t, bias=bias))
                gate = sb.query_gate(sb.batch_scores(out, bias).float(), gate_rel, gate_topk)
                mchunks.append(sb.name_logit_maps(out, bias, gate))
                del out
            for t in ths:
                u = torch.cat(chunks[t], 0).float()
                votes[t] = u if votes[t] is None else votes[t] + u
            m = torch.cat(mchunks, 0)
            maps_sum = m if maps_sum is None else maps_sum + m
        maps = maps_sum / len(tmpls)                                        # [N, h, w] name-level logits
        gts = {n: torch.from_numpy(s.gts[n]).to(device) for n in names}
        ids_t = torch.from_numpy(s.ids.astype(np.int64)).to(device)
        fg = ids_t >= 0
        lut = torch.full((max(1, len(s.names)),), -1, dtype=torch.long)
        pms = {}
        for p, nm in enumerate(s.names):
            nm = (nm or "").strip()
            if nm in s.gts:
                lut[p] = names.index(nm)
                pm = ids_t == p
                px = int(pm.sum().item())
                if px:
                    pms[p] = (nm, pm, px)
        gt_lab = torch.where(fg, lut.to(device)[ids_t.clamp(min=0, max=len(lut) - 1)], torch.full_like(ids_t, -1))
        valid = gt_lab >= 0
        n_valid = int(valid.sum().item())
        sls = ["all"] + [sl for sl, objs in slices.items() if s.obj in objs]
        for t in ths:
            unions = votes[t] > (0.5 * len(tmpls))
            painted = sb.paint(unions[:len(names)], fg)
            n_right = int((valid & (painted == gt_lab)).sum().item())
            n_wrong = int((valid & (painted >= 0) & (painted != gt_lab)).sum().item())
            rows = {"iou": [], "bind": [], "fp": [], "hfp": [], "pb": [], "cpart": [], "px": 0, "grey": 0}
            for i, n in enumerate(names):
                v = sb.iou(unions[i], gts[n])
                rows["iou"].append(v)
                rows["bind"].append(v >= 0.5)
            for j in range(len(negs)):
                rows["fp"].append(bool(unions[len(names) + j].any()))
            for j in range(len(hards)):
                hit = unions[len(names) + len(negs) + j]
                rows["hfp"].append(bool(hit.any()))
                if fired is not None and t == fired_t and bool(hit.any()):
                    # which GT parts the false hit lands on (by pixel share) - tells label ambiguity from real errors
                    on = torch.bincount(gt_lab[hit & valid].clamp(min=0), minlength=len(names)).tolist() if bool((hit & valid).any()) else []
                    top = sorted(((names[i], c) for i, c in enumerate(on) if c), key=lambda x: -x[1])[:3]
                    fired.append({"obj": s.obj, "az": s.az, "neg": hards[j], "px": int(hit.sum().item()),
                                  "on_parts": top, "own": names})
            for nm, pm, px in pms.values():
                ni = names.index(nm)
                cov = (unions[ni] & pm).sum().item() / px
                bound = cov >= 0.5
                rows["pb"].append(bound)
                rows["px"] += px
                if not bound:
                    rows["grey"] += px
                rows["cpart"].append(((painted == ni) & pm).sum().item() / px >= 0.5)
            for sl in sls:
                a = acc[(sl, t)]
                for key in ("iou", "bind", "fp", "hfp", "pb", "cpart"):
                    a[key].extend(rows[key])
                a["px"] += rows["px"]
                a["grey"] += rows["grey"]
                a["cpx"] += n_valid
                a["cacc"] += n_right
                a["cwrong"] += n_wrong
                a["images"] += 1
        # v5 argmax painting: positives-only softmax for what colorize would hand downstream; the full
        # softmax (with the negatives) for whether an absent name can still win pixels
        for tau in taus:
            pa = sb.paint_argmax(maps[:len(names)], fg, bg_logit, tau, assign_temp)
            if min_comp > 0:
                pa = sb.clean_components(pa, fg, min_comp)
            full = sb.paint_argmax(maps, fg, bg_logit, tau, assign_temp)
            won = torch.bincount(full[fg].clamp(min=0), minlength=len(allp) + 1)
            arow = {"cpx": n_valid,
                    "cacc": int((valid & (pa == gt_lab)).sum().item()),
                    "cwrong": int((valid & (pa >= 0) & (pa != gt_lab)).sum().item()),
                    "cpart": [((pa == names.index(nm)) & pm).sum().item() / px >= 0.5 for nm, pm, px in pms.values()],
                    "fp": [bool(won[len(names) + j] > 0) for j in range(len(negs))],
                    "hfp": [bool(won[len(names) + len(negs) + j] > 0) for j in range(len(hards))]}
            for sl in sls:
                a = aacc[(sl, tau)]
                for key in ("cpx", "cacc", "cwrong"):
                    a[key] += arow[key]
                for key in ("cpart", "fp", "hfp"):
                    a[key].extend(arow[key])
                a["images"] += 1

    res = _summary(acc[("all", threshold)])
    if len(ths) > 1:
        res["sweep"] = {f"{t:g}": _summary(acc[("all", t)]) for t in ths[1:]}
    res["argmax"] = {f"{tau:g}": _asummary(aacc[("all", tau)]) for tau in taus}
    if slices:
        res["slices"] = {sl: {f"{t:g}": _summary(acc[(sl, t)]) for t in ths} for sl in slices}
        for sl in slices:
            res["slices"][sl]["argmax"] = {f"{tau:g}": _asummary(aacc[(sl, tau)]) for tau in taus}
    return res


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="facebook/sam3")
    ap.add_argument("--template", default="name", help="key in sam3_bank.TEMPLATES or literal template")
    ap.add_argument("--templates", default=None,
                    help="G: comma list of TEMPLATES keys. Train samples one per image; eval majority-votes. "
                         "Overrides --template when set.")
    ap.add_argument("--azimuths", default="0,135")
    ap.add_argument("--holdout_file", default=None)
    ap.add_argument("--holdout_frac", type=float, default=0.10)
    ap.add_argument("--hard_file", default=None, help="object ids reported as an extra 'hard' slice of the holdout eval")
    ap.add_argument("--extra_eval", action="append", default=[],
                    help="NAME=ROOT: another dataset in the same layout, fully evaluated after every epoch "
                         "(e.g. an out-of-domain set standing in for external assets)")
    ap.add_argument("--min_count", type=int, default=8, help="names seen >= this often in TRAIN get their own E")
    ap.add_argument("--negatives", type=int, default=3)
    ap.add_argument("--neg_mode", choices=["random", "hard", "mixed"], default="random",
                    help="hard = co-occurring / text-similar names absent from the object (SAM 3 style); "
                         "mixed = --negatives of each kind")
    ap.add_argument("--syn_cos", type=float, default=0.85, help="hard negatives closer than this to a present name are skipped")
    ap.add_argument("--eval_only", action="store_true", help="evaluate --resume bank (or no bank) on the holdout and exit")
    ap.add_argument("--no_e0", action="store_true", help="ignore the shared offset E_0 (per-name offsets only)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr_name", type=float, default=1e-2)
    ap.add_argument("--lr_e0", type=float, default=1e-3)
    ap.add_argument("--neg_weight", type=float, default=None,
                    help="weight of the negatives' BCE(union, 0); default 0.5, or 0 when --assign_ce is on")
    ap.add_argument("--presence_weight", type=float, default=0.5)
    ap.add_argument("--bce_weight", type=float, default=None,
                    help="weight of the per-prompt BCE+Dice (v3 main term); default 1, or 0 when --assign_ce is on")
    ap.add_argument("--l2", type=float, default=1e-3, help="pull E[name] toward 0 (keeps rare names near E_0)")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--eval_thresholds", default="0.4,0.5,0.6",
                    help="extra score thresholds evaluated from the same pass (equal-FP comparison)")
    ap.add_argument("--keep_uncertain", action="store_true", help="also use uncertain parts as positives")
    ap.add_argument("--eval_images", type=int, default=240, help="holdout images used for the per-epoch eval")
    ap.add_argument("--max_train_images", type=int, default=None)
    ap.add_argument("--max_prompts", type=int, default=12,
                    help="max names per step (with --pixel_ce, positives are kept first so the softmax sees every part)")
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None, help="bank.pt to continue from (E_0/E copied; schedule restarts)")
    # ---- v4
    g = ap.add_argument_group("v4 A: cross-name competition")
    g.add_argument("--pixel_ce", type=float, default=0.0, help="weight of the per-pixel softmax CE over names (+bg)")
    g.add_argument("--ce_temp", type=float, default=1.0, help="temperature on logit(union) inside the softmax")
    g.add_argument("--ce_bg", type=float, default=0.0, help="fixed background logit (0 = union 0.5)")
    g.add_argument("--ce_mode", choices=["all", "painted"], default="all",
                   help="painted = CE only where some name would be painted (pure ranking term)")
    g.add_argument("--ce_balance", choices=["inv", "sqrt"], default="inv")
    g.add_argument("--margin", type=float, default=0.0, help="weight of the hinge margin against negatives")
    g.add_argument("--margin_m", type=float, default=1.0)
    g = ap.add_argument_group("v4 B: prompt-side capacity")
    g.add_argument("--ctx", type=int, default=0, help="K learnable context tokens prepended to the text tokens")
    g.add_argument("--lr_ctx", type=float, default=1e-3)
    g.add_argument("--lowrank", type=int, default=0, help="rank r of pooled += pooled @ down @ up (0 = off)")
    g.add_argument("--lr_lowrank", type=float, default=1e-3)
    g = ap.add_argument_group("v4 C: long tail")
    g.add_argument("--word_offsets", action="store_true", help="learn E_word per word; name offset += sum of its words")
    g.add_argument("--min_word_count", type=int, default=8)
    g.add_argument("--lr_word", type=float, default=1e-2)
    g.add_argument("--nn_cos", type=float, default=0.0,
                   help="eval / inference: OOV prompts borrow E from vocabulary names with text cosine > this (0 = off)")
    g.add_argument("--nn_temp", type=float, default=0.05)
    g = ap.add_argument_group("v4 D: online hard negatives")
    g.add_argument("--online_hard", action="store_true", help="after each epoch, replace the hard negatives by the bank's actual false hits")
    g.add_argument("--online_images", type=int, default=600, help="train images scanned per mining pass")
    g.add_argument("--online_cands", type=int, default=8, help="candidate names probed per image")
    g.add_argument("--online_weight", type=float, default=2.0, help="loss weight of mined negatives")
    g.add_argument("--mine_threshold", type=float, default=0.5)
    g = ap.add_argument_group("v4 E: per-name bias")
    g.add_argument("--name_bias", action="store_true", help="learn a per-name bias on the detection logit")
    g.add_argument("--lr_bias", type=float, default=1e-2)
    g = ap.add_argument_group("v4 H: decoder LoRA")
    g.add_argument("--decoder_lora", type=int, default=0,
                   help="LoRA rank on decoder cross-attention (0 = off). Vision encoder stays frozen.")
    g.add_argument("--lora_alpha", type=float, default=None, help="LoRA alpha (default 2 * rank)")
    g.add_argument("--lora_scope", default="mask,text",
                   help="comma list: mask = mask_decoder.prompt_cross_attn; text = detr_decoder text_cross_attn; "
                        "embed = mask_embedder MLP; proj = instance_projection (v5: both ends of the mask dot product)")
    g.add_argument("--lr_lora", type=float, default=1e-4)
    g.add_argument("--lora_file", default=None, help="decoder_lora.pt to load after inject (eval_only / resume)")
    g = ap.add_argument_group("v5: per-pixel assignment (PLAN_concept_bank_v5.md)")
    g.add_argument("--assign_ce", type=float, default=0.0,
                   help="weight of the per-pixel assignment CE over name-level mask logits + learnable bg (0 = off). "
                        "Turns --bce_weight / --neg_weight default to 0; presence BCE stays.")
    g.add_argument("--assign_temp", type=float, default=1.0, help="softmax temperature (train and argmax eval)")
    g.add_argument("--assign_balance", choices=["sqrt", "inv"], default="sqrt")
    g.add_argument("--assign_syn", type=float, default=None,
                   help="positives closer than this (text cosine) share credit in the CE; default = --syn_cos, 0 = off")
    g.add_argument("--lr_bg", type=float, default=1e-2)
    g.add_argument("--assign_taus", default="0,0.3,0.5",
                   help="eval: min winner probability for a pixel to be painted under the argmax painter (always reported)")
    g.add_argument("--gate_rel", type=float, default=0.0,
                   help="eval: drop a prompt's queries scoring under this x its best before the pixel competition (0 = off)")
    g.add_argument("--gate_topk", type=int, default=0, help="eval: keep only each prompt's k best queries (0 = off)")
    g.add_argument("--min_comp", type=float, default=0.0,
                   help="eval: re-label painted fragments smaller than this share of the silhouette (0 = off)")
    ap.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    ap.add_argument("--wandb_project", default="segvigen-sam3")
    ap.add_argument("--wandb_entity", default=None)
    ap.add_argument("--wandb_name", default=None, help="Display name (default: out dir basename)")
    ap.add_argument("--wandb_id", default=None, help="Run id to resume (default: out dir basename)")
    ap.add_argument("--wandb_mode", default="online", choices=["online", "offline"])
    args = ap.parse_args()
    if args.bce_weight is None:
        args.bce_weight = 0.0 if args.assign_ce > 0 else 1.0
    if args.neg_weight is None:
        args.neg_weight = 0.0 if args.assign_ce > 0 else 0.5
    if args.assign_syn is None:
        args.assign_syn = args.syn_cos

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    templates = sb.resolve_templates(args.templates) if args.templates else [sb.TEMPLATES.get(args.template, args.template)]
    template = templates[0]
    azimuths = [f"az{float(a):g}" for a in args.azimuths.split(",") if a.strip()]
    if len(templates) > 1:
        print(f"G templates ({len(templates)}): {templates}  (train: sample one; eval: majority vote)", flush=True)

    processor, model = load_sam3(args.model, device)
    for p in model.parameters():
        p.requires_grad_(False)
    lora_params: list = []
    if args.decoder_lora > 0:
        from lora import lora_state_dict, load_lora_state_dict
        lora_params, n_wrap = sb.inject_decoder_lora(
            model, r=args.decoder_lora, alpha=args.lora_alpha, scope=args.lora_scope)
        n_el = sum(p.numel() for p in lora_params)
        print(f"decoder LoRA: r={args.decoder_lora} scope={args.lora_scope} "
              f"{n_wrap} linears / {n_el} params", flush=True)
        if args.lora_file:
            load_lora_state_dict(model, torch.load(args.lora_file, map_location=device, weights_only=False))
            print(f"loaded LoRA from {args.lora_file}", flush=True)

    train_objs, hold_objs = make_split(args.dataset_root, args.holdout_file, args.holdout_frac, args.seed)
    with open(os.path.join(args.out, "split.json"), "w", encoding="utf-8") as f:
        json.dump({"train": train_objs, "holdout": hold_objs}, f)
    print(f"split: {len(train_objs)} train / {len(hold_objs)} holdout objects", flush=True)

    # vocabulary from TRAIN objects only decides which names get their own vector
    counts: Counter = Counter()
    for o in train_objs:
        with open(os.path.join(args.dataset_root, o, "names.json"), encoding="utf-8") as f:
            counts.update(n.strip() for n in json.load(f) if n and n.strip())
    bank_names = sorted(n for n, c in counts.items() if c >= args.min_count)
    neg_pool = list(bank_names)
    D = model.config.detr_encoder_config.hidden_size
    print(f"bank: {len(bank_names)} names with count >= {args.min_count} (D={D}); "
          f"they cover {sum(counts[n] for n in bank_names) / max(1, sum(counts.values())):.1%} of train part instances", flush=True)
    words: list[str] = []
    if args.word_offsets:
        wc: Counter = Counter()
        for n, c in counts.items():
            for w in set(sb.name_words(n)):
                wc[w] += c
        words = sorted(w for w, c in wc.items() if c >= args.min_word_count)
        covered = sum(c for n, c in counts.items() if any(w in set(words) for w in sb.name_words(n)))
        print(f"words: {len(words)} with count >= {args.min_word_count}; names with >= 1 known word cover "
              f"{covered / max(1, sum(counts.values())):.1%} of train part instances", flush=True)

    # unit text vectors of the vocabulary: hard-negative mining below, and the OOV fallback in the bank
    with torch.no_grad():
        vecs = []
        for k in range(0, len(bank_names), 64):
            tf, am = sb.text_features_batch(processor, model, bank_names[k:k + 64], device)
            vecs.append(sb.text_query_vecs(tf, am).cpu())
        tvec = F.normalize(torch.cat(vecs), dim=1) if bank_names else torch.zeros(0, D)
        cos = (tvec @ tvec.T).numpy()

    # ---- the learnable bank (leaf tensors; SAM3 stays frozen)
    def leaf(t: torch.Tensor) -> torch.Tensor:
        return t.to(device).float().detach().requires_grad_(True)

    prev = sb.ConceptBank.load(args.resume, device) if args.resume else None
    if not args.templates and prev is not None and isinstance(prev.meta, dict) and prev.meta.get("templates"):
        templates = list(prev.meta["templates"])
        template = templates[0]
        if len(templates) > 1:
            print(f"G templates from bank.meta ({len(templates)}): {templates}", flush=True)
    E_0 = leaf(prev.E_0 if prev is not None else torch.zeros(D))
    E = torch.zeros(len(bank_names), D)
    b = torch.zeros(len(bank_names)) if args.name_bias else None
    if prev is not None:
        pi = prev.index()
        for i, n in enumerate(bank_names):
            if n in pi:
                E[i] = prev.E[pi[n]].float().cpu()
                if b is not None and prev.b is not None:
                    b[i] = prev.b[pi[n]].float().cpu()
    E_word = None
    if words:
        E_word = torch.zeros(len(words), D)
        if prev is not None and prev.E_word is not None:
            wi = prev.word_index()
            for i, w in enumerate(words):
                if w in wi:
                    E_word[i] = prev.E_word[wi[w]].float().cpu()
    ctx = None
    if args.ctx > 0:
        ctx = torch.randn(args.ctx, D) * 0.02
        if prev is not None and prev.ctx is not None and prev.ctx.shape[0] == args.ctx:
            ctx = prev.ctx.float().cpu()
    lr_down = lr_up = None
    if args.lowrank > 0:
        lr_down, lr_up = torch.randn(D, args.lowrank) * 0.02, torch.zeros(args.lowrank, D)
        if prev is not None and prev.lr_down is not None and prev.lr_down.shape[1] == args.lowrank:
            lr_down, lr_up = prev.lr_down.float().cpu(), prev.lr_up.float().cpu()
    bg = None
    if args.assign_ce > 0:
        bg = prev.bg.float().cpu() if prev is not None and prev.bg is not None else torch.zeros(())
    bank = sb.ConceptBank(E_0=E_0, names=bank_names, E=leaf(E), template=template, base=args.model,
                          meta={"templates": templates, "decoder_lora": args.decoder_lora,
                                "lora_scope": args.lora_scope if args.decoder_lora else "",
                                "assign": args.assign_ce > 0, "assign_temp": args.assign_temp},
                          bg=None if bg is None else leaf(bg),
                          b=None if b is None else leaf(b), words=words,
                          E_word=None if E_word is None else leaf(E_word), tvec=tvec.to(device),
                          nn_cos=args.nn_cos, nn_temp=args.nn_temp,
                          ctx=None if ctx is None else leaf(ctx),
                          lr_down=None if lr_down is None else leaf(lr_down),
                          lr_up=None if lr_up is None else leaf(lr_up))
    name_idx = bank.index()
    word_idx = bank.word_index()
    print("bank params:", bank.describe(), flush=True)

    groups = [{"params": [bank.E], "lr": args.lr_name}, {"params": [bank.E_0], "lr": args.lr_e0}]
    if bank.b is not None:
        groups.append({"params": [bank.b], "lr": args.lr_bias})
    if bank.E_word is not None:
        groups.append({"params": [bank.E_word], "lr": args.lr_word})
    if bank.ctx is not None:
        groups.append({"params": [bank.ctx], "lr": args.lr_ctx})
    if bank.lr_down is not None:
        groups.append({"params": [bank.lr_down, bank.lr_up], "lr": args.lr_lowrank})
    if bank.bg is not None:
        groups.append({"params": [bank.bg], "lr": args.lr_bg})
    if lora_params:
        groups.append({"params": lora_params, "lr": args.lr_lora})
    for g_ in groups:
        g_["base_lr"] = g_["lr"]
    opt = torch.optim.Adam(groups)

    def snapshot(meta: dict) -> sb.ConceptBank:
        s = bank.detached()
        s.meta = {**bank.meta, **meta}
        return s

    def save_lora(tag: str) -> None:
        if not lora_params:
            return
        from lora import lora_state_dict
        torch.save(lora_state_dict(model), os.path.join(args.out, "decoder_lora.pt"))
        if tag:
            torch.save(lora_state_dict(model), os.path.join(args.out, f"decoder_lora_{tag}.pt"))

    print("loading samples ...", flush=True)
    train_imgs = list_images(args.dataset_root, train_objs, azimuths)
    hold_imgs = list_images(args.dataset_root, hold_objs, azimuths)
    if args.max_train_images:
        rng.shuffle(train_imgs)
        train_imgs = train_imgs[:args.max_train_images]
    hold_samples = [ImageSample(args.dataset_root, o, az, not args.keep_uncertain) for o, az in hold_imgs[:args.eval_images]]
    hard_ids = read_id_file(args.hard_file) & set(hold_objs)
    slices = {"hard": hard_ids} if hard_ids else {}
    if hard_ids:
        print(f"hard slice: {len(hard_ids)} holdout objects", flush=True)
    extra_sets: dict[str, tuple[str, list[str], list[ImageSample]]] = {}
    for spec in args.extra_eval:
        nm, root = spec.split("=", 1)
        objs = list_objects(root)
        imgs = list_images(root, objs, azimuths)
        extra_sets[nm] = (root, objs, [ImageSample(root, o, az, not args.keep_uncertain) for o, az in imgs])
        ec: Counter = Counter()
        for o in objs:
            with open(os.path.join(root, o, "names.json"), encoding="utf-8") as f:
                ec.update(n.strip() for n in json.load(f) if n and n.strip())
        in_vocab = sum(c for n, c in ec.items() if n in name_idx)
        print(f"extra eval '{nm}': {len(objs)} objects / {len(imgs)} images, {len(ec)} names, "
              f"vocabulary covers {in_vocab / max(1, sum(ec.values())):.1%} of part instances", flush=True)

    # fixed negatives per object (same across epochs / eval for comparability)
    def own_names(obj: str, root: str = args.dataset_root) -> set[str]:
        with open(os.path.join(root, obj, "names.json"), encoding="utf-8") as f:
            return {n.strip() for n in json.load(f) if n and n.strip()}

    def draw_random(obj: str, k: int, root: str = args.dataset_root) -> list[str]:
        own = own_names(obj, root)
        pool = [n for n in neg_pool if n not in own]
        r = random.Random(zlib.crc32(f"{args.seed}:{obj}".encode()))
        return r.sample(pool, min(k, len(pool)))

    # hard negatives: names that usually appear next to this object's names (co-occurrence over the
    # TRAIN objects) or read like them (cosine of the frozen SAM3 text vectors), but are absent here.
    # Near-synonyms of a present name (cos > --syn_cos) are skipped: they are probably the same part
    # under another label, not a true negative.
    cooc = np.zeros((len(bank_names), len(bank_names)), dtype=np.float32)
    for o in train_objs:
        idx = [name_idx[n] for n in own_names(o) if n in name_idx]
        for i in idx:
            cooc[i, idx] += 1
    np.fill_diagonal(cooc, 0)

    def hard_candidates(obj: str, root: str = args.dataset_root):
        own = own_names(obj, root)
        idx = [name_idx[n] for n in own if n in name_idx]
        if not idx:
            return None, None
        cand = [i for i, n in enumerate(bank_names) if n not in own and cos[i, idx].max() <= args.syn_cos]
        return idx, cand

    def draw_hard(obj: str, k: int, root: str = args.dataset_root) -> list[str]:
        idx, cand = hard_candidates(obj, root)
        if not cand:
            return draw_random(obj, k, root)
        r = random.Random(zlib.crc32(f"{args.seed}:hard:{obj}".encode()))
        by_cooc = sorted(cand, key=lambda i: -cooc[i, idx].sum())
        by_cos = sorted(cand, key=lambda i: -cos[i, idx].max())
        n_rand = 1 if k >= 3 else 0
        n_cooc = (k - n_rand + 1) // 2
        n_cos = k - n_rand - n_cooc
        pick: list[int] = []
        for i in by_cooc[:n_cooc * 2]:            # top-2x pools, sampled, so the negatives vary across objects
            if len(pick) < n_cooc and i not in pick:
                pick.append(i)
        cos_pool = [i for i in by_cos[:n_cos * 3] if i not in pick]
        pick += r.sample(cos_pool, min(n_cos, len(cos_pool)))
        rest = [i for i in cand if i not in pick]
        pick += r.sample(rest, min(n_rand, len(rest)))
        return [bank_names[i] for i in pick]

    def hard_pool(obj: str, k: int) -> list[str]:
        """Deterministic top-k probe list for online mining: half by co-occurrence, half by text cosine."""
        idx, cand = hard_candidates(obj)
        if not cand:
            return []
        by_cooc = sorted(cand, key=lambda i: -cooc[i, idx].sum())
        by_cos = sorted(cand, key=lambda i: -cos[i, idx].max())
        pick: list[int] = []
        for a_, b_ in zip(by_cooc, by_cos):
            for i in (a_, b_):
                if i not in pick and len(pick) < k:
                    pick.append(i)
            if len(pick) >= k:
                break
        return [bank_names[i] for i in pick]

    all_objs = set(train_objs) | set(hold_objs)
    neg_random = {o: draw_random(o, args.negatives) for o in all_objs}
    neg_hard = {o: draw_hard(o, args.negatives) for o in all_objs}
    if args.neg_mode == "mixed":
        negatives = {o: list(dict.fromkeys(neg_hard[o] + neg_random[o])) for o in all_objs}
    else:
        negatives = dict(neg_hard if args.neg_mode == "hard" else neg_random)
    online_negs: dict[str, list[str]] = {}
    ex = hold_objs[0]
    print(f"negatives ({args.neg_mode}); e.g. {ex[:8]} own={sorted(own_names(ex))[:6]} "
          f"random={neg_random[ex]} hard={neg_hard[ex]}", flush=True)
    extra_negs = {}
    for nm, (root, objs, _) in extra_sets.items():
        extra_negs[nm] = ({o: draw_random(o, args.negatives, root) for o in objs},
                          {o: draw_hard(o, args.negatives, root) for o in objs})

    log = open(os.path.join(args.out, "log.jsonl"), "a", encoding="utf-8")
    run = None
    if args.wandb:
        import wandb
        run_name = args.wandb_name or os.path.basename(os.path.normpath(args.out))
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, mode=args.wandb_mode,
                         id=args.wandb_id or run_name, name=run_name, resume="allow",
                         config={**vars(args), "n_train_images": len(train_imgs), "n_holdout_images": len(hold_samples),
                                 "bank_names": len(bank_names), "text_dim": D})
        wandb.define_metric("train/loss", summary="min")
        wandb.define_metric("holdout/*", summary="max")
        print(f"wandb: {run.url or args.wandb_mode}", flush=True)

    METRIC_KEYS = ("miou", "bind_rate", "false_positive_rate", "false_positive_rate_hard", "part_bound_rate",
                   "grey_pixel_ratio", "pixel_acc", "pixel_wrong", "pixel_unassigned", "painted_part_rate")

    def log_row(**kw):
        kw["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        log.write(json.dumps(kw, ensure_ascii=False) + "\n")
        log.flush()
        if run is not None:
            kind = kw.get("kind")
            step = kw.get("step", 0)
            if kind == "train":
                run.log({"train/loss": kw["loss"], "train/e0_norm": kw["e0_norm"], "train/e_norm": kw["e_norm"],
                         "train/vram_gib": kw.get("vram_gib", 0), "train/epoch": kw["epoch"]}, step=step)
            elif kind == "eval":
                pre = kw.get("set", "holdout")
                row = {f"{pre}/{k}": v for k, v in kw.items() if k in METRIC_KEYS and v is not None}
                for t, sub in (kw.get("sweep") or {}).items():
                    row.update({f"{pre}_t{t}/{k}": v for k, v in sub.items() if k in METRIC_KEYS and v is not None})
                for tau, sub in (kw.get("argmax") or {}).items():
                    row.update({f"{pre}_argmax{tau}/{k}": v for k, v in sub.items() if k in METRIC_KEYS and v is not None})
                run.log(row, step=step)

    def fmt(d: dict) -> str:
        def r(v):
            return round(v, 4) if isinstance(v, float) else ({k: r(x) for k, x in v.items()} if isinstance(v, dict) else v)
        return json.dumps({k: r(v) for k, v in d.items()})

    def brief(ev: dict, t: str = "0.5") -> str:
        d = ev["sweep"].get(t, ev) if t != f"{args.threshold:g}" and "sweep" in ev else ev
        return (f"t{t}: miou {d['miou']:.3f} grey {d['grey_pixel_ratio']:.3f} hardFP {d['false_positive_rate_hard']:.3f} "
                f"| paint acc {d['pixel_acc']:.3f} wrong {d['pixel_wrong']:.3f} unassigned {d['pixel_unassigned']:.3f} "
                f"parts {d['painted_part_rate']:.3f}")

    def brief_argmax(ev: dict) -> str:
        return " | ".join(
            f"tau{tau}: acc {d['pixel_acc']:.3f} wrong {d['pixel_wrong']:.3f} unassigned {d['pixel_unassigned']:.3f} "
            f"parts {d['painted_part_rate']:.3f} hardFP {d['false_positive_rate_hard'] if d['false_positive_rate_hard'] is None else round(d['false_positive_rate_hard'], 3)}"
            for tau, d in ev.get("argmax", {}).items())

    sweep = [float(t) for t in args.eval_thresholds.split(",") if t.strip()]
    taus = [float(t) for t in args.assign_taus.split(",") if t.strip()]
    eval_kw = dict(chunk=args.max_prompts, thresholds=sweep, assign_taus=taus, assign_temp=args.assign_temp,
                   gate_rel=args.gate_rel, gate_topk=args.gate_topk, min_comp=args.min_comp)

    def run_evals(bk, epoch: int, step: int, tag) -> dict:
        fired: list = []
        ev = evaluate(processor, model, hold_samples, template, bk, device, args.threshold, neg_random,
                      hard_negatives=neg_hard, slices=slices, fired=fired, templates=templates, **eval_kw)
        with open(os.path.join(args.out, f"hard_fp_{os.path.basename(str(tag)).replace('.pt', '')}.json"), "w", encoding="utf-8") as f:
            json.dump(fired, f, ensure_ascii=False, indent=0)
        print(f"holdout [{tag}]:", fmt(ev), flush=True)
        print("   ", brief(ev), flush=True)
        print("    argmax", brief_argmax(ev), flush=True)
        if "slices" in ev and "hard" in ev["slices"]:
            hd = ev["slices"]["hard"].get("0.5") or next(iter(ev["slices"]["hard"].values()))
            print(f"    hard slice t0.5: miou {hd['miou']:.3f} grey {hd['grey_pixel_ratio']:.3f} "
                  f"paint acc {hd['pixel_acc']:.3f} wrong {hd['pixel_wrong']:.3f}", flush=True)
            ha = ev["slices"]["hard"].get("argmax", {})
            if ha:
                tau0, hd0 = next(iter(ha.items()))
                print(f"    hard slice argmax tau{tau0}: acc {hd0['pixel_acc']:.3f} wrong {hd0['pixel_wrong']:.3f} "
                      f"unassigned {hd0['pixel_unassigned']:.3f}", flush=True)
        log_row(kind="eval", set="holdout", epoch=epoch, step=step, bank=tag, **ev)
        for nm, (root, objs, samples) in extra_sets.items():
            rnd, hrd = extra_negs[nm]
            ex_ev = evaluate(processor, model, samples, template, bk, device, args.threshold, rnd,
                             hard_negatives=hrd, templates=templates, **eval_kw)
            print(f"{nm} [{tag}]:", fmt(ex_ev), flush=True)
            print("   ", brief(ex_ev), flush=True)
            print("    argmax", brief_argmax(ex_ev), flush=True)
            log_row(kind="eval", set=nm, epoch=epoch, step=step, bank=tag, **ex_ev)
            ev[f"extra:{nm}"] = ex_ev
        return ev

    if args.eval_only:
        if args.resume:
            bk = sb.ConceptBank.load(args.resume, device)
            if not args.templates and isinstance(bk.meta, dict) and bk.meta.get("templates"):
                templates[:] = list(bk.meta["templates"])
                template = templates[0]
                if len(templates) > 1:
                    print(f"G templates from bank.meta ({len(templates)}): {templates}", flush=True)
            if args.nn_cos > 0:
                bk.nn_cos, bk.nn_temp = args.nn_cos, args.nn_temp
                if bk.tvec is None:          # v3 banks carry no text vectors; compute them for the fallback
                    with torch.no_grad():
                        vs = []
                        for k in range(0, len(bk.names), 64):
                            tf, am = sb.text_features_batch(processor, model, bk.names[k:k + 64], device)
                            vs.append(sb.text_query_vecs(tf, am))
                        bk.tvec = F.normalize(torch.cat(vs), dim=1)
            print("eval bank:", bk.describe(), flush=True)
            run_evals(bk, -1, 0, args.resume)
        else:
            run_evals(None, 0, 0, "none")
        if run is not None:
            run.finish()
        return
    run_evals(None, 0, 0, "none")

    @torch.no_grad()
    def mine_hard(epoch: int) -> None:
        """D: probe the current bank with each object's most confusable absent names; the ones that
        actually fire become that object's negatives for the next epoch (weighted --online_weight)."""
        bk = bank.detached()
        imgs = rng.sample(train_imgs, min(args.online_images, len(train_imgs)))
        hits: dict[str, Counter] = {}
        probed = fired = 0
        for o, az in imgs:
            cands = hard_pool(o, args.online_cands)
            if not cands:
                continue
            s = ImageSample(args.dataset_root, o, az, not args.keep_uncertain)
            vis = sb.encode_image(processor, model, s.image, device)
            for k0 in range(0, len(cands), args.max_prompts):
                part = cands[k0:k0 + args.max_prompts]
                tf, am = sb.text_features_batch(processor, model, [sb.fill_template(rng.choice(templates), n, s.obj_name) for n in part], device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    out, bias = sb.bank_forward(model, vis, tf, am, bk, part)
                px = sb.batch_union_masks(out, s.ids.shape, args.mine_threshold, bias=bias).flatten(1).sum(1)
                for j, n in enumerate(part):
                    probed += 1
                    if px[j] > 0:
                        fired += 1
                        hits.setdefault(o, Counter())[n] += int(px[j])
        for o, c in hits.items():
            mined = [n for n, _ in c.most_common(args.negatives)]
            online_negs[o] = mined
            fill = neg_random[o] if args.neg_mode != "hard" else neg_hard[o]
            negatives[o] = list(dict.fromkeys(mined + fill))
        print(f"[mine epoch {epoch}] {len(imgs)} images, {fired}/{probed} probes fired at t{args.mine_threshold:g}; "
              f"{len(hits)} objects got new negatives", flush=True)
        log_row(kind="mine", epoch=epoch, images=len(imgs), probed=probed, fired=fired, objects=len(hits))

    total_steps = args.epochs * len(train_imgs)
    step = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        order = list(train_imgs)
        rng.shuffle(order)
        run_loss, run_n = 0.0, 0
        parts_sum: Counter = Counter()
        for o, az in order:
            step += 1
            # cosine decay to 0.1x over the whole run
            frac = step / max(1, total_steps)
            scale = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac))
            for g_ in opt.param_groups:
                g_["lr"] = g_["base_lr"] * scale

            s = ImageSample(args.dataset_root, o, az, not args.keep_uncertain)
            if not s.pos_names:
                continue
            negs = negatives.get(o, [])
            pos = list(s.pos_names)
            # A (pixel CE / margin) ranks names against each other: every visible name must stay in
            # the same softmax. Drop negatives first; only then subsample positives.
            if args.pixel_ce > 0 or args.margin > 0 or args.assign_ce > 0:
                room = max(0, args.max_prompts - len(pos))
                if len(negs) > room:
                    negs = list(negs[:room])
                if len(pos) > args.max_prompts:
                    pos = rng.sample(pos, args.max_prompts)
            else:
                cap = max(1, args.max_prompts - len(negs))
                if len(pos) > cap:
                    pos = rng.sample(pos, cap)
            s.pos_names = pos
            names = s.pos_names + negs
            tmpl = rng.choice(templates)
            prompts = [sb.fill_template(tmpl, n, s.obj_name) for n in names]
            vis = sb.encode_image(processor, model, s.image, device)
            tf, am = sb.text_features_batch(processor, model, prompts, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                out, bias = sb.bank_forward(model, vis, tf, am, bank, names, None, not args.no_e0)
            union = sb.batch_soft_union(out, bias).float()                # [N, h, w]
            hw = union.shape[-2:]
            n_pos = len(s.pos_names)
            terms = {}
            if args.bce_weight > 0:
                terms["bce"] = args.bce_weight * torch.stack(
                    [bce_dice(union[i], soft_gt(torch.from_numpy(s.gts[n]).to(device), hw)) for i, n in enumerate(s.pos_names)]).mean()
            if negs and args.neg_weight > 0:
                per_neg = F.binary_cross_entropy(union[n_pos:].clamp(1e-6, 1 - 1e-6), torch.zeros_like(union[n_pos:]),
                                                 reduction="none").flatten(1).mean(1)
                wneg = torch.tensor([args.online_weight if n in online_negs.get(o, ()) else 1.0 for n in negs], device=device)
                terms["neg"] = args.neg_weight * (per_neg * wneg).sum() / wneg.sum()
            if args.assign_ce > 0:
                maps = sb.name_logit_maps(out, bias)                      # [N, h, w]
                label = pixel_labels(s, s.pos_names, hw, device)
                syn = None
                if args.assign_syn > 0:
                    qv = F.normalize(sb.text_query_vecs(tf, am), dim=1)
                    syn = (qv @ qv.T) > args.assign_syn
                    syn.fill_diagonal_(False)
                terms["assign"] = args.assign_ce * assign_loss(maps, bank.bg, label, n_pos, args.assign_temp,
                                                               syn, args.assign_balance)
            if out.presence_logits is not None:
                tgt = torch.cat([torch.ones(n_pos), torch.zeros(len(negs))]).to(device)
                terms["pres"] = args.presence_weight * F.binary_cross_entropy_with_logits(out.presence_logits.float().view(-1), tgt)
            if args.pixel_ce > 0 or args.margin > 0:
                label = pixel_labels(s, s.pos_names, hw, device)
                ce, mg = competition_losses(union, label, n_pos, args.ce_temp, args.ce_bg,
                                            args.margin_m if args.margin > 0 else 0.0, args.ce_mode, args.ce_balance)
                if args.pixel_ce > 0:
                    terms["ce"] = args.pixel_ce * ce
                if args.margin > 0:
                    terms["margin"] = args.margin * mg
            if args.l2 > 0:
                used = [name_idx[n] for n in s.pos_names if n in name_idx]
                reg = bank.E[used].pow(2).sum(1).mean() if used else union.new_zeros(())
                if bank.E_word is not None:
                    wu = sorted({word_idx[w] for n in s.pos_names for w in sb.name_words(n) if w in word_idx})
                    if wu:
                        reg = reg + bank.E_word[wu].pow(2).sum(1).mean()
                terms["l2"] = args.l2 * reg
            loss = sum(terms.values())

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_loss += float(loss.item()); run_n += 1
            for k_, v_ in terms.items():
                parts_sum[k_] += float(v_.item())
            if step % 50 == 0:
                el = time.time() - t0
                pstr = " ".join(f"{k_}={v_ / run_n:.3f}" for k_, v_ in sorted(parts_sum.items()))
                print(f"[{el / 60:6.1f} min] epoch {epoch} step {step}/{total_steps} loss {run_loss / run_n:.4f} ({pstr}) "
                      f"|E_0| {bank.E_0.norm().item():.3f} |E| {bank.E.norm(dim=1).mean().item():.3f} "
                      f"({el / step:.2f} s/img, eta {(total_steps - step) * el / step / 60:.0f} min)", flush=True)
                log_row(kind="train", epoch=epoch, step=step, loss=run_loss / run_n,
                        terms={k_: v_ / run_n for k_, v_ in parts_sum.items()},
                        e0_norm=bank.E_0.norm().item(), e_norm=bank.E.norm(dim=1).mean().item(),
                        vram_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2) if device == "cuda" else 0)
                run_loss, run_n = 0.0, 0
                parts_sum = Counter()
            if step % args.save_every == 0:
                snapshot({"epoch": epoch, "step": step, "min_count": args.min_count, "partial": True}).save(
                    os.path.join(args.out, "bank.pt"))
                save_lora("partial")

        snap = snapshot({"epoch": epoch, "step": step, "min_count": args.min_count, "args": vars(args)})
        snap.save(os.path.join(args.out, f"bank_epoch{epoch}.pt"))
        snap.save(os.path.join(args.out, "bank.pt"))
        save_lora(f"epoch{epoch}")
        run_evals(snap, epoch, step, f"epoch{epoch}")
        if args.online_hard and epoch < args.epochs:
            mine_hard(epoch)

    # text cache for Stage D: mean-pooled projected features (+ offsets) per name and object name
    print("writing text cache ...", flush=True)
    all_names: Counter = Counter()
    obj_names: set[str] = set()
    for o in train_objs + hold_objs:
        names, obj_name, _ = sb.load_object_labels(os.path.join(args.dataset_root, o))
        all_names.update(n.strip() for n in names if n and n.strip())
        if obj_name:
            obj_names.add(obj_name)
    cache = {}
    final = bank.detached()
    with torch.no_grad():
        todo = sorted(all_names) + sorted(obj_names)
        for k in range(0, len(todo), 64):
            chunk = todo[k:k + 64]
            tf, am = sb.text_features_batch(processor, model, chunk, device)
            mean = sb.text_query_vecs(tf, am)                                   # [n, D]
            off = final.offsets(chunk, mean if final.nn_cos > 0 else None)
            for i, n in enumerate(chunk):
                cache[n] = (mean[i] + off[i]).cpu()
    torch.save({"dim": D, "names": sorted(all_names), "objects": sorted(obj_names), "vectors": cache,
                "template": template, "note": "mean over text tokens of SAM3 projected text features + bank offset"},
               os.path.join(args.out, "text_cache.pt"))
    if run is not None:
        run.finish()
    print("done", flush=True)


if __name__ == "__main__":
    main()
