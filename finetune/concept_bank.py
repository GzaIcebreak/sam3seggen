"""Stage B: learn a SAM3 concept bank (shared offset E_0 + per-name offsets E) from GT part masks.

    <.venv_holo python> finetune/concept_bank.py --dataset_root E:\data\pv --out E:\data\concept_bank \
        --template name --epochs 3 --holdout_file E:\data\pv_holdout_mix.txt

M2C (arXiv 2606.26711) with a two-level bank: every prompt's projected text features get
E_0 + E[name] added; SAM3 itself is frozen (no gradients into any weight). Positives are the
visible, non-uncertain part names of an image (target = union of the parts with that name);
negatives are frequent names from other objects (target = empty, presence = 0).

Loss per image = mean over positives of [BCE + Dice on the soft union]
               + 0.5 * mean over negatives of BCE(soft union, 0)
               + 0.5 * presence BCE over all prompts.

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


def make_split(dataset_root: str, holdout_file: str | None, holdout_frac: float, seed: int):
    objs = sorted(d for d in os.listdir(dataset_root)
                  if os.path.exists(os.path.join(dataset_root, d, "views", "az0", "ids.npy")))
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


class ImageSample:
    __slots__ = ("obj", "az", "image", "ids", "names", "obj_name", "uncertain", "gts", "pos_names")

    def __init__(self, dataset_root: str, obj: str, az: str, skip_uncertain: bool):
        v = os.path.join(dataset_root, obj, "views", az)
        self.obj, self.az = obj, az
        self.image = Image.open(os.path.join(v, "render.png"))
        self.ids = np.load(os.path.join(v, "ids.npy"))
        self.names, self.obj_name, self.uncertain = sb.load_object_labels(os.path.join(dataset_root, obj))
        self.gts = sb.gt_unions(self.ids, self.names)
        bad = set()
        if skip_uncertain:
            for p, n in enumerate(self.names):
                if p < len(self.uncertain) and self.uncertain[p] and n.strip():
                    bad.add(n.strip())
        self.pos_names = [n for n in self.gts if n not in bad]


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


# --------------------------------------------------------------------------- eval

@torch.no_grad()
def evaluate(processor, model, samples: list[ImageSample], template: str, bank_fn, device: str,
             threshold: float, negatives: dict[str, list[str]], max_images: int | None = None,
             chunk: int = 12, hard_negatives: dict[str, list[str]] | None = None,
             thresholds: list[float] | None = None) -> dict:
    """`negatives` (random names) give the comparable false_positive_rate; `hard_negatives`, when
    given, add false_positive_rate_hard on co-occurring / similar names.

    The main metrics use `threshold`; `thresholds` (extra score cut-offs, same forward pass) are
    reported under "sweep" so banks can be compared at equal false-positive rate: a bank that just
    makes everything more detectable looks good at 0.3 and no better once the threshold is raised."""
    ths = [threshold] + [t for t in (thresholds or []) if t != threshold]
    acc = {t: {"iou": [], "bind": [], "fp": [], "hfp": [], "pb": [], "px": 0, "grey": 0} for t in ths}
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
        chunks = {t: [] for t in ths}
        for k in range(0, len(allp), chunk):              # bounded memory: mask logits are [N, Q, h, w]
            part = allp[k:k + chunk]
            prompts = [sb.fill_template(template, n, s.obj_name) for n in part]
            tf, am = sb.text_features_batch(processor, model, prompts, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                out = sb.run_prompts(model, vis, tf, am, bank_fn(part))
            for t in ths:
                chunks[t].append(sb.batch_union_masks(out, (h, w), t))
            del out
        gts = {n: torch.from_numpy(s.gts[n]).to(device) for n in names}
        pms = {}
        for p, nm in enumerate(s.names):
            nm = (nm or "").strip()
            if nm in s.gts:
                pm = torch.from_numpy(s.ids == p).to(device)
                px = int(pm.sum().item())
                if px:
                    pms[p] = (nm, pm, px)
        for t in ths:
            unions = torch.cat(chunks[t], 0)
            a = acc[t]
            for i, n in enumerate(names):
                v = sb.iou(unions[i], gts[n])
                a["iou"].append(v)
                a["bind"].append(v >= 0.5)
            for j in range(len(negs)):
                a["fp"].append(bool(unions[len(names) + j].any()))
            for j in range(len(hards)):
                a["hfp"].append(bool(unions[len(names) + len(negs) + j].any()))
            for nm, pm, px in pms.values():
                cov = (unions[names.index(nm)] & pm).sum().item() / px
                bound = cov >= 0.5
                a["pb"].append(bound)
                a["px"] += px
                if not bound:
                    a["grey"] += px

    def summary(a: dict) -> dict:
        return {"prompts": len(a["iou"]), "miou": float(np.mean(a["iou"])) if a["iou"] else 0.0,
                "bind_rate": float(np.mean(a["bind"])) if a["bind"] else 0.0,
                "false_positive_rate": float(np.mean(a["fp"])) if a["fp"] else 0.0,
                "false_positive_rate_hard": float(np.mean(a["hfp"])) if a["hfp"] else None,
                "part_bound_rate": float(np.mean(a["pb"])) if a["pb"] else 0.0,
                "grey_pixel_ratio": a["grey"] / max(1, a["px"])}

    res = summary(acc[threshold])
    if len(ths) > 1:
        res["sweep"] = {f"{t:g}": summary(acc[t]) for t in ths[1:]}
    return res


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="facebook/sam3")
    ap.add_argument("--template", default="name", help="key in sam3_bank.TEMPLATES or literal template")
    ap.add_argument("--azimuths", default="0,135")
    ap.add_argument("--holdout_file", default=None)
    ap.add_argument("--holdout_frac", type=float, default=0.10)
    ap.add_argument("--min_count", type=int, default=8, help="names seen >= this often in TRAIN get their own E")
    ap.add_argument("--negatives", type=int, default=3)
    ap.add_argument("--neg_mode", choices=["random", "hard", "mixed"], default="random",
                    help="hard = co-occurring / text-similar names absent from the object (SAM 3 style); "
                         "mixed = --negatives of each kind")
    ap.add_argument("--syn_cos", type=float, default=0.85, help="hard negatives closer than this to a present name are skipped")
    ap.add_argument("--eval_only", action="store_true", help="evaluate --resume bank on the holdout and exit")
    ap.add_argument("--no_e0", action="store_true", help="ignore the shared offset E_0 (per-name offsets only)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr_name", type=float, default=1e-2)
    ap.add_argument("--lr_e0", type=float, default=1e-3)
    ap.add_argument("--neg_weight", type=float, default=0.5)
    ap.add_argument("--presence_weight", type=float, default=0.5)
    ap.add_argument("--l2", type=float, default=1e-3, help="pull E[name] toward 0 (keeps rare names near E_0)")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--eval_thresholds", default="0.4,0.5,0.6",
                    help="extra score thresholds evaluated from the same pass (equal-FP comparison)")
    ap.add_argument("--keep_uncertain", action="store_true", help="also use uncertain parts as positives")
    ap.add_argument("--eval_images", type=int, default=240, help="holdout images used for the per-epoch eval")
    ap.add_argument("--max_train_images", type=int, default=None)
    ap.add_argument("--max_prompts", type=int, default=12,
                    help="prompts per step (positives are subsampled; keeps the [N, Q, h, w] mask logits bounded)")
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None, help="bank.pt to continue from (E_0/E copied; schedule restarts)")
    ap.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    ap.add_argument("--wandb_project", default="segvigen-sam3")
    ap.add_argument("--wandb_entity", default=None)
    ap.add_argument("--wandb_name", default=None, help="Display name (default: out dir basename)")
    ap.add_argument("--wandb_id", default=None, help="Run id to resume (default: out dir basename)")
    ap.add_argument("--wandb_mode", default="online", choices=["online", "offline"])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    template = sb.TEMPLATES.get(args.template, args.template)
    azimuths = [f"az{float(a):g}" for a in args.azimuths.split(",") if a.strip()]

    processor, model = load_sam3(args.model, device)
    for p in model.parameters():
        p.requires_grad_(False)

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

    if args.resume:
        prev = sb.ConceptBank.load(args.resume, device)
        E_0 = prev.E_0.clone().float().requires_grad_(True)
        E = torch.zeros(len(bank_names), D, device=device)
        pi = prev.index()
        for i, n in enumerate(bank_names):
            if n in pi:
                E[i] = prev.E[pi[n]].float()
        E.requires_grad_(True)
    else:
        E_0 = torch.zeros(D, device=device, requires_grad=True)
        E = torch.zeros(len(bank_names), D, device=device, requires_grad=True)
    name_idx = {n: i for i, n in enumerate(bank_names)}

    def offsets_for(names: list[str]) -> torch.Tensor:
        rows = []
        base = torch.zeros_like(E_0) if args.no_e0 else E_0
        for n in names:
            i = name_idx.get(n)
            rows.append(base + (E[i] if i is not None else torch.zeros_like(E_0)))
        return torch.stack(rows)

    opt = torch.optim.Adam([{"params": [E], "lr": args.lr_name}, {"params": [E_0], "lr": args.lr_e0}])

    print("loading samples ...", flush=True)
    train_imgs = list_images(args.dataset_root, train_objs, azimuths)
    hold_imgs = list_images(args.dataset_root, hold_objs, azimuths)
    if args.max_train_images:
        rng.shuffle(train_imgs)
        train_imgs = train_imgs[:args.max_train_images]
    hold_samples = [ImageSample(args.dataset_root, o, az, not args.keep_uncertain) for o, az in hold_imgs[:args.eval_images]]

    # fixed negatives per object (same across epochs / eval for comparability)
    def own_names(obj: str) -> set[str]:
        with open(os.path.join(args.dataset_root, obj, "names.json"), encoding="utf-8") as f:
            return {n.strip() for n in json.load(f) if n and n.strip()}

    def draw_random(obj: str, k: int) -> list[str]:
        own = own_names(obj)
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
    with torch.no_grad():
        vecs = []
        for k in range(0, len(bank_names), 64):
            tf, am = sb.text_features_batch(processor, model, bank_names[k:k + 64], device)
            m = am.float().unsqueeze(-1)
            vecs.append(((tf.pooler_output.float() * m).sum(1) / m.sum(1).clamp(min=1)).cpu())
        tvec = F.normalize(torch.cat(vecs), dim=1)
        cos = (tvec @ tvec.T).numpy()

    def draw_hard(obj: str, k: int) -> list[str]:
        own = own_names(obj)
        idx = [name_idx[n] for n in own if n in name_idx]
        if not idx:
            return draw_random(obj, k)
        cand = [i for i, n in enumerate(bank_names) if n not in own and cos[i, idx].max() <= args.syn_cos]
        if not cand:
            return draw_random(obj, k)
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

    all_objs = set(train_objs) | set(hold_objs)
    neg_random = {o: draw_random(o, args.negatives) for o in all_objs}
    neg_hard = {o: draw_hard(o, args.negatives) for o in all_objs}
    if args.neg_mode == "mixed":
        negatives = {o: list(dict.fromkeys(neg_hard[o] + neg_random[o])) for o in all_objs}
    else:
        negatives = neg_hard if args.neg_mode == "hard" else neg_random
    ex = hold_objs[0]
    print(f"negatives ({args.neg_mode}); e.g. {ex[:8]} own={sorted(own_names(ex))[:6]} "
          f"random={neg_random[ex]} hard={neg_hard[ex]}", flush=True)

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
                keys = ("miou", "bind_rate", "false_positive_rate", "false_positive_rate_hard",
                        "part_bound_rate", "grey_pixel_ratio")
                row = {f"holdout/{k}": v for k, v in kw.items() if k in keys and v is not None}
                for t, sub in (kw.get("sweep") or {}).items():
                    row.update({f"holdout_t{t}/{k}": v for k, v in sub.items() if k in keys and v is not None})
                run.log(row, step=step)

    def fmt(d: dict) -> str:
        def r(v):
            return round(v, 4) if isinstance(v, float) else ({k: r(x) for k, x in v.items()} if isinstance(v, dict) else v)
        return json.dumps({k: r(v) for k, v in d.items()})

    sweep = [float(t) for t in args.eval_thresholds.split(",") if t.strip()]
    eval_kw = dict(chunk=args.max_prompts, hard_negatives=neg_hard, thresholds=sweep)

    zero_bank = lambda names: torch.zeros(len(names), D, device=device)
    if args.eval_only:
        if not args.resume:
            raise SystemExit("--eval_only needs --resume <bank.pt>")
        ev = evaluate(processor, model, hold_samples, template, offsets_for, device, args.threshold, neg_random, **eval_kw)
        print(f"holdout, bank {args.resume}:", fmt(ev), flush=True)
        log_row(kind="eval", epoch=-1, step=0, bank=args.resume, **ev)
        if run is not None:
            run.finish()
        return
    base_eval = evaluate(processor, model, hold_samples, template, zero_bank, device, args.threshold, neg_random, **eval_kw)
    print("holdout, no bank:", fmt(base_eval), flush=True)
    log_row(kind="eval", epoch=0, step=0, bank=False, **base_eval)

    total_steps = args.epochs * len(train_imgs)
    step = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        order = list(train_imgs)
        rng.shuffle(order)
        run_loss, run_n = 0.0, 0
        for o, az in order:
            step += 1
            # cosine decay to 0.1x over the whole run
            frac = step / max(1, total_steps)
            scale = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac))
            opt.param_groups[0]["lr"] = args.lr_name * scale
            opt.param_groups[1]["lr"] = args.lr_e0 * scale

            s = ImageSample(args.dataset_root, o, az, not args.keep_uncertain)
            if not s.pos_names:
                continue
            negs = negatives.get(o, [])
            pos = list(s.pos_names)
            cap = max(1, args.max_prompts - len(negs))
            if len(pos) > cap:
                pos = rng.sample(pos, cap)
            s.pos_names = pos
            names = s.pos_names + negs
            prompts = [sb.fill_template(template, n, s.obj_name) for n in names]
            vis = sb.encode_image(processor, model, s.image, device)
            tf, am = sb.text_features_batch(processor, model, prompts, device)
            offs = offsets_for(names)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                out = sb.run_prompts(model, vis, tf, am, offs)
            union = sb.batch_soft_union(out).float()                      # [N, h, w]
            hw = union.shape[-2:]
            n_pos = len(s.pos_names)
            pos_loss = torch.stack([bce_dice(union[i], soft_gt(torch.from_numpy(s.gts[n]).to(device), hw))
                                    for i, n in enumerate(s.pos_names)]).mean()
            loss = pos_loss
            if negs:
                neg_loss = F.binary_cross_entropy(union[n_pos:].clamp(1e-6, 1 - 1e-6), torch.zeros_like(union[n_pos:]))
                loss = loss + args.neg_weight * neg_loss
            if out.presence_logits is not None:
                tgt = torch.cat([torch.ones(n_pos), torch.zeros(len(negs))]).to(device)
                pres = F.binary_cross_entropy_with_logits(out.presence_logits.float().view(-1), tgt)
                loss = loss + args.presence_weight * pres
            if args.l2 > 0:
                used = [name_idx[n] for n in s.pos_names if n in name_idx]
                if used:
                    loss = loss + args.l2 * E[used].pow(2).sum(1).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_loss += float(loss.item()); run_n += 1
            if step % 50 == 0:
                el = time.time() - t0
                print(f"[{el / 60:6.1f} min] epoch {epoch} step {step}/{total_steps} loss {run_loss / run_n:.4f} "
                      f"|E_0| {E_0.norm().item():.3f} |E| {E.norm(dim=1).mean().item():.3f} "
                      f"({el / step:.2f} s/img, eta {(total_steps - step) * el / step / 60:.0f} min)", flush=True)
                log_row(kind="train", epoch=epoch, step=step, loss=run_loss / run_n,
                        e0_norm=E_0.norm().item(), e_norm=E.norm(dim=1).mean().item(),
                        vram_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2) if device == "cuda" else 0)
                run_loss, run_n = 0.0, 0
            if step % args.save_every == 0:
                sb.ConceptBank(E_0.detach(), bank_names, E.detach(), template=template, base=args.model,
                               meta={"epoch": epoch, "step": step, "min_count": args.min_count, "partial": True}
                               ).save(os.path.join(args.out, "bank.pt"))

        bank = sb.ConceptBank(E_0.detach(), bank_names, E.detach(), template=template, base=args.model,
                              meta={"epoch": epoch, "step": step, "min_count": args.min_count, "args": vars(args)})
        bank.save(os.path.join(args.out, f"bank_epoch{epoch}.pt"))
        bank.save(os.path.join(args.out, "bank.pt"))
        ev = evaluate(processor, model, hold_samples, template, offsets_for, device, args.threshold, neg_random, **eval_kw)
        print(f"holdout after epoch {epoch}:", fmt(ev), flush=True)
        log_row(kind="eval", epoch=epoch, step=step, bank=True, **ev)

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
    with torch.no_grad():
        todo = sorted(all_names) + sorted(obj_names)
        for k in range(0, len(todo), 64):
            chunk = todo[k:k + 64]
            tf, am = sb.text_features_batch(processor, model, chunk, device)
            pooled = tf.pooler_output.float()                                   # [n, L, D]
            m = am.float().unsqueeze(-1)
            mean = (pooled * m).sum(1) / m.sum(1).clamp(min=1)
            for i, n in enumerate(chunk):
                off = (E_0 + (E[name_idx[n]] if n in name_idx else 0)).detach()
                cache[n] = (mean[i] + off).cpu()
    torch.save({"dim": D, "names": sorted(all_names), "objects": sorted(obj_names), "vectors": cache,
                "template": template, "note": "mean over text tokens of SAM3 projected text features + bank offset"},
               os.path.join(args.out, "text_cache.pt"))
    if run is not None:
        run.finish()
    print("done", flush=True)


if __name__ == "__main__":
    main()
