"""Stage 0 of the Mask RankGNN plan: how much is left on the table by SAM3's per-candidate scores?

    envs/sam3/bin/python finetune/oracle_candidates.py \
        --dataset_root /root/autodl-tmp/datasets/pv \
        --model weights/facebook/sam3 \
        --split_file /root/autodl-tmp/runs/v5_ce_lora/split.json \
        --resume /root/autodl-tmp/runs/v5_ce_lora/bank_epoch2.pt \
        --decoder_lora 8 --lora_scope mask,text \
        --lora_file /root/autodl-tmp/runs/v5_ce_lora/decoder_lora_epoch2.pt \
        --hard_file /root/autodl-tmp/datasets/pv_hard.txt \
        --extra_eval mw=/root/autodl-tmp/datasets/makerworld/concept_bank_clean_v2/objects \
        --out /root/autodl-tmp/runs/oracle_v5

The deployment operator since v5 is `paint_argmax` over

    L_n(p) = logsumexp_q [ mask_logit_{n,q}(p) + log score_{n,q} ]

so every (name, query) candidate enters the per-pixel competition weighted by a score SAM3 assigned
without looking at the other candidates in the image. This script measures what a perfect per-candidate
weight would buy, and splits the remaining error into "a correct candidate exists but lost" (a ranking
problem, which a RankGNN can attack) and "no correct candidate exists" (a detection problem, which it
cannot). Paintings compared, all with the same painter and the same forward pass:

    base_all      every query, original scores                 -> must reproduce the run's argmax row
    base_cand     only candidates (score > --cand_thr)         -> checks the candidate pool is complete
    oracle_sel@p  only candidates with GT precision >= p       -> ceiling for a per-candidate reweighting
    oracle_pixel  a pixel counts as right if any candidate of its own name covers it -> absolute ceiling

Writes <out>/<set>.json (per-set metrics, error split, name-pair table) and prints a summary table.
With --dump it also caches per-image candidate features for stage 1 (finetune/mask_rank.py).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn.functional as F

import sam3_bank as sb
from concept_bank import ImageSample, list_images, list_objects, read_id_file
from sam3_to_2dmap import load_sam3


# --------------------------------------------------------------------------- candidates

class Candidates:
    """Every (name, query) pair of one image: the mask logits and the score each one carries into

        L_n(p) = logsumexp_q [ mask_logit_{n,q}(p) + log score_{n,q} ]

    The whole query set is kept, not a score-thresholded subset: SAM3's lowest score on a real image is
    ~3e-3, so no query drops out of the logsumexp by itself, and a thresholded pool would make the oracle
    measure the threshold rather than the ranking. Logits stay at the mask head's resolution, which is
    what paint_argmax interpolates from, so re-deriving the maps from a subset is one logsumexp."""

    def __init__(self, logits: torch.Tensor, scores: torch.Tensor):
        self.logits = logits                                     # [N, Q, h, w], model dtype
        self.scores = scores                                     # [N, Q], float
        self.n_names, self.q = logits.shape[0], logits.shape[1]
        self.h, self.w = logits.shape[-2:]

    def __len__(self) -> int:
        return self.n_names * self.q

    def maps(self, keep: torch.Tensor | None = None) -> torch.Tensor:
        """[N, h, w] name logits from the kept candidates; -inf rows where a name keeps none."""
        out = []
        logs = self.scores.clamp(min=1e-6).log()
        for n in range(self.n_names):
            L = self.logits[n].float() + logs[n][:, None, None]
            if keep is not None:
                L = L.masked_fill(~keep[n][:, None, None], float("-inf"))
            out.append(torch.logsumexp(L, 0))
        return torch.stack(out)

    def coverage(self, keep: torch.Tensor | None = None) -> torch.Tensor:
        """[N, h, w] bool: some kept candidate of that name fires on the pixel."""
        out = []
        for n in range(self.n_names):
            m = self.logits[n] > 0
            if keep is not None:
                m = m & keep[n][:, None, None]
            out.append(m.any(0))
        return torch.stack(out)

    def gt_scores(self, lab_low: torch.Tensor) -> dict[str, torch.Tensor]:
        """Per-candidate precision / recall / IoU against the GT of its own name, [N, Q] each."""
        prec, rec, iou, area = [], [], [], []
        for n in range(self.n_names):
            m = self.logits[n] > 0
            g = lab_low == n
            inter = (m & g).sum((1, 2)).float()
            a = m.sum((1, 2)).float()
            b = g.sum().float()
            prec.append(inter / a.clamp(min=1))
            rec.append(inter / b.clamp(min=1))
            iou.append(inter / (a + b - inter).clamp(min=1))
            area.append(a)
        return {"prec": torch.stack(prec), "rec": torch.stack(rec),
                "iou": torch.stack(iou), "area": torch.stack(area)}


def low_res_labels(ids: torch.Tensor, names: list[str], sample_names: list[str], size) -> torch.Tensor:
    """Mask-resolution GT: index into `names`, -1 where the pixel belongs to no prompted name."""
    lut = torch.full((max(1, len(sample_names)),), -1, dtype=torch.long, device=ids.device)
    for p, nm in enumerate(sample_names):
        nm = (nm or "").strip()
        if nm in names:
            lut[p] = names.index(nm)
    small = F.interpolate(ids[None, None].float(), size=size, mode="nearest")[0, 0].long()
    return torch.where(small < 0, torch.full_like(small, -1), lut[small.clamp(min=0, max=len(lut) - 1)])


# --------------------------------------------------------------------------- per-image pass

@torch.no_grad()
def image_pass(processor, model, s: ImageSample, bank, template: str, device: str,
               chunk: int) -> dict | None:
    names = list(s.gts.keys())
    if not names:
        return None
    vis = sb.encode_image(processor, model, s.image, device)
    use_qv = bank is not None and bank.nn_cos > 0 and bank.tvec is not None

    base_maps, logits, scores = [], [], []
    for k0 in range(0, len(names), chunk):
        part = names[k0:k0 + chunk]
        prompts = [sb.fill_template(template, n, s.obj_name) for n in part]
        tf, am = sb.text_features_batch(processor, model, prompts, device)
        qv = sb.text_query_vecs(tf, am) if use_qv else None
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            out, bias = sb.bank_forward(model, vis, tf, am, bank, part, qv)
        base_maps.append(sb.name_logit_maps(out, bias))
        logits.append(out.pred_masks.clone())
        scores.append(sb.batch_scores(out, bias).float())
        del out
    cands = Candidates(torch.cat(logits, 0), torch.cat(scores, 0))
    base_maps = torch.cat(base_maps, 0)                               # [N, h, w]

    ids_t = torch.from_numpy(s.ids.astype(np.int64)).to(device)
    fg = ids_t >= 0
    lut = torch.full((max(1, len(s.names)),), -1, dtype=torch.long, device=device)
    for p, nm in enumerate(s.names):
        nm = (nm or "").strip()
        if nm in s.gts:
            lut[p] = names.index(nm)
    gt_lab = torch.where(fg, lut[ids_t.clamp(min=0, max=len(lut) - 1)], torch.full_like(ids_t, -1))
    lab_low = low_res_labels(ids_t, names, s.names, (cands.h, cands.w))
    return {"sample": s, "names": names, "cands": cands, "base_maps": base_maps, "fg": fg,
            "gt_lab": gt_lab, "lab_low": lab_low, **cands.gt_scores(lab_low)}


# --------------------------------------------------------------------------- metrics

def paint_stats(painted: torch.Tensor, gt_lab: torch.Tensor) -> tuple[int, int, int]:
    valid = gt_lab >= 0
    right = int((valid & (painted == gt_lab)).sum())
    wrong = int((valid & (painted >= 0) & (painted != gt_lab)).sum())
    return int(valid.sum()), right, wrong


def new_acc() -> dict:
    return {"cpx": 0, "right": 0, "wrong": 0, "images": 0}


def add_acc(a: dict, cpx: int, right: int, wrong: int) -> None:
    a["cpx"] += cpx
    a["right"] += right
    a["wrong"] += wrong
    a["images"] += 1


def summary(a: dict) -> dict:
    cpx = max(1, a["cpx"])
    return {"images": a["images"], "pixel_acc": a["right"] / cpx, "pixel_wrong": a["wrong"] / cpx,
            "pixel_unassigned": 1.0 - (a["right"] + a["wrong"]) / cpx}


# --------------------------------------------------------------------------- main

def evaluate_set(processor, model, samples: list[ImageSample], bank, template: str, device: str,
                 args, slices: dict[str, set[str]], dump_dir: str | None) -> dict:
    taus = [float(t) for t in args.taus.split(",") if t.strip()]
    ps = [float(p) for p in args.oracle_p.split(",") if p.strip()]
    bg_logit = bank.bg_logit() if bank is not None else 0.0
    keys = ([f"base_all@{t:g}" for t in taus] + [f"base_cand@{t:g}" for t in taus]
            + [f"oracle_sel{p:g}@{t:g}" for p in ps for t in taus] + ["oracle_pixel"])
    acc = {(sl, k): new_acc() for sl in ["all", *slices] for k in keys}
    split = {sl: Counter() for sl in ["all", *slices]}          # error decomposition, base_all @ args.tau
    pairs: Counter = Counter()                                   # (stolen by, should be) on salvageable pixels
    cand_stats = {"top1_iou": [], "best_iou": [], "best_rank": [], "max_dev": 0.0}
    t0 = time.time()

    for k, s in enumerate(samples):
        r = image_pass(processor, model, s, bank, template, device, args.chunk)
        if r is None:
            continue
        cands, names, gt_lab, fg = r["cands"], r["names"], r["gt_lab"], r["fg"]
        sls = ["all"] + [sl for sl, objs in slices.items() if s.obj in objs]
        prec, iou = r["prec"], r["iou"]

        # how much of the headroom is pure ranking: the score-1 candidate vs the best one
        order = cands.scores.argsort(dim=1, descending=True)
        best_q = iou.argmax(dim=1)
        for n in range(len(names)):
            cand_stats["top1_iou"].append(float(iou[n, order[n, 0]]))
            cand_stats["best_iou"].append(float(iou[n, best_q[n]]))
            cand_stats["best_rank"].append(int((order[n] == best_q[n]).nonzero()[0, 0]))

        maps_cand = cands.maps()
        cand_stats["max_dev"] = max(cand_stats["max_dev"],
                                    float((maps_cand - r["base_maps"]).abs().max()))
        maps_sel = {p: cands.maps(prec >= p) for p in ps}
        cov = cands.coverage()                                   # [N, h, w] at mask resolution
        cov_full = F.interpolate(cov[None].float(), size=fg.shape, mode="nearest")[0] > 0.5
        for tau in taus:
            pa = sb.paint_argmax(r["base_maps"], fg, bg_logit, tau, args.assign_temp)
            for sl in sls:
                add_acc(acc[(sl, f"base_all@{tau:g}")], *paint_stats(pa, gt_lab))
            if abs(tau - args.tau) < 1e-9:
                valid = gt_lab >= 0
                own = torch.zeros_like(valid)
                if len(names):
                    own[valid] = cov_full.gather(0, gt_lab.clamp(min=0)[None])[0][valid]
                miss = valid & (pa != gt_lab)
                for nm_, cond in (("wrong", miss & (pa >= 0)), ("grey", miss & (pa < 0))):
                    for sl in sls:
                        split[sl][f"{nm_}_salvageable"] += int((cond & own).sum())
                        split[sl][f"{nm_}_undetected"] += int((cond & ~own).sum())
                for sl in sls:
                    split[sl]["right"] += int((valid & (pa == gt_lab)).sum())
                    split[sl]["total"] += int(valid.sum())
                sal = miss & own & (pa >= 0)
                if bool(sal.any()):
                    for (p_i, g_i), n_px in Counter(zip(pa[sal].tolist(), gt_lab[sal].tolist())).most_common(5):
                        pairs[(names[p_i], names[g_i])] += n_px
            pa = sb.paint_argmax(maps_cand, fg, bg_logit, tau, args.assign_temp)
            for sl in sls:
                add_acc(acc[(sl, f"base_cand@{tau:g}")], *paint_stats(pa, gt_lab))
            for p in ps:
                pa = sb.paint_argmax(maps_sel[p], fg, bg_logit, tau, args.assign_temp)
                for sl in sls:
                    add_acc(acc[(sl, f"oracle_sel{p:g}@{tau:g}")], *paint_stats(pa, gt_lab))

        valid = gt_lab >= 0
        hit = torch.zeros_like(valid)
        if len(names):
            hit[valid] = cov_full.gather(0, gt_lab.clamp(min=0)[None])[0][valid]
        for sl in sls:
            add_acc(acc[(sl, "oracle_pixel")], int(valid.sum()), int((valid & hit).sum()), 0)

        if dump_dir:
            np.savez_compressed(
                os.path.join(dump_dir, f"{s.obj}_{s.az}.npz"),
                score=cands.scores.cpu().numpy().astype(np.float32),
                prec=prec.cpu().numpy().astype(np.float32), rec=r["rec"].cpu().numpy().astype(np.float32),
                iou=iou.cpu().numpy().astype(np.float32), area=r["area"].cpu().numpy().astype(np.float32),
                names=np.array(names, dtype=object))
        del r, cands, maps_cand, maps_sel
        if (k + 1) % 20 == 0:
            print(f"  {k + 1}/{len(samples)} images, {time.time() - t0:.0f}s", flush=True)

    out = {sl: {k: summary(acc[(sl, k)]) for k in keys} for sl in ["all", *slices]}
    for sl in out:
        tot = max(1, split[sl]["total"])
        out[sl]["error_split"] = {k: v / tot for k, v in split[sl].items() if k != "total"}
        out[sl]["error_split"]["total_px"] = split[sl]["total"]
    out["candidates"] = {
        "top1_iou": float(np.mean(cand_stats["top1_iou"])) if cand_stats["top1_iou"] else 0.0,
        "best_iou": float(np.mean(cand_stats["best_iou"])) if cand_stats["best_iou"] else 0.0,
        "best_is_top1": float(np.mean([r_ == 0 for r_ in cand_stats["best_rank"]])) if cand_stats["best_rank"] else 0.0,
        "best_rank_median": float(np.median(cand_stats["best_rank"])) if cand_stats["best_rank"] else 0.0,
        "maps_max_dev": cand_stats["max_dev"]}
    out["pairs"] = [{"painted": a, "should_be": b, "px": c} for (a, b), c in pairs.most_common(20)]
    return out


def print_table(tag: str, res: dict, tau: float) -> None:
    print(f"\n=== {tag} ===", flush=True)
    print(f"{'painting':<22}{'涂对':>8}{'涂错':>8}{'未涂':>8}")
    for k, d in res.items():
        if not isinstance(d, dict) or "pixel_acc" not in d:
            continue
        if k.endswith(f"@{tau:g}") or k == "oracle_pixel":
            print(f"{k:<22}{d['pixel_acc']:>8.3f}{d['pixel_wrong']:>8.3f}{d['pixel_unassigned']:>8.3f}")
    es = res.get("error_split", {})
    if es:
        print(f"  error split @tau{tau:g}: right {es.get('right', 0):.3f} | "
              f"wrong salvageable {es.get('wrong_salvageable', 0):.3f} undetected {es.get('wrong_undetected', 0):.3f} | "
              f"grey salvageable {es.get('grey_salvageable', 0):.3f} undetected {es.get('grey_undetected', 0):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--model", default="facebook/sam3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split_file", required=True, help="split.json of the run whose baseline we reproduce")
    ap.add_argument("--resume", default=None, help="bank.pt (omit for bare SAM3)")
    ap.add_argument("--decoder_lora", type=int, default=0)
    ap.add_argument("--lora_alpha", type=float, default=None)
    ap.add_argument("--lora_scope", default="mask,text")
    ap.add_argument("--lora_file", default=None)
    ap.add_argument("--hard_file", default=None)
    ap.add_argument("--extra_eval", action="append", default=[], help="NAME=ROOT")
    ap.add_argument("--azimuths", default="0,135")
    ap.add_argument("--eval_images", type=int, default=240)
    ap.add_argument("--max_images", type=int, default=None, help="smoke test: cap every set")
    ap.add_argument("--chunk", type=int, default=12)
    ap.add_argument("--taus", default="0,0.3,0.5")
    ap.add_argument("--tau", type=float, default=0.5, help="tau the error split is computed at")
    ap.add_argument("--oracle_p", default="0.6,0.7,0.8,0.9")
    ap.add_argument("--assign_temp", type=float, default=1.0)
    ap.add_argument("--template", default=None, help="override bank.meta's template")
    ap.add_argument("--dump", action="store_true", help="cache candidate features for stage 1")
    ap.add_argument("--keep_uncertain", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = load_sam3(args.model, device)
    for p in model.parameters():
        p.requires_grad_(False)
    if args.decoder_lora > 0:
        from lora import load_lora_state_dict
        _, n_wrap = sb.inject_decoder_lora(model, r=args.decoder_lora, alpha=args.lora_alpha,
                                           scope=args.lora_scope)
        print(f"decoder LoRA: r={args.decoder_lora} scope={args.lora_scope} {n_wrap} linears", flush=True)
        if args.lora_file:
            load_lora_state_dict(model, torch.load(args.lora_file, map_location=device, weights_only=False))
            print(f"loaded LoRA from {args.lora_file}", flush=True)

    bank = sb.ConceptBank.load(args.resume, device) if args.resume else None
    template = args.template or "{name}"
    if bank is not None:
        print("bank:", bank.describe(), flush=True)
        if args.template is None and isinstance(bank.meta, dict) and bank.meta.get("templates"):
            template = list(bank.meta["templates"])[0]
    print(f"template: {template!r}, bg_logit {bank.bg_logit() if bank else 0.0:.3f}", flush=True)

    with open(args.split_file, encoding="utf-8") as f:
        hold_objs = json.load(f)["holdout"]
    azimuths = [f"az{float(a):g}" for a in args.azimuths.split(",") if a.strip()]
    hold_imgs = list_images(args.dataset_root, hold_objs, azimuths)[:args.eval_images]
    hold = [ImageSample(args.dataset_root, o, az, not args.keep_uncertain) for o, az in hold_imgs]
    hard_ids = read_id_file(args.hard_file) & set(hold_objs)
    slices = {"hard": hard_ids} if hard_ids else {}
    print(f"holdout: {len(hold)} images from {len(hold_objs)} objects; hard slice {len(hard_ids)} objects",
          flush=True)

    dump_dir = None
    if args.dump:
        dump_dir = os.path.join(args.out, "feats")
        os.makedirs(dump_dir, exist_ok=True)

    sets = {"holdout": hold}
    for spec in args.extra_eval:
        nm, root = spec.split("=", 1)
        imgs = list_images(root, list_objects(root), azimuths)
        sets[nm] = [ImageSample(root, o, az, not args.keep_uncertain) for o, az in imgs]
        print(f"extra eval '{nm}': {len(sets[nm])} images", flush=True)

    for nm, samples in sets.items():
        if args.max_images:
            samples = samples[:args.max_images]
        print(f"\n--- {nm}: {len(samples)} images", flush=True)
        res = evaluate_set(processor, model, samples, bank, template, device, args,
                           slices if nm == "holdout" else {}, dump_dir if nm == "holdout" else None)
        with open(os.path.join(args.out, f"{nm}.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print(f"candidates: {res['candidates']}", flush=True)
        for sl in [s for s in res if s in ("all", "hard")]:
            print_table(f"{nm}/{sl}", res[sl], args.tau)
        if res["pairs"]:
            print("  top salvageable confusions (painted -> should be):",
                  ", ".join(f"{p['painted']}->{p['should_be']}" for p in res["pairs"][:8]), flush=True)


if __name__ == "__main__":
    main()
