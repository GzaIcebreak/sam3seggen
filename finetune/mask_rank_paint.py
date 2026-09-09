"""End-to-end evaluation of the Mask RankGNN: repaint the holdout with the learned candidate weights.

The offline rank quality in mask_rank.py says whether the ranker sorts candidates better; only this
says whether that reaches the picture. Same forward pass, same painter, same F metrics as
oracle_candidates.py, with one term added to the logsumexp:

    L_n(p) = logsumexp_q [ mask_logit_{n,q}(p) + log score_{n,q} + lambda * log keep_{n,q} ]

Candidates outside the re-ranked top-K keep a fixed weight `--tail` (1 = untouched, 0 = only the top-K
competes), swept alongside lambda so the data decides how much of the tail is worth keeping.

    python finetune/mask_rank_paint.py --dataset_root /root/autodl-tmp/datasets/pv \
        --model weights/facebook/sam3 --split_file /root/autodl-tmp/runs/v5_ce_lora/split.json \
        --resume /root/autodl-tmp/datasets/concept_bank_v3/bank.pt \
        --rank_model /root/autodl-tmp/runs/mask_rank_v3/rank.pt \
        --hard_file /root/autodl-tmp/datasets/pv_hard.txt \
        --extra_eval mw=/root/autodl-tmp/datasets/makerworld/concept_bank_clean_v2/objects \
        --out /root/autodl-tmp/runs/mask_rank_v3/paint
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import sam3_bank as sb
from concept_bank import ImageSample, list_images, list_objects, read_id_file
from mask_rank import candidate_features, load_model, select_topk, shuffle_text
from oracle_candidates import add_acc, new_acc, paint_stats, summary
from sam3_to_2dmap import load_sam3


@torch.no_grad()
def keep_weights(processor, model, ranker, s: ImageSample, bank, template: str, device: str,
                 chunk: int, topk: int, shuffle: bool):
    """One forward pass -> (outputs pieces needed to repaint, keep [N, k], top-K indices)."""
    names = list(s.gts.keys())
    if not names:
        return None
    vis = sb.encode_image(processor, model, s.image, device)
    fpn = vis.fpn_hidden_states[0][0]
    use_qv = bank is not None and bank.nn_cos > 0 and bank.tvec is not None

    logits, scores, tvecs = [], [], []
    for k0 in range(0, len(names), chunk):
        part = names[k0:k0 + chunk]
        prompts = [sb.fill_template(template, n, s.obj_name) for n in part]
        tf, am = sb.text_features_batch(processor, model, prompts, device)
        qv = sb.text_query_vecs(tf, am)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            out, bias = sb.bank_forward(model, vis, tf, am, bank, part, qv if use_qv else None)
        logits.append(out.pred_masks.clone())
        scores.append(sb.batch_scores(out, bias).float())
        shift = (bank.offsets(part, qv if use_qv else None) if bank is not None
                 else torch.zeros(len(part), qv.shape[1], device=device))
        tvecs.append((qv.float() + shift.float()).detach())
        del out
    logits, scores, tvecs = torch.cat(logits, 0), torch.cat(scores, 0), torch.cat(tvecs, 0)

    ids = torch.from_numpy(s.ids.astype(np.int64)).to(device)
    h, w = logits.shape[-2:]
    fg_low = F.interpolate((ids >= 0).float()[None, None], size=(h, w), mode="nearest")[0, 0] > 0.5
    idx = select_topk(scores, topk)
    f = candidate_features(logits, scores, fpn, fg_low, tvecs, idx)
    node = shuffle_text(f["node"], idx.shape[1]) if shuffle else f["node"]
    keep = ranker(node, f["edge"]).view(idx.shape)
    return {"names": names, "logits": logits, "scores": scores, "idx": idx, "keep": keep,
            "ids": ids, "fg": ids >= 0}


def repaint(r: dict, lam: float, tail: float, bg_logit: float, tau: float, temp: float) -> torch.Tensor:
    n, q = r["scores"].shape
    lw = torch.full((n, q), float(np.log(max(tail, 1e-6))), device=r["scores"].device)
    lw.scatter_(1, r["idx"], lam * r["keep"].clamp(min=1e-6).log())
    L = r["logits"].float() + r["scores"].clamp(min=1e-6).log()[:, :, None, None] + lw[:, :, None, None]
    return sb.paint_argmax(torch.logsumexp(L, 1), r["fg"], bg_logit, tau, temp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--model", default="facebook/sam3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split_file", required=True)
    ap.add_argument("--rank_model", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--decoder_lora", type=int, default=0)
    ap.add_argument("--lora_alpha", type=float, default=None)
    ap.add_argument("--lora_scope", default="mask,text")
    ap.add_argument("--lora_file", default=None)
    ap.add_argument("--hard_file", default=None)
    ap.add_argument("--extra_eval", action="append", default=[])
    ap.add_argument("--azimuths", default="0,135")
    ap.add_argument("--eval_images", type=int, default=240)
    ap.add_argument("--max_images", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=12)
    ap.add_argument("--taus", default="0,0.3,0.5")
    ap.add_argument("--lambdas", default="0,0.5,1,2")
    ap.add_argument("--tails", default="1,0.3,0")
    ap.add_argument("--assign_temp", type=float, default=1.0)
    ap.add_argument("--template", default=None)
    ap.add_argument("--shuffle_text", action="store_true",
                    help="control: permute the names' concept vectors before ranking")
    ap.add_argument("--keep_uncertain", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = load_sam3(args.model, device)
    for p in model.parameters():
        p.requires_grad_(False)
    if args.decoder_lora > 0:
        from lora import load_lora_state_dict
        sb.inject_decoder_lora(model, r=args.decoder_lora, alpha=args.lora_alpha, scope=args.lora_scope)
        if args.lora_file:
            load_lora_state_dict(model, torch.load(args.lora_file, map_location=device,
                                                   weights_only=False))
    ranker, cfg = load_model(args.rank_model, device)
    topk = int(cfg["topk"])
    print(f"ranker {args.rank_model}: topk {topk}, edge {not cfg['no_edge']}, "
          f"text {not cfg['no_text']}, vis {not cfg['no_vis']}", flush=True)

    bank = sb.ConceptBank.load(args.resume, device) if args.resume else None
    template = args.template or "{name}"
    if bank is not None:
        print("bank:", bank.describe(), flush=True)
        if args.template is None and isinstance(bank.meta, dict) and bank.meta.get("templates"):
            template = list(bank.meta["templates"])[0]
    bg_logit = bank.bg_logit() if bank is not None else 0.0

    azimuths = [f"az{float(a):g}" for a in args.azimuths.split(",") if a.strip()]
    with open(args.split_file, encoding="utf-8") as f:
        hold_objs = json.load(f)["holdout"]
    hold = [ImageSample(args.dataset_root, o, az, not args.keep_uncertain)
            for o, az in list_images(args.dataset_root, hold_objs, azimuths)[:args.eval_images]]
    hard_ids = read_id_file(args.hard_file) & set(hold_objs)
    sets = {"holdout": hold}
    for spec in args.extra_eval:
        nm, root = spec.split("=", 1)
        sets[nm] = [ImageSample(root, o, az, not args.keep_uncertain)
                    for o, az in list_images(root, list_objects(root), azimuths)]

    taus = [float(t) for t in args.taus.split(",") if t.strip()]
    lams = [float(t) for t in args.lambdas.split(",") if t.strip()]
    tails = [float(t) for t in args.tails.split(",") if t.strip()]
    combos = [(lam, tail) for lam in lams for tail in tails if not (lam == 0 and tail != 1)]

    for nm, samples in sets.items():
        if args.max_images:
            samples = samples[:args.max_images]
        slices = {"hard": hard_ids} if nm == "holdout" and hard_ids else {}
        acc = {(sl, lam, tail, tau): new_acc() for sl in ["all", *slices]
               for lam, tail in combos for tau in taus}
        t0 = time.time()
        print(f"\n--- {nm}: {len(samples)} images, {len(combos)} (lambda, tail) x {len(taus)} tau",
              flush=True)
        for i, s in enumerate(samples):
            r = keep_weights(processor, model, ranker, s, bank, template, device, args.chunk,
                             topk, args.shuffle_text)
            if r is None:
                continue
            lut = torch.full((max(1, len(s.names)),), -1, dtype=torch.long, device=device)
            for p, x in enumerate(s.names):
                x = (x or "").strip()
                if x in s.gts:
                    lut[p] = r["names"].index(x)
            gt_lab = torch.where(r["fg"], lut[r["ids"].clamp(min=0, max=len(lut) - 1)],
                                 torch.full_like(r["ids"], -1))
            sls = ["all"] + [sl for sl, objs in slices.items() if s.obj in objs]
            for lam, tail in combos:
                for tau in taus:
                    pa = repaint(r, lam, tail, bg_logit, tau, args.assign_temp)
                    for sl in sls:
                        add_acc(acc[(sl, lam, tail, tau)], *paint_stats(pa, gt_lab))
            del r
            if (i + 1) % 60 == 0:
                print(f"  {i + 1}/{len(samples)}, {time.time() - t0:.0f}s", flush=True)

        res = {sl: {f"lam{lam:g}_tail{tail:g}@{tau:g}": summary(acc[(sl, lam, tail, tau)])
                    for lam, tail in combos for tau in taus} for sl in ["all", *slices]}
        with open(os.path.join(args.out, f"{nm}.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        for sl, d in res.items():
            print(f"\n=== {nm}/{sl} ===\n{'setting':<24}{'涂对':>8}{'涂错':>8}{'未涂':>8}")
            for k, v in d.items():
                print(f"{k:<24}{v['pixel_acc']:>8.3f}{v['pixel_wrong']:>8.3f}{v['pixel_unassigned']:>8.3f}")


if __name__ == "__main__":
    main()
