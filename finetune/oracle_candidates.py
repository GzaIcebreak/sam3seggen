"""Phase-0 oracle for Mask RankGNN (docs/superpowers/specs/2026-09-09-mask-rankgnn-design.md).

Extracts SAM3 (name, query) candidates, paints three ways (baseline / oracle-select /
oracle-pixel), and decomposes the remaining error into salvageable vs miss.

    /root/autodl-tmp/envs/sam3/bin/python finetune/oracle_candidates.py \\
        --dataset_root /root/autodl-tmp/datasets/pv \\
        --split_file /root/autodl-tmp/runs/v5_ce_lora/split.json \\
        --resume /root/autodl-tmp/runs/v5_ce_lora/bank_epoch2.pt \\
        --decoder_lora 8 --lora_scope mask,text \\
        --lora_file /root/autodl-tmp/runs/v5_ce_lora/decoder_lora_epoch2.pt \\
        --out /root/autodl-tmp/runs/oracle_v5
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

SCORE_MIN = 0.05
AREA_MIN = 16
SELECT_PS = (0.6, 0.7, 0.8, 0.9)
TAUS = (0.0, 0.3, 0.5)
GATE_BASE = {"pixel_acc": 0.626, "pixel_wrong": 0.159, "pixel_unassigned": 0.215}
GATE_TOL = 0.005


def _paint_stats(painted: torch.Tensor, gt_lab: torch.Tensor, valid: torch.Tensor,
                 names: list[str], pms: dict) -> dict:
    n_valid = int(valid.sum().item())
    n_right = int((valid & (painted == gt_lab)).sum().item())
    n_wrong = int((valid & (painted >= 0) & (painted != gt_lab)).sum().item())
    cpart = [((painted == names.index(nm)) & pm).sum().item() / px >= 0.5
             for nm, pm, px in pms.values()]
    return {"cpx": n_valid, "cacc": n_right, "cwrong": n_wrong, "cpart": cpart}


def _acc_bucket() -> dict:
    return {"cpx": 0, "cacc": 0, "cwrong": 0, "cpart": [], "images": 0,
            "n_cand": [], "best_iou_score": [], "best_in_low": []}


def _add_paint(a: dict, st: dict) -> None:
    a["cpx"] += st["cpx"]
    a["cacc"] += st["cacc"]
    a["cwrong"] += st["cwrong"]
    a["cpart"].extend(st["cpart"])
    a["images"] += 1


def _summarize_paint(a: dict) -> dict:
    cpx = max(1, a["cpx"])
    return {
        "images": a["images"],
        "pixel_acc": a["cacc"] / cpx,
        "pixel_wrong": a["cwrong"] / cpx,
        "pixel_unassigned": 1.0 - (a["cacc"] + a["cwrong"]) / cpx,
        "painted_part_rate": float(np.mean(a["cpart"])) if a["cpart"] else 0.0,
    }


def _fmt_paint(d: dict) -> str:
    return (f"acc {d['pixel_acc']:.3f} wrong {d['pixel_wrong']:.3f} "
            f"unassigned {d['pixel_unassigned']:.3f} parts {d['painted_part_rate']:.3f}")


def _label_maps(s: ImageSample, names: list[str], device: str):
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
    gt_lab = torch.where(fg, lut.to(device)[ids_t.clamp(min=0, max=len(lut) - 1)],
                         torch.full_like(ids_t, -1))
    valid = gt_lab >= 0
    return ids_t, fg, gt_lab, valid, pms


@torch.no_grad()
def forward_names(processor, model, vis, names: list[str], obj_name: str | None,
                  template: str, bank, device: str, chunk: int):
    """One SAM3 pass over `names`. Returns scores [N,Q], masks [N,Q,h,w], bias [N] or None."""
    scores, masks, biases = [], [], []
    for k0 in range(0, len(names), chunk):
        part = names[k0:k0 + chunk]
        prompts = [sb.fill_template(template, n, obj_name) for n in part]
        tf, am = sb.text_features_batch(processor, model, prompts, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            out, bias = sb.bank_forward(model, vis, tf, am, bank, part, None)
        scores.append(sb.batch_scores(out, bias).float())
        masks.append(out.pred_masks.float())
        if bias is not None:
            biases.append(bias.float())
        del out
    sc = torch.cat(scores, 0)
    mk = torch.cat(masks, 0)
    bs = torch.cat(biases, 0) if biases else None
    return sc, mk, bs


def name_maps_from(scores: torch.Tensor, masks: torch.Tensor,
                   log_weight: torch.Tensor | None = None) -> torch.Tensor:
    sc = scores.clamp(min=1e-6)
    L = masks + sc.log()[:, :, None, None]
    if log_weight is not None:
        L = L + log_weight[:, :, None, None]
    return torch.logsumexp(L, dim=1)


def extract_candidates(scores: torch.Tensor, masks: torch.Tensor, names: list[str],
                       fg: torch.Tensor, gt_lab: torch.Tensor, size: tuple[int, int],
                       per_name_cap: int = 12):
    """Keep score>0.05 queries, upsample, clip, score against GT. Returns list of dicts + cover [N,H,W]."""
    n, q = scores.shape
    h, w = size
    cover = torch.zeros((n, h, w), dtype=torch.bool, device=scores.device)
    cands = []
    for i in range(n):
        keep = (scores[i] > SCORE_MIN).nonzero(as_tuple=False).flatten()
        if keep.numel() == 0:
            continue
        order = keep[scores[i, keep].argsort(descending=True)]
        order = order[:per_name_cap]
        up = F.interpolate(masks[i, order].sigmoid()[None], size=size,
                           mode="bilinear", align_corners=False)[0] > 0.5
        up = up & fg
        gt = gt_lab == i
        gt_n = int(gt.sum().item())
        for j, qi in enumerate(order.tolist()):
            m = up[j]
            area = int(m.sum().item())
            if area < AREA_MIN:
                continue
            inter = int((m & gt).sum().item())
            prec = inter / area
            rec = inter / gt_n if gt_n else 0.0
            union = area + gt_n - inter
            iou = inter / union if union else 0.0
            ys, xs = torch.where(m)
            y0, y1 = int(ys.min()), int(ys.max())
            x0, x1 = int(xs.min()), int(xs.max())
            cy, cx = float(ys.float().mean()), float(xs.float().mean())
            cands.append({
                "name_i": i, "name": names[i], "q": qi,
                "score": float(scores[i, qi]),
                "area": area, "area_frac": area / max(1, int(fg.sum().item())),
                "box": (x0 / w, y0 / h, (x1 + 1) / w, (y1 + 1) / h),
                "centroid": (cx / w, cy / h),
                "precision": prec, "recall": rec, "iou": iou,
            })
            cover[i] |= m
    return cands, cover


def dump_image(path: str, s, names, cands, scores, bank):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    text = []
    for c in cands:
        if bank is not None:
            v = bank.offset(c["name"]).detach().float().cpu().numpy()
        else:
            v = np.zeros(256, dtype=np.float32)
        text.append(v)
    np.savez_compressed(
        path,
        obj=s.obj, az=s.az, names=np.array(names, dtype=object),
        name_i=np.array([c["name_i"] for c in cands], dtype=np.int16),
        q=np.array([c["q"] for c in cands], dtype=np.int16),
        score=np.array([c["score"] for c in cands], dtype=np.float32),
        precision=np.array([c["precision"] for c in cands], dtype=np.float32),
        recall=np.array([c["recall"] for c in cands], dtype=np.float32),
        iou=np.array([c["iou"] for c in cands], dtype=np.float32),
        area_frac=np.array([c["area_frac"] for c in cands], dtype=np.float32),
        box=np.array([c["box"] for c in cands], dtype=np.float32) if cands else np.zeros((0, 4), np.float32),
        centroid=np.array([c["centroid"] for c in cands], dtype=np.float32) if cands else np.zeros((0, 2), np.float32),
        text_vec=np.stack(text).astype(np.float32) if text else np.zeros((0, 256), np.float32),
        n_query=int(scores.shape[1]),
    )


@torch.no_grad()
def run_split(tag: str, samples, processor, model, bank, device: str, template: str,
              chunk: int, hard_ids: set[str], dump_dir: str | None, max_images: int | None):
    buckets = {
        "base": {tau: _acc_bucket() for tau in TAUS},
        "overlay": _acc_bucket(),
        "select": {p: {tau: _acc_bucket() for tau in TAUS} for p in SELECT_PS},
        "pixel": _acc_bucket(),
        "hard_base": {tau: _acc_bucket() for tau in TAUS},
        "hard_select": {p: {tau: _acc_bucket() for tau in TAUS} for p in SELECT_PS},
        "hard_pixel": _acc_bucket(),
    }
    decomp = {"right": 0, "wrong_salv": 0, "wrong_miss": 0, "grey_salv": 0, "grey_miss": 0, "cpx": 0}
    steal = Counter()
    t0 = time.time()
    n_done = 0
    for s in samples:
        if max_images and n_done >= max_images:
            break
        if not s.gts:
            continue
        names = list(s.gts.keys())
        ids_t, fg, gt_lab, valid, pms = _label_maps(s, names, device)
        vis = sb.encode_image(processor, model, s.image, device)
        scores, masks, _bias = forward_names(
            processor, model, vis, names, s.obj_name, template, bank, device, chunk)
        maps = name_maps_from(scores, masks)
        bg = bank.bg_logit() if bank is not None else 0.0
        h, w = s.ids.shape
        size = (h, w)
        is_hard = s.obj in hard_ids

        for tau in TAUS:
            pa = sb.paint_argmax(maps, fg, bg, tau, 1.0)
            st = _paint_stats(pa, gt_lab, valid, names, pms)
            _add_paint(buckets["base"][tau], st)
            if is_hard:
                _add_paint(buckets["hard_base"][tau], st)

        unions = (scores.max(1).values > 0.5)
        # overlay needs hard unions at full res; reuse candidate extraction path via score gate
        overlay_u = torch.zeros((len(names), h, w), dtype=torch.bool, device=device)
        keep_any = scores > 0.5
        for i in range(len(names)):
            qs = keep_any[i].nonzero(as_tuple=False).flatten()
            if qs.numel() == 0:
                continue
            up = F.interpolate(masks[i, qs].sigmoid()[None], size=size,
                               mode="bilinear", align_corners=False)[0] > 0.5
            overlay_u[i] = up.any(0) & fg
        ov = sb.paint(overlay_u, fg)
        _add_paint(buckets["overlay"], _paint_stats(ov, gt_lab, valid, names, pms))

        cands, cover = extract_candidates(scores, masks, names, fg, gt_lab, size)
        if dump_dir:
            dump_image(os.path.join(dump_dir, f"{s.obj}_{s.az}.npz"), s, names, cands, scores, bank)

        pix_cover = cover.gather(0, gt_lab.clamp(min=0)[None])[0] & valid

        # oracle-pixel
        op = torch.full_like(gt_lab, -1)
        op[pix_cover] = gt_lab[pix_cover]
        st_pix = _paint_stats(op, gt_lab, valid, names, pms)
        _add_paint(buckets["pixel"], st_pix)
        if is_hard:
            _add_paint(buckets["hard_pixel"], st_pix)

        kept_idx = {p: [] for p in SELECT_PS}
        for k, c in enumerate(cands):
            for p in SELECT_PS:
                if c["precision"] >= p:
                    kept_idx[p].append(k)
        for p in SELECT_PS:
            lw = torch.full_like(scores, float("-inf"))
            for c in cands:
                if c["precision"] >= p:
                    lw[c["name_i"], c["q"]] = 0.0
            sm = name_maps_from(scores, masks, lw)
            for tau in TAUS:
                pa = sb.paint_argmax(sm, fg, bg, tau, 1.0)
                st = _paint_stats(pa, gt_lab, valid, names, pms)
                _add_paint(buckets["select"][p][tau], st)
                if is_hard:
                    _add_paint(buckets["hard_select"][p][tau], st)

        # error breakdown on deployment point (argmax tau=0.5)
        pa = sb.paint_argmax(maps, fg, bg, 0.5, 1.0)
        n_valid = int(valid.sum().item())
        decomp["cpx"] += n_valid
        right = valid & (pa == gt_lab)
        wrong = valid & (pa >= 0) & (pa != gt_lab)
        grey = valid & (pa < 0)
        decomp["right"] += int(right.sum().item())
        decomp["wrong_salv"] += int((wrong & pix_cover).sum().item())
        decomp["wrong_miss"] += int((wrong & ~pix_cover).sum().item())
        decomp["grey_salv"] += int((grey & pix_cover).sum().item())
        decomp["grey_miss"] += int((grey & ~pix_cover).sum().item())

        steal_lab = pa[wrong]
        steal_gt = gt_lab[wrong]
        for a, b in zip(steal_lab.tolist(), steal_gt.tolist()):
            steal[(names[a], names[b])] += 1

        buckets["base"][0.5]["n_cand"].append(len(cands))
        by_name: dict[str, list] = {}
        for c in cands:
            by_name.setdefault(c["name"], []).append(c)
        for nm, cs in by_name.items():
            best = max(cs, key=lambda x: x["iou"])
            buckets["base"][0.5]["best_iou_score"].append(best["score"])
            buckets["base"][0.5]["best_in_low"].append(SCORE_MIN <= best["score"] < 0.5)

        n_done += 1
        if n_done % 20 == 0 or n_done == 1:
            dt = time.time() - t0
            print(f"  [{tag}] {n_done}/{len(samples)}  {dt / 60:.1f} min  "
                  f"last cands={len(cands)}", flush=True)
        del vis, scores, masks, maps

    cpx = max(1, decomp["cpx"])
    decomp_r = {k: (decomp[k] / cpx if k != "cpx" else decomp[k]) for k in decomp}
    salv = decomp_r["wrong_salv"] + decomp_r["grey_salv"]
    low = buckets["base"][0.5]["best_in_low"]
    out = {
        "tag": tag,
        "images": n_done,
        "baseline": {f"{t:g}": _summarize_paint(buckets["base"][t]) for t in TAUS},
        "overlay": _summarize_paint(buckets["overlay"]),
        "oracle_select": {f"{p:g}": {f"{t:g}": _summarize_paint(buckets["select"][p][t])
                                     for t in TAUS} for p in SELECT_PS},
        "oracle_pixel": _summarize_paint(buckets["pixel"]),
        "decomp_tau0.5": decomp_r,
        "salvageable": salv,
        "n_cand_mean": float(np.mean(buckets["base"][0.5]["n_cand"])) if buckets["base"][0.5]["n_cand"] else 0.0,
        "best_cand_score_in_05_05": float(np.mean(low)) if low else 0.0,
        "steal_top": [{"pred": a, "gt": b, "px": c} for (a, b), c in steal.most_common(20)],
    }
    if hard_ids:
        out["hard"] = {
            "baseline": {f"{t:g}": _summarize_paint(buckets["hard_base"][t]) for t in TAUS},
            "oracle_select": {f"{p:g}": {f"{t:g}": _summarize_paint(buckets["hard_select"][p][t])
                                         for t in TAUS} for p in SELECT_PS},
            "oracle_pixel": _summarize_paint(buckets["hard_pixel"]),
        }
    return out


def print_report(res: dict) -> None:
    print(f"\n== {res['tag']}  images={res['images']}  cands/img={res['n_cand_mean']:.1f} "
          f"best-score-in-[0.05,0.5)={res['best_cand_score_in_05_05']:.3f}", flush=True)
    print(f"  overlay @0.5          {_fmt_paint(res['overlay'])}", flush=True)
    for t in TAUS:
        print(f"  baseline argmax τ={t:g}  {_fmt_paint(res['baseline'][f'{t:g}'])}", flush=True)
    for p in SELECT_PS:
        d = res["oracle_select"][f"{p:g}"]["0.5"]
        print(f"  oracle-select p={p:g} τ=0.5 {_fmt_paint(d)}", flush=True)
    print(f"  oracle-pixel          {_fmt_paint(res['oracle_pixel'])}", flush=True)
    d = res["decomp_tau0.5"]
    print(f"  decomp τ=0.5  right={d['right']:.3f}  "
          f"wrong salv/miss={d['wrong_salv']:.3f}/{d['wrong_miss']:.3f}  "
          f"grey salv/miss={d['grey_salv']:.3f}/{d['grey_miss']:.3f}  "
          f"salvageable={res['salvageable']:.3f}", flush=True)
    if res["steal_top"]:
        top = ", ".join(f"{x['pred']}→{x['gt']}({x['px']})" for x in res["steal_top"][:8])
        print(f"  steal  {top}", flush=True)
    if "hard" in res:
        hb = res["hard"]["baseline"]["0.5"]
        hs = res["hard"]["oracle_select"]["0.8"]["0.5"]
        print(f"  hard baseline τ=0.5   {_fmt_paint(hb)}", flush=True)
        print(f"  hard select p=0.8     {_fmt_paint(hs)}", flush=True)
        print(f"  hard oracle-pixel     {_fmt_paint(res['hard']['oracle_pixel'])}", flush=True)


def gate_holdout(res: dict) -> str:
    b = res["baseline"]["0.5"]
    ok_replay = all(abs(b[k] - GATE_BASE[k]) <= GATE_TOL for k in GATE_BASE)
    sel = res["oracle_select"]["0.8"]["0.5"]["pixel_acc"]
    pix = res["oracle_pixel"]["pixel_acc"]
    salv = res["salvageable"]
    print("\n== GATE (holdout, v5_ce_lora ep2 replay)", flush=True)
    print(f"  replay τ=0.5  acc {b['pixel_acc']:.3f} wrong {b['pixel_wrong']:.3f} "
          f"unassigned {b['pixel_unassigned']:.3f}  "
          f"target {GATE_BASE['pixel_acc']:.3f}/{GATE_BASE['pixel_wrong']:.3f}/{GATE_BASE['pixel_unassigned']:.3f}  "
          f"{'OK' if ok_replay else 'MISMATCH — stop and check weights'}", flush=True)
    print(f"  oracle-select(0.8) acc {sel:.3f}  need >= 0.72", flush=True)
    print(f"  salvageable {salv:.3f}  need >= 0.12", flush=True)
    print(f"  oracle-pixel acc {pix:.3f}  fail if <= 0.70", flush=True)
    if not ok_replay:
        return "replay_fail"
    if pix <= 0.70 or salv < 0.06:
        return "fail"
    if sel >= 0.72 and salv >= 0.12:
        return "pass"
    sel7 = res["oracle_select"]["0.7"]["0.5"]["pixel_acc"]
    print(f"  mid: select(0.7) acc {sel7:.3f}", flush=True)
    if sel7 >= 0.72 and salv >= 0.12:
        return "pass_p07"
    return "fail"


def load_model(args, device: str):
    processor, model = load_sam3(args.model, device)
    for p in model.parameters():
        p.requires_grad_(False)
    if args.decoder_lora > 0:
        from lora import load_lora_state_dict
        params, n_wrap = sb.inject_decoder_lora(
            model, r=args.decoder_lora, alpha=args.lora_alpha, scope=args.lora_scope)
        print(f"decoder LoRA: r={args.decoder_lora} scope={args.lora_scope} "
              f"{n_wrap} linears / {sum(p.numel() for p in params)} params", flush=True)
        if args.lora_file:
            load_lora_state_dict(model, torch.load(args.lora_file, map_location=device, weights_only=False))
            print(f"loaded LoRA from {args.lora_file}", flush=True)
    bank = sb.ConceptBank.load(args.resume, device) if args.resume else None
    if bank is not None:
        print("bank:", bank.describe(), flush=True)
    return processor, model, bank


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="/root/autodl-tmp/sam3seggen/weights/facebook/sam3")
    ap.add_argument("--split_file", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--hard_file", default=None)
    ap.add_argument("--extra_eval", action="append", default=[], help="NAME=ROOT")
    ap.add_argument("--azimuths", default="0,135")
    ap.add_argument("--eval_images", type=int, default=240)
    ap.add_argument("--max_prompts", type=int, default=12)
    ap.add_argument("--template", default="name")
    ap.add_argument("--decoder_lora", type=int, default=0)
    ap.add_argument("--lora_alpha", type=float, default=None)
    ap.add_argument("--lora_scope", default="mask,text")
    ap.add_argument("--lora_file", default=None)
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--max_images", type=int, default=None, help="debug cap per split")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    template = sb.TEMPLATES.get(args.template, args.template)
    azimuths = [f"az{float(a):g}" for a in args.azimuths.split(",") if a.strip()]

    with open(args.split_file, encoding="utf-8") as f:
        split = json.load(f)
    hold_objs = list(split["holdout"])
    hold_imgs = list_images(args.dataset_root, hold_objs, azimuths)[:args.eval_images]
    print(f"holdout: {len(hold_objs)} objects / {len(hold_imgs)} images", flush=True)
    hold_samples = [ImageSample(args.dataset_root, o, az, True) for o, az in hold_imgs]
    hard_ids = read_id_file(args.hard_file) & set(hold_objs)
    if hard_ids:
        print(f"hard slice: {len(hard_ids)} holdout objects", flush=True)

    extras: list[tuple[str, list]] = []
    for spec in args.extra_eval:
        nm, root = spec.split("=", 1)
        objs = list_objects(root)
        imgs = list_images(root, objs, azimuths)
        extras.append((nm, [ImageSample(root, o, az, True) for o, az in imgs]))
        print(f"extra '{nm}': {len(objs)} objects / {len(imgs)} images", flush=True)

    processor, model, bank = load_model(args, device)
    dump_dir = os.path.join(args.out, "feats") if args.dump else None

    hold = run_split("holdout", hold_samples, processor, model, bank, device, template,
                     args.max_prompts, hard_ids, dump_dir, args.max_images)
    print_report(hold)
    with open(os.path.join(args.out, "holdout.json"), "w") as f:
        json.dump(hold, f)
    if "hard" in hold:
        with open(os.path.join(args.out, "hard.json"), "w") as f:
            json.dump(hold["hard"], f)

    extras_out = {}
    for nm, samples in extras:
        r = run_split(nm, samples, processor, model, bank, device, template,
                      args.max_prompts, set(), dump_dir, args.max_images)
        print_report(r)
        extras_out[nm] = r
        with open(os.path.join(args.out, f"{nm}.json"), "w") as f:
            json.dump(r, f)

    verdict = gate_holdout(hold)
    summary = {"verdict": verdict, "holdout": hold, "extra": extras_out}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f)
    print(f"\nVERDICT: {verdict}", flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
