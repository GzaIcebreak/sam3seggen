"""End-to-end evaluation of the Mask RankGNN: repaint the holdout with the learned candidate weights.

The offline rank quality in mask_rank.py says whether the ranker sorts candidates better; only this
says whether that reaches the picture. Two painters share one SAM3 forward pass per image:

* `argmax`  (v5 operator, pixel-level): one more term inside the logsumexp,
      L_n(p) = logsumexp_q [ mask_logit_{n,q}(p) + log score_{n,q} + lambda * log keep_{n,q} ]
  then every pixel goes to the highest name. Kept as the reference that REPORT_concept_bank_v5_eval §7
  showed fragments by construction.

* `select`  (mask-level, the deployment candidate): keep the candidates whose ranker weight is at
  least `kappa`, take their hard masks (sigmoid > 0.5 at full resolution, like sam3_to_2dmap does with
  `--threshold`) and overlay them as units. `order=small` unions each name's kept masks and paints the
  smallest union first (exactly v3's rule, so `--select score --kappa 0.4` reproduces v3); `order=keep`
  paints the individual masks in descending ranker weight, never overwriting, so an overlap is settled
  by which candidate the ranker trusts more instead of by area. Nothing is decided per pixel, so the
  painted blocks are SAM3's masks or nothing.

Besides 涂对/涂错/未涂 the report counts what the eye sees: 4-connected blocks per image relative to the
ground truth (`comp_ratio`, 1 = as many blocks as the label map) and blocks smaller than 0.5 % of the
silhouette (`small`, the speckle).

    python finetune/mask_rank_paint.py --dataset_root /root/autodl-tmp/datasets/pv \
        --model weights/facebook/sam3 --split_file /root/autodl-tmp/runs/v5_ce_lora/split.json \
        --resume /root/autodl-tmp/datasets/concept_bank_v3/bank.pt \
        --rank_model /root/autodl-tmp/runs/mask_rank_v3/rank.pt \
        --hard_file /root/autodl-tmp/datasets/pv_hard.txt \
        --extra_eval mw=/root/autodl-tmp/datasets/makerworld/concept_bank_clean_v2/objects \
        --out /root/autodl-tmp/runs/mask_rank_v3/paint --dump 12
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
from sam3_to_2dmap import load_sam3

SMALL_FRAC = 0.005


# --------------------------------------------------------------------------- forward

@torch.no_grad()
def keep_weights(processor, model, ranker, s: ImageSample, bank, template: str, device: str,
                 chunk: int, topk: int, shuffle: bool):
    """One forward pass -> the pieces needed to repaint, plus keep [N, k] over the top-K indices."""
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


# --------------------------------------------------------------------------- painters

def repaint_argmax(r: dict, lam: float, tail: float, bg_logit: float, tau: float) -> torch.Tensor:
    n, q = r["scores"].shape
    lw = torch.full((n, q), float(np.log(max(tail, 1e-6))), device=r["scores"].device)
    lw.scatter_(1, r["idx"], lam * r["keep"].clamp(min=1e-6).log())
    L = r["logits"].float() + r["scores"].clamp(min=1e-6).log()[:, :, None, None] + lw[:, :, None, None]
    return sb.paint_argmax(torch.logsumexp(L, 1), r["fg"], bg_logit, tau, 1.0)


paint_select = sb.paint_select        # mask-level painters live in sam3_bank so sam3_to_2dmap shares them
paint_hybrid = sb.paint_hybrid


# --------------------------------------------------------------------------- metrics

def n_components(lab: torch.Tensor, n_fg: int) -> tuple[int, int]:
    """(blocks, blocks smaller than SMALL_FRAC of the silhouette) among the painted labels."""
    comp = sb.connected_components(lab)
    if not bool((comp >= 0).any()):
        return 0, 0
    _, counts = torch.unique(comp[comp >= 0], return_counts=True)
    return int(counts.numel()), int((counts < SMALL_FRAC * n_fg).sum().item())


def paint_stats(painted: torch.Tensor, gt_lab: torch.Tensor, valid: torch.Tensor, n_fg: int,
                gt_comp: int) -> dict:
    right = int((valid & (painted == gt_lab)).sum().item())
    wrong = int((valid & (painted >= 0) & (painted != gt_lab)).sum().item())
    parts = []
    for j in torch.unique(gt_lab[valid]).tolist():
        pm = gt_lab == j
        parts.append(float(((painted == j) & pm).sum().item() / pm.sum().item()) >= 0.5)
    comp, small = n_components(painted, n_fg)
    return {"px": int(valid.sum().item()), "right": right, "wrong": wrong, "parts": parts,
            "comp_ratio": comp / max(1, gt_comp), "small": small}


def new_acc() -> dict:
    return {"px": 0, "right": 0, "wrong": 0, "parts": [], "comp_ratio": [], "small": [], "images": 0}


def add_acc(a: dict, st: dict) -> None:
    a["px"] += st["px"]
    a["right"] += st["right"]
    a["wrong"] += st["wrong"]
    a["parts"].extend(st["parts"])
    a["comp_ratio"].append(st["comp_ratio"])
    a["small"].append(st["small"])
    a["images"] += 1


def summary(a: dict) -> dict:
    px = max(1, a["px"])
    return {"images": a["images"], "pixel_acc": a["right"] / px, "pixel_wrong": a["wrong"] / px,
            "pixel_unassigned": 1.0 - (a["right"] + a["wrong"]) / px,
            "painted_part_rate": float(np.mean(a["parts"])) if a["parts"] else 0.0,
            "comp_ratio": float(np.mean(a["comp_ratio"])) if a["comp_ratio"] else 0.0,
            "small_blocks": float(np.mean(a["small"])) if a["small"] else 0.0}


# --------------------------------------------------------------------------- visual dump

def colorize(lab: torch.Tensor, fg: torch.Tensor) -> np.ndarray:
    pal = np.array([[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
                    [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212],
                    [0, 128, 128], [220, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
                    [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128]],
                   dtype=np.uint8)
    l = lab.cpu().numpy()
    f = fg.cpu().numpy()
    img = np.full((*l.shape, 3), 255, np.uint8)
    img[f] = 120
    m = l >= 0
    img[m] = pal[l[m] % len(pal)]
    return img


def dump_panel(path: str, image, panels: list[tuple[str, np.ndarray]], width: int = 320) -> None:
    from PIL import Image, ImageDraw
    tiles = []
    for title, arr in [("image", np.asarray(image.convert("RGB")))] + panels:
        im = Image.fromarray(arr)
        im = im.resize((width, int(width * im.height / im.width)))
        ImageDraw.Draw(im).text((4, 4), title, fill=(0, 0, 0))
        tiles.append(im)
    h = max(t.height for t in tiles)
    sheet = Image.new("RGB", (width * len(tiles), h), (255, 255, 255))
    for i, t in enumerate(tiles):
        sheet.paste(t, (i * width, 0))
    sheet.save(path)


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--model", default="facebook/sam3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split_file", required=True)
    ap.add_argument("--rank_model", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--hard_file", default=None)
    ap.add_argument("--extra_eval", action="append", default=[])
    ap.add_argument("--azimuths", default="0,135")
    ap.add_argument("--eval_images", type=int, default=240)
    ap.add_argument("--max_images", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=12)
    ap.add_argument("--taus", default="0,0.3", help="argmax painter: assignment floor")
    ap.add_argument("--lambdas", default="0,1", help="argmax painter: weight on log keep")
    ap.add_argument("--tails", default="1,0", help="argmax painter: weight of non-top-K queries")
    ap.add_argument("--kappas", default="0.3,0.5,0.7", help="select painter: ranker floor")
    ap.add_argument("--score_kappas", default="0.4", help="select painter with SAM3 score (v3)")
    ap.add_argument("--drops", default="0.2,0.35", help="hybrid: drop v3 members with keep below")
    ap.add_argument("--adds", default="0.6,0.75", help="hybrid: add candidates with keep at least")
    ap.add_argument("--min_comp", type=float, default=0.0,
                    help="also report every select setting after clean_components(min_frac)")
    ap.add_argument("--template", default=None)
    ap.add_argument("--shuffle_text", action="store_true",
                    help="control: permute the names' concept vectors before ranking")
    ap.add_argument("--keep_uncertain", action="store_true")
    ap.add_argument("--dump", type=int, default=0, help="save panels for the first N images per set")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = load_sam3(args.model, device)
    for p in model.parameters():
        p.requires_grad_(False)
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
    hard_ids = read_id_file(args.hard_file) & set(hold_objs) if args.hard_file else set()
    sets = {"holdout": hold}
    for spec in args.extra_eval:
        nm, root = spec.split("=", 1)
        sets[nm] = [ImageSample(root, o, az, not args.keep_uncertain)
                    for o, az in list_images(root, list_objects(root), azimuths)]

    floats = lambda s: [float(t) for t in s.split(",") if t.strip()]
    settings: list[tuple[str, callable]] = []
    for kap in floats(args.score_kappas):
        fn = lambda r, kap=kap: paint_select(r, "score", kap, "small", False)
        settings.append((f"v3 score@{kap:g} small", fn))
        if args.min_comp > 0:
            settings.append((f"v3 score@{kap:g} small clean{args.min_comp:g}",
                             lambda r, fn=fn: sb.clean_components(fn(r), r["fg"], args.min_comp)))
    for lam in floats(args.lambdas):
        for tail in floats(args.tails):
            if lam == 0 and tail != 1:
                continue
            for tau in floats(args.taus):
                settings.append((f"argmax lam{lam:g} tail{tail:g} tau{tau:g}",
                                 lambda r, lam=lam, tail=tail, tau=tau: repaint_argmax(r, lam, tail, bg_logit, tau)))
    for kap in floats(args.kappas):
        for order in ("small", "keep"):
            for fb in (False, True):
                tag = f"select k{kap:g} {order}{' +fb' if fb else ''}"
                fn = lambda r, kap=kap, order=order, fb=fb: paint_select(r, "keep", kap, order, fb)
                settings.append((tag, fn))
                if args.min_comp > 0:
                    settings.append((tag + f" clean{args.min_comp:g}",
                                     lambda r, fn=fn: sb.clean_components(fn(r), r["fg"], args.min_comp)))
    # one mask per name: SAM3's top-1 vs the ranker's pick (Venice-H1's protocol); kappa > 1 with fallback = argmax only
    settings.append(("top1 score", lambda r: paint_select(r, "score", 1.01, "small", True)))
    settings.append(("top1 rank", lambda r: paint_select(r, "keep", 1.01, "small", True)))
    if args.min_comp > 0:
        settings.append((f"top1 rank clean{args.min_comp:g}",
                         lambda r: sb.clean_components(paint_select(r, "keep", 1.01, "small", True), r["fg"], args.min_comp)))
    score_k = floats(args.score_kappas)[0]
    for drop in floats(args.drops):
        for add in floats(args.adds):
            for order in ("small", "keep"):
                tag = f"hybrid d{drop:g} a{add:g} {order}"
                fn = lambda r, drop=drop, add=add, order=order: paint_hybrid(r, score_k, drop, add, order)
                settings.append((tag, fn))
                if args.min_comp > 0:
                    settings.append((tag + f" clean{args.min_comp:g}",
                                     lambda r, fn=fn: sb.clean_components(fn(r), r["fg"], args.min_comp)))
    print(f"{len(settings)} settings", flush=True)

    for nm, samples in sets.items():
        if args.max_images:
            samples = samples[:args.max_images]
        slices = {"hard": hard_ids} if nm == "holdout" and hard_ids else {}
        acc = {(sl, tag): new_acc() for sl in ["all", *slices] for tag, _ in settings}
        dump_dir = os.path.join(args.out, f"dump_{nm}")
        if args.dump:
            os.makedirs(dump_dir, exist_ok=True)
        t0 = time.time()
        print(f"\n--- {nm}: {len(samples)} images", flush=True)
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
            valid = gt_lab >= 0
            n_fg = int(r["fg"].sum().item())
            gt_comp, _ = n_components(gt_lab, n_fg)
            sls = ["all"] + [sl for sl, objs in slices.items() if s.obj in objs]
            panels = [("gt", colorize(gt_lab, r["fg"]))] if i < args.dump else None
            for tag, fn in settings:
                pa = fn(r)
                st = paint_stats(pa, gt_lab, valid, n_fg, gt_comp)
                for sl in sls:
                    add_acc(acc[(sl, tag)], st)
                if panels is not None and (tag.startswith("v3") or tag in DUMP_TAGS):
                    panels.append((tag, colorize(pa, r["fg"])))
            if panels is not None:
                dump_panel(os.path.join(dump_dir, f"{s.obj}_{s.az}.png"), s.image, panels)
            del r
            if (i + 1) % 60 == 0:
                print(f"  {i + 1}/{len(samples)}, {time.time() - t0:.0f}s", flush=True)

        res = {sl: {tag: summary(acc[(sl, tag)]) for tag, _ in settings} for sl in ["all", *slices]}
        with open(os.path.join(args.out, f"{nm}.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        for sl, d in res.items():
            print(f"\n=== {nm}/{sl} ({d[settings[0][0]]['images']} images) ===\n"
                  f"{'setting':<34}{'涂对':>7}{'涂错':>7}{'未涂':>7}{'块比':>7}{'碎块':>7}{'部件率':>8}")
            for k, v in d.items():
                print(f"{k:<34}{v['pixel_acc']:>7.3f}{v['pixel_wrong']:>7.3f}{v['pixel_unassigned']:>7.3f}"
                      f"{v['comp_ratio']:>7.2f}{v['small_blocks']:>7.1f}{v['painted_part_rate']:>8.3f}",
                      flush=True)


DUMP_TAGS = {"argmax lam0 tail1 tau0", "top1 score", "top1 rank", "select k0.5 small",
             "hybrid d0.1 a0.9 small", "hybrid d0.2 a0.75 small"}


if __name__ == "__main__":
    main()
