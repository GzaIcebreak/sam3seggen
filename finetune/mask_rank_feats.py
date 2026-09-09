"""Cache the Mask RankGNN's candidate features so training never has to touch SAM3.

One npz per image, holding the top-K candidates per name: node features, the pairwise edge features,
and the GT precision / recall each candidate would earn. Rows are grouped by name and sorted by SAM3
score, so row n*K is always that name's score-top-1 and the ranker's job is visible as "which row does
it move to the front".

    python finetune/mask_rank_feats.py --dataset_root /root/autodl-tmp/datasets/pv \
        --model weights/facebook/sam3 --split_file /root/autodl-tmp/runs/v5_ce_lora/split.json \
        --part train --resume /root/autodl-tmp/datasets/concept_bank_v3/bank.pt \
        --out /root/autodl-tmp/runs/mask_rank_v3/feats_train

`--part` picks train / holdout from the split file; `--root` (with no split file) takes every object of
an external set. Uses the same ImageSample / forward path as concept_bank.py, so the candidates are the
ones the deployed painter actually sees.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import sam3_bank as sb
from concept_bank import ImageSample, list_images, list_objects
from mask_rank import candidate_features, select_topk
from oracle_candidates import low_res_labels
from sam3_to_2dmap import load_sam3


@torch.no_grad()
def image_feats(processor, model, s: ImageSample, bank, template: str, device: str, chunk: int,
                topk: int) -> dict | None:
    names = list(s.gts.keys())
    if not names:
        return None
    vis = sb.encode_image(processor, model, s.image, device)
    fpn = vis.fpn_hidden_states[0][0]                                   # [256, h, w], mask resolution
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
    logits, scores = torch.cat(logits, 0), torch.cat(scores, 0)
    tvecs = torch.cat(tvecs, 0)

    ids = torch.from_numpy(s.ids.astype(np.int64)).to(device)
    h, w = logits.shape[-2:]
    fg_low = torch.nn.functional.interpolate(
        (ids >= 0).float()[None, None], size=(h, w), mode="nearest")[0, 0] > 0.5
    lab = low_res_labels(ids, names, s.names, (h, w))

    idx = select_topk(scores, topk)
    f = candidate_features(logits, scores, fpn, fg_low, tvecs, idx)

    n, k = idx.shape
    sel = logits.gather(1, idx[:, :, None, None].expand(-1, -1, h, w)) > 0
    prec, rec = [], []
    for i in range(n):
        g = lab == i
        inter = (sel[i] & g).sum((1, 2)).float()
        prec.append(inter / sel[i].sum((1, 2)).clamp(min=1).float())
        rec.append(inter / g.sum().clamp(min=1).float())
    return {"node": f["node"].half().cpu().numpy(), "edge": f["edge"].half().cpu().numpy(),
            "prec": torch.cat(prec).cpu().numpy().astype(np.float16),
            "rec": torch.cat(rec).cpu().numpy().astype(np.float16),
            "query": f["query"].cpu().numpy().astype(np.int16),
            "names": np.array(names, dtype=object), "topk": np.int16(k)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", default=None)
    ap.add_argument("--split_file", default=None)
    ap.add_argument("--part", choices=["train", "holdout"], default="train")
    ap.add_argument("--root", default=None, help="external set: every object under this root")
    ap.add_argument("--model", default="facebook/sam3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", default=None, help="bank.pt")
    ap.add_argument("--decoder_lora", type=int, default=0)
    ap.add_argument("--lora_alpha", type=float, default=None)
    ap.add_argument("--lora_scope", default="mask,text")
    ap.add_argument("--lora_file", default=None)
    ap.add_argument("--azimuths", default="0,135")
    ap.add_argument("--max_images", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=12)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--template", default=None)
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
            print(f"loaded LoRA from {args.lora_file}", flush=True)

    bank = sb.ConceptBank.load(args.resume, device) if args.resume else None
    template = args.template or "{name}"
    if bank is not None:
        print("bank:", bank.describe(), flush=True)
        if args.template is None and isinstance(bank.meta, dict) and bank.meta.get("templates"):
            template = list(bank.meta["templates"])[0]

    azimuths = [f"az{float(a):g}" for a in args.azimuths.split(",") if a.strip()]
    if args.root:
        root = args.root
        imgs = list_images(root, list_objects(root), azimuths)
    else:
        import json
        with open(args.split_file, encoding="utf-8") as f:
            objs = json.load(f)[args.part]
        root = args.dataset_root
        imgs = list_images(root, objs, azimuths)
    if args.max_images:
        imgs = imgs[:args.max_images]
    print(f"{len(imgs)} images -> {args.out} (topk {args.topk})", flush=True)

    t0, n_done, n_cand = time.time(), 0, 0
    for i, (o, az) in enumerate(imgs):
        path = os.path.join(args.out, f"{o}_{az}.npz")
        if os.path.exists(path):
            continue
        s = ImageSample(root, o, az, not args.keep_uncertain)
        d = image_feats(processor, model, s, bank, template, device, args.chunk, args.topk)
        if d is None:
            continue
        np.savez_compressed(path, **d)
        n_done += 1
        n_cand += d["node"].shape[0]
        if (i + 1) % 200 == 0:
            el = time.time() - t0
            print(f"  {i + 1}/{len(imgs)}  {el / 60:.1f} min  eta {el / max(1, i + 1) * (len(imgs) - i - 1) / 60:.1f} min",
                  flush=True)
    print(f"wrote {n_done} files, {n_cand / max(1, n_done):.0f} candidates per image, "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
