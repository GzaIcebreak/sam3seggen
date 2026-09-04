"""Stage A benchmark: how well does SAM3 turn our part names into masks, per prompt template?

    <.venv_holo python> finetune/bench_sam3.py --dataset_root E:\data\pv --every 5 \
        --templates name,obj_name,name_of_obj --out E:\data\bench_sam3

For every object x view (render.png + ids.npy) and every unique part name of the object, the
GT is the union of all visible parts with that name. Per prompt we record IoU, boundary F1 and
whether the mask "binds" (IoU >= 0.5). Hard negatives are names from other objects; a
detection on them is a false positive. Part-level binding follows make_samples_a (a part is
bound when its name's mask covers >= --cover of the part), which gives the grey-pixel ratio
the colour maps would end up with.

--concept_bank <bank.pt> evaluates a learned bank (Stage B) with the same protocol.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
from PIL import Image

import sam3_bank as sb
from sam3_to_2dmap import load_sam3


def pick_objects(dataset_root: str, objects_file: str | None, every: int, limit: int | None) -> list[str]:
    if objects_file:
        with open(objects_file, encoding="utf-8") as f:
            names = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    else:
        names = sorted(d for d in os.listdir(dataset_root)
                       if os.path.exists(os.path.join(dataset_root, d, "views", "az0", "ids.npy")))
    names = names[::max(1, every)]
    return names[:limit] if limit else names


def summarize(records: list[dict], negatives: list[dict], parts: list[dict]) -> dict:
    n = len(records)
    bound = sum(r["iou"] >= 0.5 for r in records)
    detected = sum(r["detected"] for r in records)
    fp = sum(r["detected"] for r in negatives)
    fg = sum(p["px"] for p in parts)
    grey = sum(p["px"] for p in parts if not p["bound"])
    return {
        "prompts": n,
        "bind_rate": bound / max(1, n),
        "detect_rate": detected / max(1, n),
        "miou": float(np.mean([r["iou"] for r in records])) if n else 0.0,
        "boundary_f1": float(np.mean([r["bf1"] for r in records])) if n else 0.0,
        "neg_prompts": len(negatives),
        "false_positive_rate": fp / max(1, len(negatives)),
        "parts": len(parts),
        "part_bound_rate": sum(p["bound"] for p in parts) / max(1, len(parts)),
        "grey_pixel_ratio": grey / max(1, fg),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--objects_file", default=None)
    ap.add_argument("--every", type=int, default=5, help="take every k-th object (default 5 -> ~400 objects)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--azimuths", default="0,135")
    ap.add_argument("--templates", default="name,obj_name,name_of_obj",
                    help="comma list of keys in sam3_bank.TEMPLATES, or literal templates with {name}/{object}")
    ap.add_argument("--concept_bank", default=None)
    ap.add_argument("--no_e0", action="store_true", help="with --concept_bank: apply per-name offsets only")
    ap.add_argument("--model", default="facebook/sam3")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--cover", type=float, default=0.5)
    ap.add_argument("--negatives", type=int, default=5)
    ap.add_argument("--min_count", type=int, default=8, help="negatives are drawn from names seen >= this often")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default=None, help="output file stem (default: template key or bank name)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = load_sam3(args.model, device)
    bank = sb.ConceptBank.load(args.concept_bank, device) if args.concept_bank else None

    templates = []
    for t in args.templates.split(","):
        t = t.strip()
        if not t:
            continue
        templates.append((t, sb.TEMPLATES.get(t, t)))
    if bank is not None:
        templates = [(args.tag or os.path.splitext(os.path.basename(args.concept_bank))[0], bank.template)]

    azimuths = [f"az{float(a):g}" for a in args.azimuths.split(",") if a.strip()]
    objects = pick_objects(args.dataset_root, args.objects_file, args.every, args.limit)
    vocab = sb.vocabulary(args.dataset_root)
    neg_pool = sorted(n for n, c in vocab.items() if c >= args.min_count)
    rng = random.Random(args.seed)
    print(f"{len(objects)} objects x {len(azimuths)} views; templates {[k for k, _ in templates]}; "
          f"negative pool {len(neg_pool)} names", flush=True)

    per_t = {k: {"records": [], "negatives": [], "parts": []} for k, _ in templates}
    t0 = time.time()
    n_img = 0
    for oi, oname in enumerate(objects):
        odir = os.path.join(args.dataset_root, oname)
        names, obj_name, uncertain = sb.load_object_labels(odir)
        own = set(n.strip() for n in names if n and n.strip())
        negs = rng.sample([n for n in neg_pool if n not in own], min(args.negatives, len(neg_pool)))
        for az in azimuths:
            vdir = os.path.join(odir, "views", az)
            rp, ip = os.path.join(vdir, "render.png"), os.path.join(vdir, "ids.npy")
            if not (os.path.exists(rp) and os.path.exists(ip)):
                continue
            ids = np.load(ip)
            image = Image.open(rp)
            h, w = ids.shape
            gts = sb.gt_unions(ids, names)
            if not gts:
                continue
            gts_t = {n: torch.from_numpy(m).to(device) for n, m in gts.items()}
            vis = sb.encode_image(processor, model, image, device)
            n_img += 1

            for key, tmpl in templates:
                unions: dict[str, torch.Tensor] = {}
                for n, gt in gts_t.items():
                    prompt = sb.fill_template(tmpl, n, obj_name)
                    tf, am = sb.text_features(processor, model, prompt, device)
                    off = bank.offset(n, use_e0=not args.no_e0) if bank is not None else None
                    with torch.no_grad():
                        out = sb.run_prompt(model, vis, tf, am, off)
                        um = sb.union_mask(out, (h, w), args.threshold)
                        best = float(sb.instance_scores(out).max().item())
                    unions[n] = um
                    per_t[key]["records"].append({
                        "object": oname, "view": az, "name": n, "prompt": prompt, "iou": sb.iou(um, gt),
                        "bf1": sb.boundary_f1(um, gt), "detected": bool(um.any()), "score": best,
                        "gt_px": int(gt.sum().item()), "pred_px": int(um.sum().item()),
                        "uncertain": any(uncertain[p] for p, nm in enumerate(names) if nm.strip() == n)})
                for n in negs:
                    prompt = sb.fill_template(tmpl, n, obj_name)
                    tf, am = sb.text_features(processor, model, prompt, device)
                    off = bank.offset(n, use_e0=not args.no_e0) if bank is not None else None
                    with torch.no_grad():
                        out = sb.run_prompt(model, vis, tf, am, off)
                        um = sb.union_mask(out, (h, w), args.threshold)
                    per_t[key]["negatives"].append({"object": oname, "view": az, "name": n, "prompt": prompt,
                                                    "detected": bool(um.any()), "pred_px": int(um.sum().item())})
                # part-level binding as make_samples_a would do it with these unions
                for p, nm in enumerate(names):
                    nm = (nm or "").strip()
                    pm = torch.from_numpy(ids == p).to(device)
                    px = int(pm.sum().item())
                    if px == 0 or nm not in unions:
                        continue
                    cov = (unions[nm] & pm).sum().item() / px
                    per_t[key]["parts"].append({"object": oname, "view": az, "part": p, "name": nm, "px": px,
                                                "cover": cov, "bound": cov >= args.cover})
        if (oi + 1) % 20 == 0 or oi + 1 == len(objects):
            el = time.time() - t0
            print(f"[{el / 60:5.1f} min] {oi + 1}/{len(objects)} objects, {n_img} images, "
                  f"{el / max(1, n_img):.2f} s/img", flush=True)

    summary = {}
    for key, tmpl in templates:
        d = per_t[key]
        s = summarize(d["records"], d["negatives"], d["parts"])
        s["template"] = tmpl
        s["concept_bank"] = args.concept_bank
        summary[key] = s
        # per-name table for the ambiguity list
        by_name = defaultdict(list)
        for r in d["records"]:
            by_name[r["name"]].append(r["iou"])
        names_tab = sorted(({"name": n, "n": len(v), "miou": float(np.mean(v)), "bind_rate": float(np.mean([x >= 0.5 for x in v]))}
                            for n, v in by_name.items()), key=lambda r: (r["miou"], -r["n"]))
        with open(os.path.join(args.out, f"{key}.json"), "w", encoding="utf-8") as f:
            json.dump({"summary": s, "args": vars(args), "names": names_tab, **d}, f, ensure_ascii=False)
        print(f"== {key}: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()})}")

    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
