"""Write a review markdown for a sample of objects: render + GT part mask + per-part thumbnail and name.

This is the step no automated check can replace. check_names.py can prove a name obeys the rules; only
looking at the mask can prove the name describes *that* piece of geometry. Roughly 5 % sampling is the
working compromise for large batches, on top of the mandatory screen_large.py pass.

    python finetune/relabel/review_sheet.py --root /data/pv_new --out /data/relabel_new/review --n 30 --seed 42

Without --seed the sample is evenly spaced and therefore reproducible (same objects every run); with a
seed it is random, so a second reviewer can draw a fresh sample instead of re-reading the same objects.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rules import PALETTE, font, open_render, part_thumb


def read(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def mask_image(ids: np.ndarray, n_parts: int):
    """Part-id map in palette colours with the id printed at each part's centroid."""
    h, w = ids.shape
    rgb = np.full((h, w, 3), 245, np.uint8)
    rgb[ids == -1] = (255, 255, 255)
    stats = {}
    for p in range(n_parts):
        m = ids == p
        if not m.any():
            continue
        rgb[m] = PALETTE[p % len(PALETTE)]
        ys, xs = np.nonzero(m)
        stats[p] = {"px": int(m.sum()), "cx": float(xs.mean()), "cy": float(ys.mean())}
    im = Image.fromarray(rgb)
    d = ImageDraw.Draw(im)
    f = font(22, bold=True)
    for p, s in stats.items():
        bbox = d.textbbox((s["cx"], s["cy"]), str(p), font=f, anchor="mm")
        d.rectangle([bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2], fill=(0, 0, 0))
        d.text((s["cx"], s["cy"]), str(p), font=f, fill=(255, 255, 255), anchor="mm")
    return im, stats


def panel(images):
    w = sum(im.width for _, im in images) + 10 * (len(images) - 1)
    h = max(im.height for _, im in images) + 26
    out = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(out)
    f = font(15)
    x = 0
    for title, im in images:
        d.text((x + 4, 4), title, font=f, fill=(40, 40, 40))
        out.paste(im.convert("RGB"), (x, 26))
        x += im.width + 10
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--az", default="az0")
    ap.add_argument("--ids_file", default=None)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "img"), exist_ok=True)
    if args.ids_file:
        with open(args.ids_file, encoding="utf-8") as f:
            cand = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    else:
        cand = sorted(d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d)))
    cand = [o for o in cand if os.path.exists(os.path.join(args.root, o, "views", args.az, "ids.npy"))]
    if not cand:
        print(f"no object under {args.root} has views/{args.az}/ids.npy -- render first")
        return
    n = min(args.n, len(cand))
    if args.seed is None:
        pick = cand[::max(1, len(cand) // n)][:n]
    else:
        pick = random.Random(args.seed).sample(cand, n)

    md = [f"# 部件名审阅（{n} 个样本，root = `{args.root}`）", "",
          f"每个对象：左 = 纹理渲染（{args.az}），右 = GT 部件掩码（数字 = 部件 id）。",
          "表里每行是一个部件的高亮缩略图 + 名字 + caption。`?` = 标注时被标为 uncertain。",
          "掩码里没有数字的部件在该视角不可见，列在表格下方。", "",
          "**看三件事**：名字和掩码指的是同一块吗；名字和整体物体自洽吗（物体是 chair 却出现 blade 就要查）；",
          "同类部件是否同名（三条腿都该叫 leg）。", ""]

    for k, oid in enumerate(pick, 1):
        od = os.path.join(args.root, oid)
        names = read(os.path.join(od, "names.json"))
        meta_path = os.path.join(od, "names_meta.json")
        meta = read(meta_path) if os.path.exists(meta_path) else {}
        unc = meta.get("uncertain") or [False] * len(names)
        old_path = os.path.join(od, "names_v1.json")
        old = read(old_path) if os.path.exists(old_path) else None
        cap = read(os.path.join(od, "captions.json"))
        pids = cap["source_part_ids"]
        ids = np.load(os.path.join(od, "views", args.az, "ids.npy"))
        mask, stats = mask_image(ids, len(names))
        render = open_render(os.path.join(od, "views", args.az, "render.png"))

        imgs = ([(f"render {args.az}", render)] if render is not None else []) + [("GT part mask (id)", mask)]
        img_name = f"{k:03d}_{oid[:8]}.png"
        panel(imgs).save(os.path.join(args.out, "img", img_name))

        total_fg = max(1, int((ids >= 0).sum()))
        head = "| id | 掩码 | 名字 |" + (" 旧名字 |" if old else "") + " 可见占比 | caption（短） |"
        md += [f"## {k}. `{oid[:8]}` — object: **{meta.get('object', '?')}**", "",
               f"![{oid[:8]}](img/{img_name})", "",
               head, "|---|---|---|" + ("---|" if old else "") + "---|---|"]
        hidden = []
        for p, name in enumerate(names):
            pid = pids[p] if p < len(pids) else str(p)
            c = cap["captions"].get(pid) or ["", ""]
            short = re.sub(r"\s+", " ", (c[0] or "").strip())[:110]
            flag = " ?" if p < len(unc) and unc[p] else ""
            oldcol = f" {old[p]} |" if old else ""
            if p in stats:
                thumb = f"{k:03d}_{oid[:8]}_p{p}.png"
                part_thumb(ids, p, render).save(os.path.join(args.out, "img", thumb))
                md.append(f"| {p} | ![p{p}](img/{thumb}) | **{name}**{flag} |{oldcol} "
                          f"{100 * stats[p]['px'] / total_fg:.0f}% | {short} |")
            else:
                hidden.append(f"{p}: **{name}**{flag}")
        if hidden:
            md += ["", f"{args.az} 不可见：" + "；".join(hidden)]
        md.append("")

    path = os.path.join(args.out, "review.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"wrote {path} ({n} objects: {[o[:8] for o in pick]})")


if __name__ == "__main__":
    main()
