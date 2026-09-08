"""Mandatory screen: a part covering most of the object but carrying a *local* name.

This is the error class that hurt SAM3 most in the first 2000-object round. When a caption describes
the whole object as if it were one part, the labeller tends to answer with a local noun -- a whole
snowman labelled `head`, a whole chair labelled `seat`. SAM3 then binds that name to the wrong region
and everything downstream inherits it. 139 such errors were found and fixed this way.

Flags every part whose visible share >= --min_frac and whose name is not in rules.BODY_LIKE, renders
one highlighted thumbnail each and tiles them into contact sheets. A flag is not a verdict: some are
genuinely correct local names that simply dominate the front view (sunglasses `lens`, sword `blade`).
Open the sheet and check whether the highlighted region really covers the whole object.

    python finetune/relabel/screen_large.py --root /data/pv_new --out /data/relabel_new/review/large
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rules import BODY_LIKE, font, open_render, part_thumb

CELL = 170
COLS = 6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--names", default=None,
                    help="names_v2.json to screen; default reads each object's names.json")
    ap.add_argument("--min_frac", type=float, default=0.45)
    ap.add_argument("--az", default="az0")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.names:
        with open(args.names, encoding="utf-8") as f:
            recs = {o: (r["names"], r.get("object", "?")) for o, r in json.load(f).items()}
    else:
        recs = {}
        for o in sorted(os.listdir(args.root)):
            np_ = os.path.join(args.root, o, "names.json")
            if not os.path.exists(np_):
                continue
            with open(np_, encoding="utf-8") as f:
                names = json.load(f)
            mp = os.path.join(args.root, o, "names_meta.json")
            obj = "?"
            if os.path.exists(mp):
                with open(mp, encoding="utf-8") as f:
                    obj = json.load(f).get("object", "?")
            recs[o] = (names, obj)

    flagged, no_view = [], 0
    for oid, (names, obj) in recs.items():
        ip = os.path.join(args.root, oid, "views", args.az, "ids.npy")
        if not os.path.exists(ip):
            no_view += 1
            continue
        ids = np.load(ip)
        fg = max(1, int((ids >= 0).sum()))
        for p, name in enumerate(names):
            share = int((ids == p).sum()) / fg
            if share >= args.min_frac and name not in BODY_LIKE:
                flagged.append((oid, p, share, name, obj))

    flagged.sort(key=lambda x: -x[2])
    print(f"screened {len(recs) - no_view} objects ({no_view} without views/{args.az}/ids.npy)")
    print(f"flagged parts: {len(flagged)}")

    f13 = font(13)
    cells = []
    for oid, p, share, name, obj in flagged:
        ids = np.load(os.path.join(args.root, oid, "views", args.az, "ids.npy"))
        render = open_render(os.path.join(args.root, oid, "views", args.az, "render.png"))
        cell = Image.new("RGB", (CELL, CELL), (255, 255, 255))
        cell.paste(part_thumb(ids, p, render, size=CELL - 20), (10, 0))
        ImageDraw.Draw(cell).text((4, CELL - 18), f"{oid[:8]}:p{p} {share:.2f} {name} | {obj}",
                                  font=f13, fill=(0, 0, 0))
        cells.append(cell)

    per_sheet = COLS * COLS
    for s in range(0, len(cells), per_sheet):
        chunk = cells[s:s + per_sheet]
        rows = (len(chunk) + COLS - 1) // COLS
        sheet = Image.new("RGB", (COLS * CELL, rows * CELL), (230, 230, 230))
        for i, c in enumerate(chunk):
            sheet.paste(c, ((i % COLS) * CELL, (i // COLS) * CELL))
        path = os.path.join(args.out, f"sheet_{s // per_sheet:03d}.png")
        sheet.save(path)
        print("wrote", path)

    with open(os.path.join(args.out, "flagged.json"), "w", encoding="utf-8") as f:
        json.dump([{"object_id": o, "part": p, "share": round(s, 3), "name": n, "object": b}
                   for o, p, s, n, b in flagged], f, ensure_ascii=False, indent=1)
    print("wrote", os.path.join(args.out, "flagged.json"))


if __name__ == "__main__":
    main()
