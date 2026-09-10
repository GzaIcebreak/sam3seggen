"""Replace only the ease+v6 column (rightmost 300px) of the 20-col compare sheets.

Does not touch any other column. Verifies the kept region is byte-identical to the backup.
"""
from __future__ import annotations

import os
import sys

from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ext_bench import ASSETS, OUT, front_json, wd  # noqa: E402

S = 300
ASSETS_DIR = os.path.join(ROOT, "assets", "ext_bench")
BACKUP_DIR = OUT
COL = "lift_ease_v6"


def tile_path(key: str, side: str) -> str:
    az = float(front_json(key)["azimuth"])
    if side == "back":
        az = (az + 180) % 360
    return os.path.join(wd(key), "vis", f"{COL}_{az:g}.png")


def paste_rgba(tile_png: str) -> Image.Image:
    im = Image.open(tile_png).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, im).convert("RGB").resize((S, S))


def patch_side(side: str) -> None:
    src = os.path.join(BACKUP_DIR, f"compare_20col_{side}.png")
    if not os.path.exists(src):
        src = os.path.join(ASSETS_DIR, f"compare_{side}.png")
    dst = os.path.join(ASSETS_DIR, f"compare_{side}.png")
    base = Image.open(src).convert("RGB")
    keep_w = base.width - S
    if keep_w != 19 * S:
        raise SystemExit(f"{src}: width {base.width}, expected 20*{S}={20 * S}")
    keys = list(ASSETS)
    if base.height != len(keys) * (S + 22) + 22:
        raise SystemExit(f"{src}: height {base.height}, expected {len(keys) * (S + 22) + 22}")
    canvas = base.copy()
    missing = []
    for r, key in enumerate(keys):
        y = 22 + r * (S + 22) + 22
        png = tile_path(key, side)
        if not os.path.exists(png):
            missing.append(png)
            continue
        canvas.paste(paste_rgba(png), (keep_w, y))
    if missing:
        raise SystemExit("missing tiles:\n  " + "\n  ".join(missing))
    old = list(base.crop((0, 0, keep_w, base.height)).getdata())
    new = list(canvas.crop((0, 0, keep_w, canvas.height)).getdata())
    if old != new:
        raise SystemExit(f"{side}: kept region changed — aborting save")
    canvas.save(dst, optimize=True)
    print(f"patched {dst} ({canvas.width}x{canvas.height}), kept 0:{keep_w} identical")


def main():
    for side in ("front", "back"):
        patch_side(side)


if __name__ == "__main__":
    main()
