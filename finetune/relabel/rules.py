"""Shared naming rules and drawing helpers for the relabel toolkit.

The word lists live here and nowhere else: the ad-hoc scripts this replaces carried three
independent copies of the blacklist, which drifted. Every checker imports from this module so a
rule change takes effect everywhere at once.
"""
from __future__ import annotations

import os
import re

# Words that must never appear in a part name. Two kinds: placeholders that say nothing about
# identity ("component", "section") and appearance words that describe how a part looks rather
# than what it is ("cylindrical", colours). "cylindrical component" is the canonical failure.
JUNK = {
    "color", "colour", "component", "components", "part", "parts", "piece", "section", "element",
    "object", "model", "3d", "representation", "view", "close-up", "image", "render", "low-poly",
    "detailed", "highlighted", "shape", "texture", "gradient", "design", "geometric",
    "cylindrical", "rectangular", "spherical", "circular",
    "red", "blue", "green", "teal", "black", "white", "gray", "grey", "brown", "yellow",
    "orange", "purple", "pink", "gold", "silver",
}

# Acceptable names for a part that covers most of the object. A part with a large visible share
# and a *local* name (e.g. a whole snowman labelled "head") is the single most damaging error
# class for SAM3, so screen_large.py flags everything outside this set for human review.
BODY_LIKE = {
    "body", "main body", "upper body", "lower body", "base", "base plate", "stand", "frame",
    "housing", "shell", "upper shell", "lower shell", "case", "casing", "torso", "hull",
    "chassis", "column", "pillar", "pedestal", "container", "tank", "pot", "vase", "bottle",
    "bowl", "cup", "mug", "jar", "box", "cabinet", "tower", "wall", "block", "tabletop",
    "skirt", "door", "roof", "canopy", "platform", "deck", "panel", "board", "slab",
}

MAX_WORDS = 3


def words(name: str) -> list[str]:
    return re.findall(r"[a-z0-9\-']+", name.lower())


def check_name(name: object) -> list[str]:
    """Every rule violation in one name, as human-readable strings. Empty list = clean.

    A junk-word hit is reported but is not automatically an error: `orange` is blacklisted as a
    colour yet three parts in the existing set really are oranges. Always re-read the caption
    before changing a flagged name.
    """
    if not isinstance(name, str) or not name.strip():
        return ["empty name"]
    problems = []
    if name != name.lower():
        problems.append(f"not lowercase: {name!r}")
    if name != name.strip() or name.endswith("."):
        problems.append(f"stray whitespace or trailing period: {name!r}")
    w = words(name)
    if len(w) > MAX_WORDS:
        problems.append(f">{MAX_WORDS} words: {name!r}")
    hits = [x for x in w if x in JUNK]
    if hits:
        problems.append(f"junk/appearance word {hits} in {name!r}")
    return problems


def normalise(name: str) -> str:
    return name.strip().lower().rstrip(".")


# --- drawing -------------------------------------------------------------------------------
# 20 well-separated colours; part id -> PALETTE[id % 20].
PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
    (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
    (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195), (128, 128, 0), (255, 215, 180),
    (0, 0, 128), (128, 128, 128),
]

_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]
_FONT_BOLD_CANDIDATES = [
    r"C:\Windows\Fonts\arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]


def font(size: int, bold: bool = False):
    """A TrueType font on Windows, Linux or macOS; PIL's bitmap default if none is installed."""
    from PIL import ImageFont
    for p in (_FONT_BOLD_CANDIDATES if bold else _FONT_CANDIDATES):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def open_render(path: str):
    """render.png (RGBA) composited onto white, or None when the object has no render."""
    from PIL import Image
    if not os.path.exists(path):
        return None
    rgba = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, rgba).convert("RGB")


def part_thumb(ids, part: int, render, size: int = 200):
    """One part highlighted: the rest of the render dimmed, cropped to the object's bbox.

    Tiny parts would vanish at thumbnail size, so the part boundary is drawn in its palette
    colour on top. Without a render, a grey silhouette is used instead.
    """
    import numpy as np
    from PIL import Image

    m = ids == part
    fg = ids >= 0
    if render is not None:
        base = np.asarray(render, dtype=np.float32)
        out = np.where(m[..., None], base, base * 0.25 + 255 * 0.75)
    else:
        out = np.full((*ids.shape, 3), 255, np.float32)
        out[fg] = (215, 215, 215)
        out[m] = PALETTE[part % len(PALETTE)]
    edge = m ^ np.roll(m, 1, 0) | m ^ np.roll(m, 1, 1)
    out = out.astype(np.uint8)
    out[edge & (fg | m)] = PALETTE[part % len(PALETTE)]
    ys, xs = np.nonzero(fg if fg.any() else m)
    if len(ys) == 0:
        return Image.new("RGB", (size, size), (255, 255, 255))
    im = Image.fromarray(out[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
    im.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return canvas
