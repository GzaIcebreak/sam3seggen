"""Pick a "front" (or best-conditioned) view for the 2D-guided segmentation pipeline.

Three strategies, cheapest first:

- metric: score candidate renders by silhouette statistics -- symmetry around the
  foreground centroid's vertical axis, foreground coverage, and centeredness.
  Fully offline and deterministic.
- auto: take the metric top-K, run SAM3 on each, and pick the view whose legend has
  the best mean prompt score. This answers "which view SEGMENTS best", which the
  silhouette metrics can only approximate.
- vlm: take the metric top-N, lay them out as a numbered grid, and ask a
  vision-language model (Kimi/Moonshot by default) which one is the canonical
  front. This answers "which view IS the front", a semantic question the metrics
  cannot encode. Needs network + an API key.
"""
import base64
import io
import json
import math
import os
import re
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FRONT_VIEW_MODES = ("metric", "auto", "vlm")
# Full turntable at 45-degree steps, rendered with the conditioning camera itself
# (render_cond_view), so the winning azimuth feeds straight back into the pipeline.
DEFAULT_CANDIDATE_AZIMUTHS = (0, 45, 90, 135, 180, 225, 270, 315)

# Composite metric weights: symmetry is the strongest "front" signal for man-made
# objects, coverage rejects edge-on/grazing views, centeredness is a weak prior.
W_SYMMETRY = 0.45
W_COVERAGE = 0.35
W_CENTERED = 0.20
# Coverage saturates at this foreground fraction and is penalised past `clip` (the
# object starts leaving the frame).
COVERAGE_TARGET = 0.30
COVERAGE_CLIP = 0.90


def silhouette_metrics(image_path, alpha_threshold=16):
    """Foreground statistics of one render, from its alpha channel."""
    alpha = np.asarray(Image.open(image_path).convert("RGBA"))[..., 3]
    fg = alpha > alpha_threshold
    height, width = fg.shape
    count = int(fg.sum())
    if count == 0:
        return {"fg_ratio": 0.0, "centeredness": 0.0, "symmetry": 0.0, "score": 0.0}

    fg_ratio = count / fg.size
    ys, xs = np.nonzero(fg)
    # Normalised centroid offset in [-1, 1] per axis; 1 means dead centre.
    dx = xs.mean() / width * 2 - 1
    dy = ys.mean() / height * 2 - 1
    centeredness = 1.0 - min(1.0, math.hypot(dx, dy) / math.sqrt(2.0))
    symmetry = _mirror_iou(fg, float(xs.mean()))

    coverage = min(fg_ratio / COVERAGE_TARGET, 1.0)
    if fg_ratio > COVERAGE_CLIP:
        coverage *= 0.5
    score = W_SYMMETRY * symmetry + W_COVERAGE * coverage + W_CENTERED * centeredness
    return {
        "fg_ratio": fg_ratio,
        "centeredness": centeredness,
        "symmetry": symmetry,
        "score": score,
    }


def _mirror_iou(fg, center_col):
    """IoU between the mask and its mirror image around a vertical axis."""
    height, width = fg.shape
    flipped = np.fliplr(fg)
    shift = int(round(2 * center_col - (width - 1)))
    mirrored = np.roll(flipped, shift, axis=1)
    if shift > 0:
        mirrored[:, :shift] = False
    elif shift < 0:
        mirrored[:, shift:] = False
    union = int((fg | mirrored).sum())
    return float((fg & mirrored).sum()) / union if union else 0.0


def rank_views(image_paths, alpha_threshold=16):
    """Return [(path, metrics)] sorted best-first by composite silhouette score."""
    scored = [(path, silhouette_metrics(path, alpha_threshold)) for path in image_paths]
    scored.sort(key=lambda item: item[1]["score"], reverse=True)
    return scored


def sam3_legend_score(legend, expected_names):
    """Mean per-prompt SAM3 score of a 2D-map legend; a missing prompt scores 0.

    Rewards views where every prompt was detected confidently, over views where a
    prompt was missed entirely but the rest scored high.
    """
    by_part = {entry.get("part"): entry.get("score") or 0.0 for entry in legend}
    if not expected_names:
        return 0.0
    return sum(by_part.get(name, 0.0) for name in expected_names) / len(expected_names)


def build_grid(image_paths, cell=384):
    """Tile renders into a numbered grid for the VLM; returns (PIL image, count)."""
    count = len(image_paths)
    cols = math.ceil(math.sqrt(count))
    rows = math.ceil(count / cols)
    grid = Image.new("RGB", (cols * cell, rows * cell), (24, 24, 24))
    try:
        font = ImageFont.truetype("arial.ttf", cell // 6)
    except OSError:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(grid)
    for index, path in enumerate(image_paths):
        image = Image.open(path).convert("RGBA")
        # Composite onto dark grey so transparent backgrounds stay readable.
        tile = Image.new("RGBA", image.size, (24, 24, 24, 255))
        tile.alpha_composite(image)
        tile = tile.convert("RGB").resize((cell, cell))
        x, y = (index % cols) * cell, (index // cols) * cell
        grid.paste(tile, (x, y))
        label = str(index + 1)
        draw.rectangle([x + 6, y + 6, x + cell // 4, y + cell // 4], fill=(255, 255, 255))
        draw.text((x + 14, y + 10), label, fill=(200, 0, 0), font=font)
    return grid, count


def parse_vlm_choice(text, count):
    """Extract the chosen 0-based view index from a VLM reply; None if unusable."""
    for token in re.findall(r"\d+", text or ""):
        value = int(token)
        if 1 <= value <= count:
            return value - 1
    return None


def _env_or_dotenv(name):
    """Read `name` from the environment, falling back to the repo-root .env file.

    The .env file is gitignored local configuration (KEY=VALUE per line, '#' comments);
    it exists so the VLM key does not have to be exported in every shell.
    """
    value = os.environ.get(name)
    if value:
        return value
    dotenv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.isfile(dotenv):
        with open(dotenv, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    if key.strip() == name:
                        return val.strip()
    return None


def vlm_pick(image_paths, object_hint=None, api_key=None, base_url=None, model=None,
             timeout=120):
    """Ask a vision-language model which of `image_paths` is the canonical front view.

    OpenAI-compatible chat API; defaults target Kimi/Moonshot. Configuration:
    MOONSHOT_API_KEY (required; env var or the repo-root .env file),
    SEGVIGEN_VLM_BASE_URL, SEGVIGEN_VLM_MODEL.
    Returns the 0-based index into image_paths.
    """
    api_key = api_key or _env_or_dotenv("MOONSHOT_API_KEY")
    if not api_key:
        raise RuntimeError("vlm front-view mode needs MOONSHOT_API_KEY (env var or .env file)")
    base_url = (base_url or _env_or_dotenv("SEGVIGEN_VLM_BASE_URL")
                or "https://api.moonshot.cn/v1").rstrip("/")
    model = model or _env_or_dotenv("SEGVIGEN_VLM_MODEL") or "kimi-latest"

    grid, count = build_grid(image_paths)
    buffer = io.BytesIO()
    grid.save(buffer, format="JPEG", quality=90)
    data_url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()

    hint = f" of {object_hint}" if object_hint else ""
    question = (
        f"These are {count} renders{hint} of the SAME 3D object, photographed from "
        f"different directions and numbered 1-{count} (red label, top-left of each tile). "
        "Which number shows the most canonical FRONT view of the object -- the angle a "
        "product photo or a person facing it would use? Reply with ONLY the number."
    )
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": question},
            ],
        }],
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.load(response)
    reply = body["choices"][0]["message"]["content"]
    choice = parse_vlm_choice(reply, count)
    if choice is None:
        raise RuntimeError(f"VLM did not return a usable view number: {reply!r}")
    return choice
