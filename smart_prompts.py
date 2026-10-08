"""智能分割模式: a vision-language model (Kimi) turns the bank's candidates into part names.

The concept-bank proposal (auto_prompts.py) is blind to what the object is: on a car it
offered `wing` because SAM3 fired that word on the fenders, on a robot `engine` and
`shield`. Kimi sees the renders, so it can say "this is a car" and keep wheel / door /
window while dropping wing. It is given the candidate words with how much of the
silhouette each covered, and asked for a category, a main-body word and 2-6 part words,
preferring the candidates so that what it picks is something SAM3 has already shown it
can find on this model. Words outside the bank are dropped: the bank is what SAM3's
decoder LoRA was trained against.

Any OpenAI-compatible chat API with image input works. Configuration (environment or the
gitignored repo .env):

    SEGVIGEN_VLM_API_KEY    the key (MOONSHOT_API_KEY is still read as a fallback)
    SEGVIGEN_VLM_BASE_URL   e.g. https://dashscope.aliyuncs.com/compatible-mode/v1 (Qwen)
                            or https://api.moonshot.cn/v1 (Kimi, the default)
    SEGVIGEN_VLM_MODEL      e.g. qwen3.8-max; on Moonshot it may be left unset and a
                            vision-capable model the key can use is picked from /models
    SEGVIGEN_VLM_THINKING   off | on: Qwen (DashScope) thinks for ~100 s and 5k tokens on
                            this prompt unless told not to; off sends enable_thinking=false
                            (default off on DashScope, ignored elsewhere)

No key -> the request is refused up front (serve_api), never silently degraded.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import urllib.request

from PIL import Image

from data_toolkit.front_view import _env_or_dotenv, build_grid
from auto_prompts import GENERIC_WORDS

DEFAULT_BASE_URL = "https://api.moonshot.cn/v1"
KEY_VARS = ("SEGVIGEN_VLM_API_KEY", "MOONSHOT_API_KEY")
# The key decides which models exist: this one returns 404 for `kimi-latest` and lists
# kimi-k2.6 / kimi-k2.7-code instead. Without SEGVIGEN_VLM_MODEL the first vision-capable,
# non-code model the account can see is used.
DEFAULT_MODEL = None
_MODEL_CACHE = {}
MAX_CANDIDATES = 30
MAX_VIEWS = 4
MAX_SHORTLIST = 8


def vlm_api_key():
    for name in KEY_VARS:
        value = _env_or_dotenv(name)
        if value:
            return value
    return None


def vlm_key_available():
    return bool(vlm_api_key())


def vlm_config():
    """(base_url, model or None) as configured; what /health reports for mode=smart."""
    base_url = (_env_or_dotenv("SEGVIGEN_VLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    return base_url, _env_or_dotenv("SEGVIGEN_VLM_MODEL") or None


def provider_params(base_url):
    """Request fields a provider needs beyond the OpenAI shape."""
    if "dashscope" in base_url and (_env_or_dotenv("SEGVIGEN_VLM_THINKING") or "off").lower() != "on":
        return {"enable_thinking": False}
    return {}


def resolve_model(api_key, base_url, timeout=30):
    """A model id this key may use that accepts images; cached per base_url."""
    if base_url in _MODEL_CACHE:
        return _MODEL_CACHE[base_url]
    request = urllib.request.Request(f"{base_url}/models",
                                     headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        rows = json.load(response).get("data", [])
    vision = [row["id"] for row in rows if row.get("supports_image_in")]
    if not vision:
        raise RuntimeError(f"no vision-capable model available to this key: {[r.get('id') for r in rows]}")
    general = [m for m in vision if "code" not in m and "thinking" not in m]
    ranked = sorted(general or vision, reverse=True)      # newest version string first
    _MODEL_CACHE[base_url] = ranked[0]
    return ranked[0]


def build_question(candidates):
    rows = ", ".join(f"{c['concept']} ({c['area']:.0%}, seen in {c['views']} views)"
                     for c in candidates[:MAX_CANDIDATES])
    return (
        "These tiles are renders of ONE 3D object from different directions. "
        "A segmentation model was asked for many part words; these are the ones it found, "
        f"with the share of the silhouette each covered: {rows}. "
        "Task: split this object into its natural parts for a 3D part library. "
        "Answer with ONLY a JSON object of this shape: "
        '{"object": "<what the object is, 1-3 words>", '
        '"main": "<the word for the main body that the remaining surface belongs to>", '
        '"parts": ["<part word>", ...]}. '
        "Rules: 2 to 6 parts, each a common English part noun in lowercase; prefer words "
        "from the list above; drop list words that do not belong to this kind of object "
        "(a car has no wing, a chair has no engine); do not repeat the main word in parts; "
        "do not use left/right/upper/lower; a part should be a visible, physically "
        "separable piece, not a material or a surface pattern."
    )


GUIDE_MIN_SHARE = 0.03   # a colour under this share of the guide's foreground is shading/anti-aliasing


def guide_part_count(path, min_share=GUIDE_MIN_SHARE):
    """How many flat colours the reference segmentation uses (background = transparent or
    near-black). Each colour is one part the caller wants; left/right pairs may share a word."""
    image = Image.open(path).convert("RGBA")
    image.thumbnail((512, 512))
    pixels = list(image.getdata())
    fg = [(r, g, b) for r, g, b, a in pixels if a > 16 and max(r, g, b) > 40]
    if not fg:
        return 0
    bins = {}
    for r, g, b in fg:
        key = (r // 48, g // 48, b // 48)
        bins[key] = bins.get(key, 0) + 1
    return sum(1 for count in bins.values() if count >= min_share * len(fg))


def guide_instruction(colours):
    plural = "" if colours == 1 else "s"
    return (
        f" The LAST tile is a reference segmentation of this same object drawn by the user: "
        f"every flat colour patch is one part they want (roughly {colours} colour{plural}; "
        "a colour may be reused on another part, and a left/right pair may be painted "
        "differently); the background is black. Name exactly the parts the reference "
        "separates: one vocabulary word per kind of part, where a left/right or front/back "
        "pair of the same thing shares one word, and the largest central piece is the main "
        "word. Do not add parts the reference does not colour separately, and do not merge "
        "two separately coloured parts into one word. Also add a key \"separate\": a list of "
        "the part words whose two or more instances the reference paints in DIFFERENT "
        "colours (a red left leg and a green right leg -> \"leg\"), so they are exported as "
        "separate pieces; leave it empty when every instance of a word shares one colour."
    )


def build_shortlist_question(vocabulary, max_parts=MAX_SHORTLIST, guide_colours=None):
    """Ask for the object and its part words BEFORE any segmentation has run.

    Sweeping SAM3 over the whole bank (884 words x 8 views) took 6-12 minutes per job,
    half of the pipeline; the VLM can name the parts from the renders alone in seconds,
    and SAM3 then only has to measure those few words. With `guide_colours` the user's
    reference segmentation is attached as the last tile and sets the parts."""
    words = ", ".join(sorted(set(vocabulary)))
    guide = guide_instruction(guide_colours) if guide_colours else ""
    return (
        "These tiles are renders of ONE 3D object from different directions." + guide + " "
        "Task: split this object into its natural parts for a 3D part library. "
        "Answer with ONLY a JSON object of this shape: "
        '{"object": "<what the object is, 1-3 words>", '
        '"main": "<the word for the main body that the remaining surface belongs to>", '
        '"parts": ["<part word>", ...]}. '
        f"Rules: 1 to {max_parts} parts, most important first; every word (main and parts) "
        "MUST be taken verbatim from this vocabulary, lowercase: " + words + ". "
        "Pick words for visible, physically separable pieces of THIS object (a car has "
        "wheels and doors, a dog has ears and a tail); never a material, a surface pattern "
        "or a shape word; do not use left/right/upper/lower; do not repeat the main word "
        "in parts."
    )


def _last_json_object(text):
    """The last balanced {...} in `text` that parses as JSON.

    A reasoning model thinks out loud before the answer, and the thinking may contain
    braces of its own; the answer is the final object.
    """
    end = len(text)
    while True:
        close = text.rfind("}", 0, end)
        if close < 0:
            return None
        depth, start = 0, None
        for index in range(close, -1, -1):
            if text[index] == "}":
                depth += 1
            elif text[index] == "{":
                depth -= 1
                if depth == 0:
                    start = index
                    break
        if start is not None:
            try:
                data = json.loads(text[start:close + 1])
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass
        end = close
        if end <= 0:
            return None


def parse_reply(text, allowed, generic=()):
    """{"object","main","parts","dropped"} from the VLM reply.

    Keeps only `allowed` (bank) words, and drops `generic` shape words (plank, panel,
    column, ...) that a model reaches for on furniture; the rule-based path never
    offers them either.
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    data = _last_json_object(text)
    if data is None:
        raise ValueError(f"no JSON object in the VLM reply: {text[:200]!r}")
    allowed = set(allowed)
    clean = lambda word: str(word).strip().lower()
    parts, dropped, seen = [], [], set()
    for word in data.get("parts") or []:
        word = clean(word)
        if not word or word in seen:
            continue
        seen.add(word)
        (parts if word in allowed and word not in generic else dropped).append(word)
    main = clean(data.get("main") or "")
    if main and main not in allowed:
        dropped.append(main)
        main = ""
    parts = [p for p in parts if p != main]
    separate = []
    for word in data.get("separate") or []:
        word = clean(word)
        if word in parts and word not in separate:
            separate.append(word)
    return {"object": str(data.get("object") or "").strip(), "main": main or None,
            "parts": parts, "dropped": dropped, "separate": separate}


TOKEN_BUDGETS = (8000, 16000)   # kimi-k3 reasons for ~2-3k tokens before the answer; retry once if cut


def kimi_select(image_paths, candidates, allowed_words, api_key=None, base_url=None,
                model=None, timeout=300):
    """Review SAM3's candidate words: object / main / parts (the full-sweep path)."""
    return ask_vlm(image_paths, build_question(candidates), allowed_words,
                   api_key=api_key, base_url=base_url, model=model, timeout=timeout)


def kimi_shortlist(image_paths, vocabulary, api_key=None, base_url=None, model=None,
                   timeout=300, guide_image=None):
    """Name object / main / parts from the renders alone, restricted to the bank vocabulary.

    `guide_image`: the user's reference segmentation (one flat colour per part); it is
    shown to the model as the last tile and fixes which parts to name."""
    colours = guide_part_count(guide_image) if guide_image else None
    return ask_vlm(image_paths, build_shortlist_question(vocabulary, guide_colours=colours),
                   vocabulary, api_key=api_key, base_url=base_url, model=model,
                   timeout=timeout, guide_image=guide_image)


def ask_vlm(image_paths, question, allowed_words, api_key=None, base_url=None,
            model=None, timeout=300, guide_image=None):
    """Ask the VLM `question` about the renders. Raises on a missing key or an unusable reply.

    The reasoning models answer after thinking out loud; a reply cut off by max_tokens
    (finish_reason "length", empty content) is retried once with double the budget.
    """
    api_key = api_key or vlm_api_key()
    if not api_key:
        raise RuntimeError("智能分割模式 (mode=smart) needs SEGVIGEN_VLM_API_KEY (env var or the repo .env file)")
    default_base, default_model = vlm_config()
    base_url = (base_url or default_base).rstrip("/")
    model = model or default_model
    if not model:
        if "moonshot" not in base_url:
            raise RuntimeError(f"set SEGVIGEN_VLM_MODEL: {base_url} does not advertise image support per model")
        model = resolve_model(api_key, base_url)
    tiles = list(image_paths)[:MAX_VIEWS]
    if guide_image:
        tiles = tiles[:MAX_VIEWS - 1] + [guide_image]
    grid, _ = build_grid(tiles)
    buffer = io.BytesIO()
    grid.save(buffer, format="JPEG", quality=88)
    data_url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
    # These models reject any temperature but 1, and a reasoning model spends tokens before
    # the answer: leave temperature alone and give it room.
    last_error = None
    for budget in TOKEN_BUDGETS:
        payload = {
            "model": model, "max_tokens": budget, **provider_params(base_url),
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": question},
            ]}],
        }
        request = urllib.request.Request(
            f"{base_url}/chat/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.load(response)
        choice = body["choices"][0]
        message = choice["message"]
        reply = message.get("content") or ""
        if choice.get("finish_reason") == "length" or not reply.strip():
            last_error = RuntimeError(f"VLM reply cut off at {budget} tokens before the answer")
            continue
        try:
            result = parse_reply(reply, allowed_words, GENERIC_WORDS)
        except ValueError as error:
            last_error = error
            continue
        result["model"] = model
        result["raw"] = reply
        result["tokens"] = body.get("usage", {}).get("completion_tokens")
        return result
    raise last_error
