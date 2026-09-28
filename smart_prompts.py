"""智能分割模式: a vision-language model (Kimi) turns the bank's candidates into part names.

The concept-bank proposal (auto_prompts.py) is blind to what the object is: on a car it
offered `wing` because SAM3 fired that word on the fenders, on a robot `engine` and
`shield`. Kimi sees the renders, so it can say "this is a car" and keep wheel / door /
window while dropping wing. It is given the candidate words with how much of the
silhouette each covered, and asked for a category, a main-body word and 2-6 part words,
preferring the candidates so that what it picks is something SAM3 has already shown it
can find on this model. Words outside the bank are dropped: the bank is what SAM3's
decoder LoRA was trained against.

Configuration is the same as the front-view VLM: MOONSHOT_API_KEY (env or repo .env),
SEGVIGEN_VLM_BASE_URL, SEGVIGEN_VLM_MODEL. No key -> the request is refused up front
(serve_api), never silently degraded to the heuristic.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import urllib.request

from data_toolkit.front_view import _env_or_dotenv, build_grid

DEFAULT_BASE_URL = "https://api.moonshot.cn/v1"
DEFAULT_MODEL = "kimi-latest"
MAX_CANDIDATES = 30
MAX_VIEWS = 4


def vlm_key_available():
    return bool(_env_or_dotenv("MOONSHOT_API_KEY"))


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


def parse_reply(text, allowed):
    """{"object","main","parts","dropped"} from Kimi's reply, keeping only allowed words."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError(f"no JSON object in the VLM reply: {text[:200]!r}")
    data = json.loads(match.group(0))
    allowed = set(allowed)
    clean = lambda word: str(word).strip().lower()
    parts, dropped, seen = [], [], set()
    for word in data.get("parts") or []:
        word = clean(word)
        if not word or word in seen:
            continue
        seen.add(word)
        (parts if word in allowed else dropped).append(word)
    main = clean(data.get("main") or "")
    if main and main not in allowed:
        dropped.append(main)
        main = ""
    parts = [p for p in parts if p != main]
    return {"object": str(data.get("object") or "").strip(), "main": main or None,
            "parts": parts, "dropped": dropped}


def kimi_select(image_paths, candidates, allowed_words, api_key=None, base_url=None,
                model=None, timeout=180):
    """Ask Kimi for object / main / parts. Raises on a missing key or an unusable reply."""
    api_key = api_key or _env_or_dotenv("MOONSHOT_API_KEY")
    if not api_key:
        raise RuntimeError("智能分割模式 (mode=smart) needs MOONSHOT_API_KEY (env var or the repo .env file)")
    base_url = (base_url or _env_or_dotenv("SEGVIGEN_VLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    model = model or _env_or_dotenv("SEGVIGEN_VLM_MODEL") or DEFAULT_MODEL
    grid, _ = build_grid(list(image_paths)[:MAX_VIEWS])
    buffer = io.BytesIO()
    grid.save(buffer, format="JPEG", quality=88)
    data_url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
    payload = {
        "model": model, "temperature": 0,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": build_question(candidates)},
        ]}],
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.load(response)
    reply = body["choices"][0]["message"]["content"]
    result = parse_reply(reply, allowed_words)
    result["model"] = model
    result["raw"] = reply
    return result
