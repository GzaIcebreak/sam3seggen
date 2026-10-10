"""Propose part names from the concept bank when the caller gave no prompts.

Without prompts the pipeline used to name everything 主体 / 底座, which on the ten test
models scored 0 or 1 out of 4 every time. The concept bank knows 280 part words, and
SAM3 can be asked all of them over a view grid in about a minute. What comes back is a
pile of overlapping candidates -- body, leg, legs, boot, pants, hat, helmet -- and the
job here is to pick the few that make a partition:

  * a *whole-object* word (covers half the silhouette or more) names the remainder,
    not a part; `body` wins that role when SAM3 saw it;
  * parts are chosen coarse-first and must be nearly disjoint; a later, larger word
    that contains an earlier pick replaces it (head over ear, leg over boot);
  * words that describe a shape rather than a part (housing, panel, column, ...) and
    plurals of a word already present (legs next to leg) are not eligible;
  * a word has to be seen in enough views and cover enough area to be worth a prompt.

The result is an ordinary prompt list plus an unassigned_to, so everything downstream
(vote, refine, export, repair) is unchanged.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))

# Shape and material words: SAM3 fires them on almost anything, and none is a part name a
# person would choose. Kept short on purpose; the containment and area rules do most work.
GENERIC_WORDS = frozenset({
    "housing", "cover", "column", "platform", "stand", "shell", "plate", "panel",
    "display", "block", "base block", "bun", "antenna", "surface", "object", "part",
    "piece", "structure", "material", "texture", "pattern", "decoration", "detail",
    "edge", "rim", "side", "side panel", "body panel", "plank", "board",
})
DEFAULT_MAX_PARTS = 6
DEFAULT_MIN_AREA = 0.02       # share of the silhouette, summed over views
DEFAULT_WHOLE = 0.5           # at least this share: a name for the object, not a part
DEFAULT_WHOLE_VLM = 0.6       # for a word the VLM already vouches for as a part
DEFAULT_SHORTLIST_MIN_VIEWS = 0.25   # a shortlisted word only has to show up in a quarter of the views
DEFAULT_SHORTLIST_IOU = 0.6          # two shortlisted words whose masks overlap this much are one part
DEFAULT_SHORTLIST_INSIDE = 0.8       # a word this far inside another kept word's mask is its sub-part
DEFAULT_SHORTLIST_MIN_PARTS = 2      # fewer found parts than this: the full sweep gives a second opinion
VARIANT_MAX_GROWTH = 2.0             # a variant may cover at most this x the own word's area
MAX_VARIANT_PHRASES = 48             # SAM3 phrases measured for one shortlist (parts x variants)
VLM_ATTEMPTS = 3                     # tries per VLM call before 智能分割模式 gives up
VLM_RETRY_WAIT = 10.0                # seconds before the 2nd try; x2 before the 3rd


class VlmUnavailable(RuntimeError):
    """智能分割模式 got no usable answer from the VLM. The job fails rather than falling
    back to the rule-based pick: when the proxy tunnel dropped, 28 of 36 smart-mode jobs
    quietly did that and named a grenade `lock` + `machine body`."""


def ask_vlm_or_fail(call, what, attempts=VLM_ATTEMPTS, wait=VLM_RETRY_WAIT):
    """call() until it answers, up to `attempts` times; then raise VlmUnavailable."""
    import time

    last = None
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as error:
            last = error
            print(f"[smart] VLM {what}: attempt {attempt}/{attempts} failed "
                  f"({type(error).__name__}: {str(error)[:160]})")
            if attempt < attempts and wait > 0:
                time.sleep(wait * attempt)
    raise VlmUnavailable(
        f"智能分割模式：大模型调用失败（{what}，已重试 {attempts} 次），任务中止，没有退回规则挑词。"
        f"请检查 SEGVIGEN_VLM_BASE_URL / 网络后重试。最后的错误：{type(last).__name__}: {str(last)[:300]}"
    ) from last


IRREGULAR_SINGULAR = {"feet": "foot", "teeth": "tooth", "antennae": "antenna", "geese": "goose",
                      "men": "man", "women": "woman", "children": "child", "knives": "knife",
                      "leaves": "leaf", "shelves": "shelf", "halves": "half", "wolves": "wolf",
                      "hooves": "hoof", "glasses": "glasses", "pants": "pants", "shorts": "shorts",
                      "jeans": "jeans", "scissors": "scissors", "tongs": "tongs", "pliers": "pliers"}
OBJECT_SUFFIXES = ("figurine", "figure", "toy", "model", "statue", "character", "sculpture")
DEFAULT_OVERLAP = 0.3         # parts sharing more than this of the smaller one are the same thing
DEFAULT_MIN_VIEWS = 0.5       # seen in at least this share of the views
PREFERRED_MAIN = ("body", "torso", "主体")
AUTO_AZIMUTHS = "0,45,90,135,180,225,270,315"
AUTO_ELEVATIONS = "15"


def _plural_of_present(name, names):
    return name.endswith("s") and name[:-1] in names


def propose_from_masks(masks, foreground, scores, concepts, max_parts=DEFAULT_MAX_PARTS,
                       min_area=DEFAULT_MIN_AREA, whole=DEFAULT_WHOLE,
                       overlap=DEFAULT_OVERLAP, min_views=DEFAULT_MIN_VIEWS):
    """Pick a main-body name and up to `max_parts` disjoint part names.

    masks: bool [views, concepts, H, W] (raw, overlapping); foreground: bool [views, H, W];
    scores: [views, concepts], 0 where a concept was not detected in that view.
    Returns {"main": str | None, "parts": [str], "candidates": [...]}.
    """
    masks = np.asarray(masks, dtype=bool)
    foreground = np.asarray(foreground, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    views, count = scores.shape
    masks = masks & foreground[:, None]
    fg_total = max(int(foreground.sum()), 1)
    area = masks.sum(axis=(2, 3)).sum(axis=0) / fg_total
    blob = largest_blob_share(masks, foreground)
    detected = (scores > 0).sum(axis=0)
    mean_score = np.where(detected > 0, scores.sum(axis=0) / np.maximum(detected, 1), 0.0)
    names = list(concepts)
    present = {names[i] for i in range(count) if detected[i] > 0}
    eligible = np.array([
        detected[i] >= min_views * views and names[i] not in GENERIC_WORDS
        and not _plural_of_present(names[i], present)
        for i in range(count)
    ])

    whole_mask = eligible & (area >= whole)
    main = None
    if whole_mask.any():
        preferred = [i for i in np.flatnonzero(whole_mask) if names[i] in PREFERRED_MAIN]
        main = preferred[0] if preferred else int(np.argmax(np.where(whole_mask, area * mean_score, -1)))

    def share(a, b):
        return (a & b).sum() / max(min(a.sum(), b.sum()), 1)

    # a word whose single biggest blob is under `whole` can be a part even when all its
    # blobs together are not: six shelves sum to most of a rack, one shelf does not
    candidates = [i for i in np.flatnonzero(eligible & (area >= min_area) & (blob < whole))]
    candidates.sort(key=lambda i: -(area[i] * mean_score[i]))
    picked = []
    for i in candidates:
        m = masks[:, i]
        # a larger word that swallows an earlier, smaller pick takes its place (head > ear)
        swallowed = [j for j in picked if (m & masks[:, j]).sum() / max(masks[:, j].sum(), 1) > 0.7
                     and area[i] > area[j]]
        others = [j for j in picked if j not in swallowed]
        if any(share(m, masks[:, j]) > overlap for j in others):
            continue
        picked = others + [i]
        if len(picked) > max_parts:
            picked = sorted(picked, key=lambda j: -(area[j] * mean_score[j]))[:max_parts]
    picked.sort(key=lambda j: -area[j])
    covered = np.zeros_like(foreground)
    for j in picked:
        covered |= masks[:, j]
    rows = [{"concept": names[i], "area": float(area[i]), "blob": float(blob[i]),
             "score": float(mean_score[i]), "views": int(detected[i]),
             "eligible": bool(eligible[i])}
            for i in np.argsort(-(area * mean_score))[:20] if detected[i] > 0]
    return {
        "main": names[main] if main is not None else None,
        "parts": [names[j] for j in picked],
        "covered": float(covered.sum() / fg_total),
        "candidates": rows,
    }


def largest_blob_share(masks, foreground):
    """Per concept: the biggest connected blob's share of the silhouette, averaged over the
    views where the concept fired. The summed mask cannot tell six shelves from one rack."""
    from scipy import ndimage

    views, count = masks.shape[:2]
    out = np.zeros(count)
    for i in range(count):
        shares = []
        for v in range(views):
            mask = masks[v, i]
            if not mask.any():
                continue
            labels, n = ndimage.label(mask)
            sizes = np.bincount(labels.ravel())[1:] if n else np.array([0])
            shares.append(sizes.max() / max(int(foreground[v].sum()), 1))
        out[i] = float(np.mean(shares)) if shares else 0.0
    return out


def word_stats(masks, foreground, scores, concepts, min_area=DEFAULT_MIN_AREA,
               min_views=DEFAULT_MIN_VIEWS):
    """Candidate-style rows for the given concepts: area, biggest blob, score, views,
    and whether SAM3 found the word often enough and large enough to prompt with."""
    masks = np.asarray(masks, dtype=bool)
    foreground = np.asarray(foreground, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    views = scores.shape[0]
    masks = masks & foreground[:, None]
    fg_total = max(int(foreground.sum()), 1)
    area = masks.sum(axis=(2, 3)).sum(axis=0) / fg_total
    blob = largest_blob_share(masks, foreground)
    detected = (scores > 0).sum(axis=0)
    mean_score = np.where(detected > 0, scores.sum(axis=0) / np.maximum(detected, 1), 0.0)
    return [{"concept": str(name), "area": float(area[i]), "blob": float(blob[i]),
             "score": float(mean_score[i]), "views": int(detected[i]),
             "eligible": bool(detected[i] >= min_views * views and area[i] >= min_area)}
            for i, name in enumerate(concepts)]


def accept_shortlist(chosen, rows, max_parts=DEFAULT_MAX_PARTS, whole=DEFAULT_WHOLE_VLM,
                     masks=None, foreground=None, concepts=None, iou=DEFAULT_SHORTLIST_IOU):
    """(proposal, reason). The VLM's shortlist after SAM3 has measured it and its variants:
    a word is found if any of its phrases is (the best one segments it: `arm` when the
    VLM said `arms`, `dog head` for a head SAM3 only saw from behind); words SAM3 did not
    find at all are dropped, words whose biggest blob covers `whole` of the object are
    whole-object words, words whose mask is another kept word's (IoU >= `iou`, needs the
    masks) are collapsed into it, and what is left must be a main word plus at least one
    part. proposal["parts"] holds prompt specs (`name` or `name=phrase`)."""
    by_name = {row["concept"]: row for row in rows}
    picked = pick_variants(chosen, rows, whole)
    found = [p for p in chosen["parts"] if picked.get(p)]
    missing = [p for p in chosen["parts"] if p not in found]
    # a word whose own mask names the whole object but has no usable variant
    whole_words = [p for p in missing
                   if float(by_name.get(p, {}).get("blob", by_name.get(p, {}).get("area", 0))) >= whole]
    missing = [p for p in missing if p not in whole_words]
    overlaps = []
    if masks is not None and foreground is not None and concepts is not None and len(found) > 1:
        found, overlaps = resolve_overlaps(found, picked, masks, foreground, concepts, iou)
    main = chosen.get("main")
    if not main:
        return None, "the VLM named no main-body word"
    if not found:
        return None, (f"none of the VLM's parts survived: not found by SAM3 {missing}, "
                      f"whole-object words {whole_words}")
    kept = found[:max_parts]
    proposal = {"main": main, "parts": [part_spec(p, picked.get(p)) for p in kept],
                "part_names": kept, "phrases": {p: picked.get(p) or p for p in kept + [main]},
                "candidates": rows, "mode": "smart", "shortlist": True, "kimi": dict(chosen)}
    renamed = {p: picked[p] for p in kept if picked.get(p) and picked[p] != p}
    if renamed:
        proposal["kimi"]["variants_used"] = renamed
    if missing:
        proposal["kimi"]["not_found"] = missing
    if whole_words:
        proposal["kimi"]["dropped_whole"] = whole_words
    if overlaps:
        proposal["kimi"]["dropped_overlap"] = {word: same for word, same in overlaps}
    return proposal, None



def covered_share(masks, foreground, concepts, parts):
    masks = np.asarray(masks, dtype=bool) & np.asarray(foreground, dtype=bool)[:, None]
    covered = np.zeros(masks.shape[0:1] + masks.shape[2:], dtype=bool)
    names = list(concepts)
    for part in parts:
        if part in names:
            covered |= masks[:, names.index(part)]
    return float(covered.sum() / max(int(np.asarray(foreground).sum()), 1))


def sam3_sweep(py_sam3, sam3_model, concept_bank, prompt_dir, views_dir, painted,
               masks_npz, words):
    subprocess.run([
        py_sam3, os.path.join(ROOT, "sam3_multiview.py"),
        "--views_dir", prompt_dir, "--out", masks_npz, "--raw",
        "--model", sam3_model, "--concept_bank", concept_bank,
        *(["--extra_views_dir", views_dir] if painted else []),
        "--prompts", *words,
    ], check=True, stdout=subprocess.DEVNULL)


def bank_concepts(py_sam3, bank_path, cache_dir):
    """The bank's concept names, read once through the SAM3 venv (torch lives there)."""
    # Keyed by the bank file's mtime: a retrained bank dropped in at the same path must not
    # keep proposing from the old word list.
    stamp = int(os.path.getmtime(bank_path))
    cache = os.path.join(cache_dir, f"concept_bank_names.{stamp}.txt")
    if not os.path.isfile(cache):
        code = ("import sys, torch; b = torch.load(sys.argv[1], map_location='cpu', "
                "weights_only=False); print('\\n'.join(b['names']))")
        text = subprocess.run([py_sam3, "-c", code, bank_path], check=True,
                              capture_output=True, text=True).stdout
        with open(cache, "w", encoding="utf-8") as handle:
            handle.write(text)
    with open(cache, encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def drop_whole_words(parts, candidates, whole=DEFAULT_WHOLE_VLM):
    """(kept, dropped): a 'part' whose single biggest blob covers `whole` of the silhouette
    names the object, not a part of it -- voted as a part it swallows the remainder (a
    sword's 'crossbar' at 83% left nothing for the blade). Judged per blob so that a rack's
    'shelf' (six blobs, 60% together, ~10% each) stays a part."""
    area = {row["concept"]: float(row.get("blob", row["area"])) for row in candidates}
    dropped = [name for name in parts if area.get(name, 0.0) >= whole]
    return [name for name in parts if name not in dropped], dropped


def singular(phrase):
    """A naive English singular of the last word ('robot legs' -> 'robot leg'); the word
    itself when it is not plural. SAM3 ignores `arms` (weapons) and `hands` but finds
    `arm` and `hand`, so the singular is always worth measuring."""
    head, _, last = phrase.rpartition(" ")
    if last in IRREGULAR_SINGULAR:
        word = IRREGULAR_SINGULAR[last]
    elif len(last) > 3 and last.endswith("ies"):
        word = last[:-3] + "y"
    elif len(last) > 4 and last.endswith(("ses", "xes", "shes", "ches", "zes")):
        word = last[:-2]
    elif len(last) > 3 and last.endswith("s") and not last.endswith("ss"):
        word = last[:-1]
    else:
        word = last
    return f"{head} {word}".strip()


def object_noun(object_name):
    """The noun to prefix part words with: 'dog figurine' -> 'dog', 'robot' -> 'robot',
    'lego minifigure' -> 'minifigure'; None for a long or empty description."""
    words = str(object_name or "").lower().split()
    if not words or len(words) > 3:
        return None
    if len(words) >= 2 and words[-1] in OBJECT_SUFFIXES:
        return words[-2]
    return words[-1]


def variant_phrases(chosen, limit=MAX_VARIANT_PHRASES):
    """{part word: [phrases to measure]} for the VLM's parts and main word: the word, the
    VLM's alternatives, its singular, and '<object> <word>'. Order = preference on a tie."""
    noun = object_noun(chosen.get("object"))
    alternatives = chosen.get("alternatives") or {}
    out = {}
    total = 0
    words = list(chosen.get("parts") or []) + ([chosen["main"]] if chosen.get("main") else [])
    for word in dict.fromkeys(words):
        phrases = [word]
        for phrase in list(alternatives.get(word) or []) + [singular(word)]:
            phrase = str(phrase).strip().lower()
            if phrase and phrase not in phrases:
                phrases.append(phrase)
        if noun and not word.startswith(noun + " ") and word != noun:
            compound = f"{noun} {word}"
            if compound not in phrases:
                phrases.append(compound)
        room = max(limit - total, 1)
        out[word] = phrases[:room]
        total += len(out[word])
    return out


def pick_variants(chosen, rows, whole=DEFAULT_WHOLE_VLM, own_views=DEFAULT_MIN_VIEWS,
                  max_growth=VARIANT_MAX_GROWTH):
    """{part word: best phrase or None}. The VLM's own word when SAM3 sees it in at least
    `own_views` of the views (and it is not a whole-object mask); otherwise the eligible
    variant seen in the most views -- among those, the one closest in size to the own
    word when SAM3 found the own word at all (and never more than `max_growth` x its
    per-view area: `male arm` was the whole upper body), else the smaller mask. A variant stands in
    for a word SAM3 does not know (`arms`); it is not a licence to pick the biggest mask
    (`glove` would swallow the forearm, `boot` the shin)."""
    by_name = {row["concept"]: row for row in rows}
    total_views = max((row.get("views", 0) for row in rows), default=0)
    picked = {}
    for word, phrases in variant_phrases(chosen).items():
        own = by_name.get(word)
        own_found = bool(own and own.get("views", 0) > 0 and own.get("area", 0) > 0)
        if own and own.get("eligible") and float(own.get("blob", own["area"])) < whole \
                and own["views"] >= own_views * max(total_views, 1):
            picked[word] = word
            continue
        best = None
        for index, phrase in enumerate(phrases):
            row = by_name.get(phrase)
            if not row or not row.get("eligible") or float(row.get("blob", row["area"])) >= whole:
                continue
            per_view = row["area"] / max(row["views"], 1)
            own_per_view = own["area"] / max(own["views"], 1) if own_found else None
            if own_found and per_view > max_growth * own_per_view:
                continue
            size = -abs(per_view - own_per_view) if own_found else -per_view
            key = (row["views"], round(size, 4), -index)
            if best is None or key > best[0]:
                best = (key, phrase)
        picked[word] = best[1] if best else None
    return picked


def union_masks(masks, foreground, concepts, phrase):
    masks = np.asarray(masks, dtype=bool)
    if phrase not in list(concepts):
        return None
    return masks[:, list(concepts).index(phrase)] & np.asarray(foreground, dtype=bool)


def resolve_overlaps(order, phrase_of, masks, foreground, concepts, iou=DEFAULT_SHORTLIST_IOU,
                     inside=DEFAULT_SHORTLIST_INSIDE):
    """(kept, dropped): a word goes when its mask is the same thing as an earlier kept
    word's (IoU >= `iou` over all views: `legs` beats `jeans`, `handle` beats `shaft`) or
    lies inside a kept word's mask (>= `inside` of its own pixels: a hand inside the arm,
    a cushion inside the ear cup) -- a sub-part folds into the piece it belongs to. Fewer,
    whole parts rather than many fragile ones. The container is decided by the masks, not
    the order, so a small word listed first still folds into the big one listed later."""
    unions = {}
    for word in order:
        mine = union_masks(masks, foreground, concepts, phrase_of.get(word) or word)
        if mine is not None:
            unions[word] = mine
    kept, dropped = [], []
    for word in order:
        mine = unions.get(word)
        if mine is None:
            kept.append(word)
            continue
        same = None
        for other in kept:
            theirs = unions.get(other)
            if theirs is None:
                continue
            inter = np.logical_and(mine, theirs).sum()
            union = np.logical_or(mine, theirs).sum()
            if union and inter / union >= iou:
                same = other
                break
        if same is None:
            kept.append(word)
        else:
            dropped.append((word, same))
    # containment, judged against every surviving word (big ones may come later in the list)
    folded = True
    while folded:
        folded = False
        for word in list(kept):
            mine = unions.get(word)
            if mine is None:
                continue
            area = mine.sum()
            for other in kept:
                theirs = unions.get(other)
                if other == word or theirs is None or theirs.sum() <= area:
                    continue
                if area and np.logical_and(mine, theirs).sum() / area >= inside:
                    kept.remove(word)
                    dropped.append((word, other))
                    folded = True
                    break
            if folded:
                break
    return kept, dropped


def part_spec(word, phrase):
    """The prompt spec downstream: the VLM's name, segmented by the phrase SAM3 knows."""
    return word if phrase in (None, word) else f"{word}={phrase}"


def propose_prompts(glb, work_dir, seg_glb, py_sam3, sam3_model, concept_bank,
                    flat_paint="auto", reuse=True, radius=2.0, resolution=512,
                    azimuths=AUTO_AZIMUTHS, elevations=AUTO_ELEVATIONS,
                    max_parts=DEFAULT_MAX_PARTS, mode="auto", guide_image=None):
    """Render a view ring and pick the part names.

    mode="auto": ask SAM3 for every bank concept and pick by the rules above.
    mode="smart" (智能分割模式): the VLM first names the object and its parts from the
    renders (vocabulary = the bank), SAM3 measures just those words, and the ones it finds
    become the prompts; if the VLM fails or none of its words is found, fall back to the
    full sweep with the VLM reviewing SAM3's candidates. `guide_image` (smart only): the
    user's reference segmentation, one flat colour per wanted part, shown to the VLM so
    the names follow it. Writes work_dir/auto_prompts.json.
    Returns {"prompts": [...],
    "unassigned_to": str | None, "proposal": {...}}; "prompts" is empty when nothing usable
    was recognised.
    """
    from data_toolkit.lift_sam3 import load_masks
    from merge_parts import flat_paint_stage, render_views

    if not concept_bank:
        raise ValueError("auto prompts need the concept bank (concept_bank / SEGVIGEN_CONCEPT_BANK)")
    views_dir = render_views(glb, os.path.join(work_dir, "views_auto"), azimuths, elevations,
                             radius, resolution, reuse)
    prompt_dir, painted = flat_paint_stage(
        seg_glb, views_dir, os.path.join(work_dir, "views_auto_flat"), flat_paint, reuse)
    concepts = bank_concepts(py_sam3, concept_bank, os.path.dirname(os.path.abspath(concept_bank)))
    suffix = "_grey" if painted else ""
    images = []
    if mode == "smart":
        # the textured renders, all of them: the grey flat paint made a gramophone a
        # "house" and a skull-studded chest a "toy car"
        with open(os.path.join(views_dir, "cameras.json"), encoding="utf-8") as handle:
            views = json.load(handle)["views"]
        images = [os.path.join(views_dir, view["image"]) for view in views][:8]

    proposal = None
    shortlist_error = None
    shortlist_proposal = None     # a weak shortlist result kept while the sweep is tried
    # SEGVIGEN_SMART_SHORTLIST=0: skip the VLM-first shortlist (A/B against the full sweep)
    if mode == "smart" and os.environ.get("SEGVIGEN_SMART_SHORTLIST", "1") != "0":
        # Shortlist first: the VLM names the parts from the renders, SAM3 only measures
        # those words. The full bank sweep below is the fallback.
        from smart_prompts import kimi_shortlist

        print(f"[smart] asking the VLM to name the object and its parts "
              f"(vocabulary of {len(concepts)} bank words"
              + (", following the guide image" if guide_image else "") + ") ...")
        chosen = ask_vlm_or_fail(lambda: kimi_shortlist(images, concepts, guide_image=guide_image),
                                 "naming the parts")
        shortlist_error = None
        print(f"[smart] {chosen.get('model', 'VLM')}: object={chosen['object']!r} "
              f"main={chosen['main']!r} parts={chosen['parts']}"
              + (f" (dropped, not in bank: {chosen['dropped']})" if chosen["dropped"] else ""))
        words = list(dict.fromkeys(
            phrase for phrases in variant_phrases(chosen).values() for phrase in phrases))
        if words:
            import hashlib

            digest = hashlib.sha1("|".join(words).encode("utf-8")).hexdigest()[:10]
            short_npz = os.path.join(work_dir, f"auto_shortlist_{digest}{suffix}.npz")
            if not (reuse and os.path.isfile(short_npz)):
                print(f"[smart] asking SAM3 for the {len(words)} shortlisted words and "
                      f"variants over {len(azimuths.split(','))} views ...")
                sam3_sweep(py_sam3, sam3_model, concept_bank, prompt_dir, views_dir,
                           painted, short_npz, words)
            short = load_masks(short_npz)
            rows = word_stats(short.masks, short.foreground, short.scores, short.concepts,
                              min_views=DEFAULT_SHORTLIST_MIN_VIEWS)
            proposal, reason = accept_shortlist(
                chosen, rows, max_parts=max_parts, masks=short.masks,
                foreground=short.foreground, concepts=short.concepts)
            if proposal is None:
                print(f"[smart] shortlist unusable ({reason}); falling back to the full bank sweep")
            else:
                proposal["covered"] = covered_share(
                    short.masks, short.foreground, short.concepts,
                    [proposal["phrases"][p] for p in proposal["part_names"]])
                kimi = proposal["kimi"]
                if kimi.get("variants_used"):
                    print(f"[smart] SAM3 knows these better: "
                          + ", ".join(f"{k} -> {v}" for k, v in kimi["variants_used"].items()))
                if kimi.get("not_found") or kimi.get("dropped_whole") or kimi.get("dropped_overlap"):
                    print(f"[smart] not found by SAM3: {kimi.get('not_found') or []}; "
                          f"whole-object words dropped: {kimi.get('dropped_whole') or []}; "
                          f"same thing as another word: {kimi.get('dropped_overlap') or {}}")
                if len(proposal["part_names"]) < DEFAULT_SHORTLIST_MIN_PARTS:
                    print(f"[smart] only {len(proposal['part_names'])} part found "
                          f"({proposal['covered']:.0%} of the silhouette); asking the full "
                          "bank sweep for a second opinion")
                    shortlist_proposal, proposal = proposal, None

    if proposal is None:
        masks_npz = os.path.join(work_dir, f"auto_concepts{suffix}.npz")
        if not (reuse and os.path.isfile(masks_npz)):
            print(f"[auto] asking SAM3 for all {len(concepts)} bank concepts over "
                  f"{len(azimuths.split(','))} views ...")
            sam3_sweep(py_sam3, sam3_model, concept_bank, prompt_dir, views_dir, painted,
                       masks_npz, concepts)
        mask_set = load_masks(masks_npz)
        proposal = propose_from_masks(mask_set.masks, mask_set.foreground, mask_set.scores,
                                      mask_set.concepts, max_parts=max_parts)
        proposal["mode"] = mode
        if mode == "smart":
            from smart_prompts import kimi_select

            if shortlist_error:
                proposal["shortlist_error"] = shortlist_error
            print(f"[smart] asking the VLM to review {len(proposal['candidates'])} candidate words ...")
            proposal["heuristic"] = {"main": proposal["main"], "parts": proposal["parts"]}
            chosen = ask_vlm_or_fail(lambda: kimi_select(images, proposal["candidates"], concepts),
                                     "reviewing the candidate words")
            proposal["kimi"] = chosen
            print(f"[smart] {chosen.get('model', 'VLM')}: object={chosen['object']!r} "
                  f"main={chosen['main']!r} parts={chosen['parts']}"
                  + (f" (dropped, not in bank: {chosen['dropped']})" if chosen["dropped"] else ""))
            chosen["parts"], whole_words = drop_whole_words(chosen["parts"], proposal["candidates"])
            if whole_words:
                chosen["dropped_whole"] = whole_words
                print(f"[smart] dropped {whole_words}: each mask covers half the silhouette "
                      f"or more, so it names the object rather than a part")
            # main + one part is a split (pineapple: fruit + leaves); only no part at all,
            # or parts with nothing to name the remainder, falls back to the rules
            if len(chosen["parts"]) >= 2 or (chosen["parts"] and chosen["main"]):
                proposal["parts"] = chosen["parts"][:max_parts]
                proposal["main"] = chosen["main"] or proposal["main"]
            else:
                print("[smart] VLM returned no usable part; keeping the rule-based pick")
    if shortlist_proposal is not None:
        # the sweep's review is the second opinion: it wins when it names more parts
        # and covers more (a one-word shortlist is what sent us here)
        if len(proposal.get("parts") or []) > len(shortlist_proposal["part_names"]) \
                and proposal.get("covered", 0.0) > shortlist_proposal["covered"]:
            proposal["shortlist_overruled"] = {
                "parts": shortlist_proposal["parts"], "covered": shortlist_proposal["covered"]}
            print(f"[smart] the sweep covers {proposal.get('covered', 0.0):.0%} vs the "
                  f"shortlist's {shortlist_proposal['covered']:.0%}; using the sweep")
        else:
            shortlist_proposal["sweep_overruled"] = {
                "parts": proposal.get("parts"), "covered": proposal.get("covered")}
            proposal = shortlist_proposal
            print(f"[smart] the sweep did not cover more; keeping the shortlist")
    prompts = list(proposal["parts"])
    main = proposal["main"]
    if main and main not in prompts and main not in [p.split("=", 1)[0] for p in prompts]:
        phrase = (proposal.get("phrases") or {}).get(main)
        prompts.append(part_spec(main, phrase))
    unassigned_to = main or (prompts[0] if prompts else None)
    if guide_image:
        proposal["guide_image"] = os.path.basename(guide_image)
    result = {"prompts": prompts, "unassigned_to": unassigned_to, "painted": painted,
              "proposal": proposal, "separate": dict(proposal.get("separate") or {})}
    with open(os.path.join(work_dir, "auto_prompts.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    if prompts:
        print(f"[auto] parts {proposal['parts']} + remainder -> {unassigned_to} "
              f"(parts cover {proposal['covered']:.0%} of the silhouette)")
    else:
        print("[auto] SAM3 recognised no usable part word on this model")
    return result
