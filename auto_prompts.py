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


def accept_shortlist(chosen, rows, max_parts=DEFAULT_MAX_PARTS, whole=DEFAULT_WHOLE_VLM):
    """(proposal, reason). The VLM's shortlist after SAM3 has measured it: words SAM3 did
    not find are dropped, words whose biggest blob covers `whole` of the object are
    whole-object words, and what is left must be a main word plus at least one part."""
    by_name = {row["concept"]: row for row in rows}
    found = [p for p in chosen["parts"] if by_name.get(p, {}).get("eligible")]
    missing = [p for p in chosen["parts"] if p not in found]
    kept, whole_words = drop_whole_words(found, rows, whole)
    main = chosen.get("main")
    if not main:
        return None, "the VLM named no main-body word"
    if not kept:
        return None, (f"none of the VLM's parts survived: not found by SAM3 {missing}, "
                      f"whole-object words {whole_words}")
    proposal = {"main": main, "parts": kept[:max_parts], "candidates": rows,
                "mode": "smart", "shortlist": True, "kimi": dict(chosen),
                "separate": {w: n for w, n in (chosen.get("separate") or {}).items() if w in kept}}
    if missing:
        proposal["kimi"]["not_found"] = missing
    if whole_words:
        proposal["kimi"]["dropped_whole"] = whole_words
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
        with open(os.path.join(prompt_dir, "cameras.json"), encoding="utf-8") as handle:
            views = json.load(handle)["views"]
        images = [os.path.join(prompt_dir, view["image"]) for view in views][::2][:4]

    proposal = None
    if mode == "smart":
        # Shortlist first: the VLM names the parts from the renders, SAM3 only measures
        # those words. The full bank sweep below is the fallback.
        from smart_prompts import kimi_shortlist

        print(f"[smart] asking the VLM to name the object and its parts "
              f"(vocabulary of {len(concepts)} bank words"
              + (", following the guide image" if guide_image else "") + ") ...")
        try:
            chosen = kimi_shortlist(images, concepts, guide_image=guide_image)
        except Exception as error:  # a VLM hiccup must not sink a twenty-minute job
            print(f"[smart] VLM failed ({type(error).__name__}: {str(error)[:160]}); "
                  "falling back to the full bank sweep")
            shortlist_error = f"{type(error).__name__}: {error}"
        else:
            shortlist_error = None
            print(f"[smart] {chosen.get('model', 'VLM')}: object={chosen['object']!r} "
                  f"main={chosen['main']!r} parts={chosen['parts']}"
                  + (f" (dropped, not in bank: {chosen['dropped']})" if chosen["dropped"] else ""))
            words = list(dict.fromkeys(chosen["parts"] + ([chosen["main"]] if chosen["main"] else [])))
            if words:
                import hashlib

                digest = hashlib.sha1("|".join(words).encode("utf-8")).hexdigest()[:10]
                short_npz = os.path.join(work_dir, f"auto_shortlist_{digest}{suffix}.npz")
                if not (reuse and os.path.isfile(short_npz)):
                    print(f"[smart] asking SAM3 for the {len(words)} shortlisted words over "
                          f"{len(azimuths.split(','))} views ...")
                    sam3_sweep(py_sam3, sam3_model, concept_bank, prompt_dir, views_dir,
                               painted, short_npz, words)
                short = load_masks(short_npz)
                rows = word_stats(short.masks, short.foreground, short.scores, short.concepts,
                                  min_views=DEFAULT_SHORTLIST_MIN_VIEWS)
                proposal, reason = accept_shortlist(chosen, rows, max_parts=max_parts)
                if proposal is None:
                    print(f"[smart] shortlist unusable ({reason}); falling back to the full bank sweep")
                else:
                    proposal["covered"] = covered_share(short.masks, short.foreground,
                                                        short.concepts, proposal["parts"])
                    extra = proposal["kimi"].get("not_found"), proposal["kimi"].get("dropped_whole")
                    if any(extra):
                        print(f"[smart] not found by SAM3: {extra[0] or []}; whole-object words "
                              f"dropped: {extra[1] or []}")

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
            try:
                chosen = kimi_select(images, proposal["candidates"], concepts)
            except Exception as error:  # a VLM hiccup must not sink a twenty-minute job
                proposal["kimi_error"] = f"{type(error).__name__}: {error}"
                print(f"[smart] VLM failed ({proposal['kimi_error'][:160]}); keeping the rule-based pick")
            else:
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
    prompts = list(proposal["parts"])
    main = proposal["main"]
    if main and main not in prompts:
        prompts.append(main)
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
