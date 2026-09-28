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

    candidates = [i for i in np.flatnonzero(eligible & (area >= min_area) & (area < whole))]
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
    rows = [{"concept": names[i], "area": float(area[i]), "score": float(mean_score[i]),
             "views": int(detected[i]), "eligible": bool(eligible[i])}
            for i in np.argsort(-(area * mean_score))[:20] if detected[i] > 0]
    return {
        "main": names[main] if main is not None else None,
        "parts": [names[j] for j in picked],
        "covered": float(covered.sum() / fg_total),
        "candidates": rows,
    }


def bank_concepts(py_sam3, bank_path, cache_dir):
    """The bank's concept names, read once through the SAM3 venv (torch lives there)."""
    cache = os.path.join(cache_dir, "concept_bank_names.txt")
    if not os.path.isfile(cache):
        code = ("import sys, torch; b = torch.load(sys.argv[1], map_location='cpu', "
                "weights_only=False); print('\\n'.join(b['names']))")
        text = subprocess.run([py_sam3, "-c", code, bank_path], check=True,
                              capture_output=True, text=True).stdout
        with open(cache, "w", encoding="utf-8") as handle:
            handle.write(text)
    with open(cache, encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def propose_prompts(glb, work_dir, seg_glb, py_sam3, sam3_model, concept_bank,
                    flat_paint="auto", reuse=True, radius=2.0, resolution=512,
                    azimuths=AUTO_AZIMUTHS, elevations=AUTO_ELEVATIONS,
                    max_parts=DEFAULT_MAX_PARTS, mode="auto"):
    """Render a view ring, ask SAM3 for every bank concept, and pick the part names.

    mode="auto" picks by the rules above; mode="smart" (智能分割模式) hands the candidates
    and four renders to Kimi, which knows what the object is and drops words that do not
    belong to it. Writes work_dir/auto_prompts.json. Returns {"prompts": [...],
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
    masks_npz = os.path.join(work_dir, "auto_concepts.npz")
    if not (reuse and os.path.isfile(masks_npz)):
        print(f"[auto] asking SAM3 for all {len(concepts)} bank concepts over "
              f"{len(azimuths.split(','))} views ...")
        subprocess.run([
            py_sam3, os.path.join(ROOT, "sam3_multiview.py"),
            "--views_dir", prompt_dir, "--out", masks_npz, "--raw",
            "--model", sam3_model, "--concept_bank", concept_bank,
            "--prompts", *concepts,
        ], check=True, stdout=subprocess.DEVNULL)
    mask_set = load_masks(masks_npz)
    proposal = propose_from_masks(mask_set.masks, mask_set.foreground, mask_set.scores,
                                  mask_set.concepts, max_parts=max_parts)
    proposal["mode"] = mode
    if mode == "smart":
        from smart_prompts import kimi_select

        with open(os.path.join(prompt_dir, "cameras.json"), encoding="utf-8") as handle:
            views = json.load(handle)["views"]
        images = [os.path.join(prompt_dir, view["image"]) for view in views][::2][:4]
        print(f"[smart] asking Kimi to review {len(proposal['candidates'])} candidate words ...")
        chosen = kimi_select(images, proposal["candidates"], concepts)
        proposal["heuristic"] = {"main": proposal["main"], "parts": proposal["parts"]}
        proposal["kimi"] = chosen
        print(f"[smart] Kimi: object={chosen['object']!r} main={chosen['main']!r} "
              f"parts={chosen['parts']}" + (f" (dropped, not in bank: {chosen['dropped']})"
                                            if chosen["dropped"] else ""))
        if len(chosen["parts"]) >= 2:
            proposal["parts"] = chosen["parts"][:max_parts]
            proposal["main"] = chosen["main"] or proposal["main"]
        else:
            print("[smart] Kimi returned fewer than two usable parts; keeping the rule-based pick")
    prompts = list(proposal["parts"])
    main = proposal["main"]
    if main and main not in prompts:
        prompts.append(main)
    unassigned_to = main or (prompts[0] if prompts else None)
    result = {"prompts": prompts, "unassigned_to": unassigned_to, "painted": painted,
              "proposal": proposal}
    with open(os.path.join(work_dir, "auto_prompts.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    if prompts:
        print(f"[auto] parts {proposal['parts']} + remainder -> {unassigned_to} "
              f"(parts cover {proposal['covered']:.0%} of the silhouette)")
    else:
        print("[auto] SAM3 recognised no usable part word on this model")
    return result
