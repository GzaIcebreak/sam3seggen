"""Run SAM3 over a whole view grid and save per-concept masks.

Run with .venv_holo (transformers 5.x), like sam3_to_2dmap.py.

Default is concept-bank v3 + the same smallest-first overlay maps.png used: a pixel
belongs to at most one prompt, the more specific mask winning. `--raw` keeps the
overlapping unions instead, which is what used to hand the backpack to `arm`.

A concept missing from *some* views is normal (a hand is hidden from behind). A
concept no view detects is skipped by default -- the rest of the prompts still
vote -- so a leftover word does not abort the run. `--require_masks` restores
the old fail.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from phrase_rescue import (
    choose_phrase, phrase_stats, rescue_enabled, rescue_phrases, weak_concepts,
)
from prompt_specs import normalize_part_specs, part_names, validate_target_name
from sam3_to_2dmap import (
    DEFAULT_SAM3, attach_decoder_lora, auto_parts, foreground_mask, load_concept_bank,
    load_sam3, rank_fields, rank_parts, segment_prompts,
)

# Calibrated with the bank; raw SAM3 scores sit lower.
BANK_THRESHOLD = 0.4
PLAIN_THRESHOLD = 0.3


def concept_table(specs):
    """Flatten specs into parallel concept/part lists, keeping first-seen order."""
    concepts, owners = [], []
    for name, prompts in specs:
        for prompt in prompts:
            if prompt not in concepts:
                concepts.append(prompt)
                owners.append(name)
    return concepts, owners


def unseen_concepts(detections, concepts, owners, unassigned_to=None):
    """Concepts no view detected, ignoring the catch-all part (it is allowed to be empty)."""
    return [
        concept
        for concept, count in detections.items()
        if count == 0 and owners[concepts.index(concept)] != unassigned_to
    ]


def overlay_v3(masks, foreground):
    """Smallest-first disjoint overlay: the rule maps.png's stain was painted with.

    A large `arm` union that covers the backpack loses those pixels to whatever
    smaller mask already claimed them, and to nothing if no smaller mask did --
    then `torso` (or unassigned) can still take the backpack on the mesh.
    """
    painted = np.zeros(masks.shape, dtype=bool)
    occupied = np.zeros(foreground.shape, dtype=bool)
    order = np.argsort([int(m.sum()) for m in masks])
    for index in order:
        free = masks[index] & foreground & ~occupied
        painted[index] = free
        occupied |= free
    return painted


def rescue_weak_words(concepts, owners, raw, manifest, views_dir, unassigned_to,
                      processor, model, threshold, device, bank):
    """{word: record} for the plain words SAM3 barely saw that another phrasing finds
    clearly better (phrase_rescue.py). One extra SAM3 pass over the views, only for the
    candidate phrases, and only when some word is weak."""
    n_views = len(manifest["views"])
    weak = weak_concepts(concepts, owners, raw, n_views, unassigned_to)
    if not weak:
        return {}
    trials = {word: rescue_phrases(word, taken=concepts) for word in weak}
    phrases = list(dict.fromkeys(p for options in trials.values() for p, _ in options))
    print(f"[rescue] SAM3 barely sees {weak}; measuring {phrases}")
    measured = {phrase: [] for phrase in phrases}
    for view in manifest["views"]:
        image = Image.open(os.path.join(views_dir, view["image"]))
        foreground = foreground_mask(image)
        silhouette = max(int(foreground.sum()), 1)
        for part in segment_prompts(processor, model, image, phrases, threshold, device, bank=bank):
            if part["prompt"] in measured:
                measured[part["prompt"]].append(
                    (float(part["score"]), int((part["mask"].astype(bool) & foreground).sum()) / silhouette))
    rescued = {}
    for word, options in trials.items():
        own = phrase_stats(raw.get(word, []), n_views)
        candidates = [dict(phrase_stats(measured[p], n_views), phrase=p, tier=t) for p, t in options]
        pick = choose_phrase(own, candidates, n_views)
        tried = ", ".join(f"{c['phrase']} {c['seen']}/{n_views} {c['score']:.2f}" for c in candidates)
        if pick is None:
            print(f"[rescue] {word}: {own['seen']}/{n_views} {own['score']:.2f}; nothing better ({tried})")
            continue
        print(f"[rescue] {word}: {own['seen']}/{n_views} {own['score']:.2f} -> '{pick['phrase']}' "
              f"{pick['seen']}/{n_views} {pick['score']:.2f} (tried {tried})")
        rescued[word] = {"phrase": pick["phrase"], "seen": pick["seen"], "score": pick["score"],
                         "own_seen": own["seen"], "own_score": own["score"], "views": n_views}
    return rescued


def main():
    parser = argparse.ArgumentParser(description="SAM3 over a view grid -> raw per-concept masks")
    parser.add_argument("--views_dir", required=True, help="Directory holding cameras.json and renders")
    parser.add_argument("--out", required=True, help="Output .npz of packed masks")
    parser.add_argument("--prompts", nargs="+", required=True,
                        help="One entry per output part; join concepts with '+' to merge them, "
                             "e.g. 'armor staff base body=head+face+hand+boot'.")
    parser.add_argument("--unassigned_to", default=None,
                        help="Recorded for the 3D stage, which assigns faces no concept claimed.")
    parser.add_argument("--model", default=DEFAULT_SAM3)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Score gate. Default 0.4 with the concept bank, 0.3 without.")
    parser.add_argument("--concept_bank", default=os.environ.get("SEGVIGEN_CONCEPT_BANK") or next(
        (p for p in (os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "concept_bank_v6", "bank.pt"),
                     "/root/autodl-tmp/datasets/concept_bank_v3/bank.pt") if os.path.exists(p)),
        "/root/autodl-tmp/datasets/concept_bank_v3/bank.pt"),
                        help="v3 bank.pt; the maps.png stain. Empty string = raw SAM3.")
    parser.add_argument("--raw", action="store_true",
                        help="Keep overlapping unions instead of the v3 smallest-first overlay")
    parser.add_argument("--extra_views_dir", default=None,
                        help="A second render of the same cameras (the unpainted grey views when "
                             "--views_dir is flat-painted). Each prompt keeps, per view, whichever of "
                             "the two images gave it the higher score. --assign paint only.")
    parser.add_argument("--no_rescue", action="store_true",
                        help="Keep a plain prompt word SAM3 barely sees as it is instead of trying "
                             "other phrasings ('animal head' for 'head'); see phrase_rescue.py")
    parser.add_argument("--require_masks", action="store_true",
                        help="Exit if a prompt (other than --unassigned_to) is unseen in every view")
    parser.add_argument("--assign", choices=["paint", "rank", "auto"],
                        default=os.environ.get("SEGVIGEN_ASSIGN", "paint"),
                        help="paint = v3 score-threshold overlay; rank = that overlay edited by the "
                             "EASE Mask RankGNN (--rank_model); auto = rank unless it deletes a prompt")
    parser.add_argument("--rank_model", default=os.environ.get("SEGVIGEN_RANK_MODEL") or None,
                        help="mask_rank.py checkpoint for --assign rank/auto; missing = fall back to paint")
    parser.add_argument("--rank_drop", type=float, default=0.2,
                        help="--assign rank: drop an overlaid mask the ranker scores below this")
    parser.add_argument("--rank_add", type=float, default=0.9,
                        help="--assign rank: add a mask the overlay skipped that scores at least this")
    args = parser.parse_args()

    views_dir = os.path.abspath(args.views_dir)
    with open(os.path.join(views_dir, "cameras.json"), "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    specs = normalize_part_specs(args.prompts)
    validate_target_name(args.unassigned_to, part_names(specs))
    concepts, owners = concept_table(specs)
    bank_path = args.concept_bank or None
    threshold = args.threshold if args.threshold is not None else (
        BANK_THRESHOLD if bank_path else PLAIN_THRESHOLD)

    assign, rank_model = args.assign, args.rank_model
    if assign == "paint":
        rank_model = None   # the plain overlay never consults the ranker, even if one is configured
    if assign in ("rank", "auto"):
        if not rank_model or not os.path.exists(rank_model):
            print(f"rank model not found ({rank_model}); painting with the plain overlay")
            assign, rank_model = "paint", None
        elif not bank_path:
            print("--assign rank/auto needs the concept bank the ranker was trained with; "
                  "painting with the plain overlay")
            assign, rank_model = "paint", None
    extra_dir = os.path.abspath(args.extra_views_dir) if args.extra_views_dir else None
    if extra_dir and assign != "paint":
        print(f"--extra_views_dir is only read with --assign paint; ignoring it for {assign}")
        extra_dir = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    overlay = "raw" if args.raw else ("v3" if assign == "paint" else assign)
    print(f"SAM3 device={device} views={len(manifest['views'])} concepts={concepts} "
          f"bank={'on' if bank_path else 'off'} overlay={overlay}"
          + (f" ranker={os.path.basename(rank_model)} drop={args.rank_drop:g} add={args.rank_add:g}"
             if rank_model else ""))
    processor, model = load_sam3(args.model, device)
    bank = load_concept_bank(bank_path, device)
    if bank is not None:
        attach_decoder_lora(model, bank, bank_path, None, device)

    def sweep(concepts):
        """SAM3 over every view: the packed rows to save, the per-concept detection
        counts after the overlay, and the raw hits [(score, silhouette share)] before it."""
        mask_rows, foreground_rows, score_rows, instance_rows = [], [], [], []
        detections = {concept: 0 for concept in concepts}
        raw = {concept: [] for concept in concepts}
        for view in manifest["views"]:
            image = Image.open(os.path.join(views_dir, view["image"]))
            print(f"[{view['name']}]")
            sweep_view(view, image, concepts, mask_rows, foreground_rows, score_rows,
                       detections, raw, instance_rows)
        return mask_rows, foreground_rows, score_rows, detections, raw, instance_rows

    def sweep_view(view, image, concepts, mask_rows, foreground_rows, score_rows, detections, raw,
                   instance_rows):
        if assign == "paint":
            parts = segment_prompts(processor, model, image, concepts, threshold, device, bank=bank)
            if extra_dir:
                extra = segment_prompts(processor, model, Image.open(os.path.join(extra_dir, view["image"])),
                                        concepts, threshold, device, bank=bank)
                best = {part["prompt"]: part for part in parts}
                taken = []
                for part in extra:
                    mine = best.get(part["prompt"])
                    if mine is None or float(part["score"]) > float(mine["score"]):
                        best[part["prompt"]] = part
                        taken.append(part["prompt"])
                if taken:
                    print(f"  from the unpainted render: {sorted(taken)}")
                parts = list(best.values())
        else:
            # One SAM3 forward pass; the ranker weighs each prompt's top-K candidates against
            # each other and edits v3's overlay set. rank_parts already returns disjoint,
            # smallest-first overlaid masks, so overlay_v3 below is skipped for it.
            fields = rank_fields(processor, model, image, concepts, device, rank_model, bank=bank)
            if assign == "rank":
                parts = rank_parts(fields, concepts, threshold, args.rank_drop, args.rank_add, "small")
            else:
                parts, record = auto_parts(fields, concepts, threshold, args.rank_drop, args.rank_add, "small")
                print(f"  auto -> {record['chosen']} (lost_to_ranker={record['lost_to_ranker']})")
            del fields
        found = {part["prompt"]: part for part in parts}
        foreground = foreground_mask(image)
        silhouette = max(int(foreground.sum()), 1)
        for concept in concepts:
            hit = found.get(concept)
            if hit is not None:
                raw[concept].append((float(hit["score"]),
                                     int((hit["mask"].astype(bool) & foreground).sum()) / silhouette))
        masks = np.zeros((len(concepts), *foreground.shape), dtype=bool)
        scores = np.zeros(len(concepts), dtype=np.float32)
        instances = np.zeros(len(concepts), dtype=np.int32)
        for index, concept in enumerate(concepts):
            hit = found.get(concept)
            if hit is None:
                continue
            instances[index] = int(hit.get("instances", 1))
            # SAM3 masks bleed a few pixels past the silhouette; those pixels would
            # otherwise back-project onto whatever surface lies behind the object.
            masks[index] = hit["mask"].astype(bool) & foreground
            scores[index] = float(hit["score"])
        if not args.raw and assign == "paint":
            masks = overlay_v3(masks, foreground)
            for index in range(len(concepts)):
                if not masks[index].any():
                    scores[index] = 0
        for index, concept in enumerate(concepts):
            detections[concept] += int(masks[index].any())
        mask_rows.append(np.packbits(masks, axis=-1))
        foreground_rows.append(np.packbits(foreground, axis=-1))
        score_rows.append(scores)
        instance_rows.append(instances)

    mask_rows, foreground_rows, score_rows, detections, raw, instance_rows = sweep(concepts)
    rescued = {}
    if rescue_enabled(args.no_rescue):
        rescued = rescue_weak_words(concepts, owners, raw, manifest, views_dir, args.unassigned_to,
                                    processor, model, threshold, device, bank)
        if rescued:
            for word, record in rescued.items():
                concepts[concepts.index(word)] = record["phrase"]
            print(f"[rescue] SAM3 again with {concepts}")
            mask_rows, foreground_rows, score_rows, detections, raw, instance_rows = sweep(concepts)

    # Concepts feeding the --unassigned_to part may legitimately see nothing: that
    # part's job is to absorb whatever no mask claimed, so it needs no detections.
    never_seen = unseen_concepts(detections, concepts, owners, args.unassigned_to)
    if never_seen and args.require_masks:
        raise SystemExit(f"SAM3 found no mask in any view for: {never_seen}")
    if never_seen:
        print(f"skipping unused prompts (no mask in any view): {never_seen}")

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(
        out,
        masks=np.stack(mask_rows),
        foreground=np.stack(foreground_rows),
        scores=np.stack(score_rows),
        instances=np.stack(instance_rows),
        width=np.int32(manifest["resolution"]),
        concepts=np.array(concepts),
        owners=np.array(owners),
        views=np.array([view["name"] for view in manifest["views"]]),
        part_order=np.array(part_names(specs)),
        unassigned_to=np.array(args.unassigned_to or ""),
        assign=np.array(assign),
        rescued=np.array(json.dumps(rescued)),
    )
    print(f"saved {out}")
    for concept, count in detections.items():
        print(f"  {concept}: detected in {count}/{len(manifest['views'])} views")


if __name__ == "__main__":
    main()
