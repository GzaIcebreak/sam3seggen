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

from prompt_specs import normalize_part_specs, part_names, validate_target_name
from sam3_to_2dmap import (
    DEFAULT_SAM3, attach_decoder_lora, foreground_mask, load_concept_bank, load_sam3,
    segment_prompts,
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
    parser.add_argument("--concept_bank", default=os.environ.get(
        "SEGVIGEN_CONCEPT_BANK", "/root/autodl-tmp/datasets/concept_bank_v3/bank.pt"),
                        help="v3 bank.pt; the maps.png stain. Empty string = raw SAM3.")
    parser.add_argument("--raw", action="store_true",
                        help="Keep overlapping unions instead of the v3 smallest-first overlay")
    parser.add_argument("--require_masks", action="store_true",
                        help="Exit if a prompt (other than --unassigned_to) is unseen in every view")
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"SAM3 device={device} views={len(manifest['views'])} concepts={concepts} "
          f"bank={'v3' if bank_path else 'off'} overlay={'raw' if args.raw else 'v3'}")
    processor, model = load_sam3(args.model, device)
    bank = load_concept_bank(bank_path, device)
    if bank is not None:
        attach_decoder_lora(model, bank, bank_path, None, device)

    mask_rows, foreground_rows, score_rows = [], [], []
    detections = {concept: 0 for concept in concepts}
    for view in manifest["views"]:
        image = Image.open(os.path.join(views_dir, view["image"]))
        print(f"[{view['name']}]")
        found = {
            part["prompt"]: part
            for part in segment_prompts(
                processor, model, image, concepts, threshold, device, bank=bank)
        }
        foreground = foreground_mask(image)
        masks = np.zeros((len(concepts), *foreground.shape), dtype=bool)
        scores = np.zeros(len(concepts), dtype=np.float32)
        for index, concept in enumerate(concepts):
            hit = found.get(concept)
            if hit is None:
                continue
            # SAM3 masks bleed a few pixels past the silhouette; those pixels would
            # otherwise back-project onto whatever surface lies behind the object.
            masks[index] = hit["mask"].astype(bool) & foreground
            scores[index] = float(hit["score"])
        if not args.raw:
            masks = overlay_v3(masks, foreground)
            for index in range(len(concepts)):
                if not masks[index].any():
                    scores[index] = 0
        for index, concept in enumerate(concepts):
            detections[concept] += int(masks[index].any())
        mask_rows.append(np.packbits(masks, axis=-1))
        foreground_rows.append(np.packbits(foreground, axis=-1))
        score_rows.append(scores)

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
        width=np.int32(manifest["resolution"]),
        concepts=np.array(concepts),
        owners=np.array(owners),
        views=np.array([view["name"] for view in manifest["views"]]),
        part_order=np.array(part_names(specs)),
        unassigned_to=np.array(args.unassigned_to or ""),
    )
    print(f"saved {out}")
    for concept, count in detections.items():
        print(f"  {concept}: detected in {count}/{len(manifest['views'])} views")


if __name__ == "__main__":
    main()
