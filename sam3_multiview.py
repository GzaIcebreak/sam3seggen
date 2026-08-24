"""Run SAM3 over a whole view grid and save raw per-concept masks.

Run with .venv_holo (transformers 5.x), like sam3_to_2dmap.py.

Unlike sam3_to_2dmap.py this deliberately does *not* colourise. Painting a map
forces three lossy decisions in 2D -- overlaps resolved by paint order, concepts
collapsed into a part colour, unclaimed foreground absorbed by a named part -- and
each one throws away evidence the 3D vote could have used. Masks and scores are
kept as SAM3 produced them so those decisions can be made once, on the mesh.

A concept that no view detects is an error; a concept missing from *some* views is
normal and expected (a hand is hidden from behind).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from prompt_specs import normalize_part_specs, part_names, validate_target_name
from sam3_to_2dmap import foreground_mask, load_sam3, segment_prompts


def concept_table(specs):
    """Flatten specs into parallel concept/part lists, keeping first-seen order."""
    concepts, owners = [], []
    for name, prompts in specs:
        for prompt in prompts:
            if prompt not in concepts:
                concepts.append(prompt)
                owners.append(name)
    return concepts, owners


def main():
    parser = argparse.ArgumentParser(description="SAM3 over a view grid -> raw per-concept masks")
    parser.add_argument("--views_dir", required=True, help="Directory holding cameras.json and renders")
    parser.add_argument("--out", required=True, help="Output .npz of packed masks")
    parser.add_argument("--prompts", nargs="+", required=True,
                        help="One entry per output part; join concepts with '+' to merge them, "
                             "e.g. 'armor staff base body=head+face+hand+boot'.")
    parser.add_argument("--unassigned_to", default=None,
                        help="Recorded for the 3D stage, which assigns faces no concept claimed.")
    parser.add_argument("--model", default="facebook/sam3")
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    views_dir = os.path.abspath(args.views_dir)
    with open(os.path.join(views_dir, "cameras.json"), "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    specs = normalize_part_specs(args.prompts)
    validate_target_name(args.unassigned_to, part_names(specs))
    concepts, owners = concept_table(specs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"SAM3 device={device} views={len(manifest['views'])} concepts={concepts}")
    processor, model = load_sam3(args.model, device)

    mask_rows, foreground_rows, score_rows = [], [], []
    detections = {concept: 0 for concept in concepts}
    for view in manifest["views"]:
        image = Image.open(os.path.join(views_dir, view["image"]))
        print(f"[{view['name']}]")
        found = {
            part["prompt"]: part
            for part in segment_prompts(processor, model, image, concepts, args.threshold, device)
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
            detections[concept] += int(masks[index].any())
        mask_rows.append(np.packbits(masks, axis=-1))
        foreground_rows.append(np.packbits(foreground, axis=-1))
        score_rows.append(scores)

    # Concepts feeding the --unassigned_to part may legitimately see nothing: that
    # part's job is to absorb whatever no mask claimed, so it needs no detections.
    never_seen = [
        concept
        for concept, count in detections.items()
        if count == 0 and owners[concepts.index(concept)] != args.unassigned_to
    ]
    if never_seen:
        raise SystemExit(f"SAM3 found no mask in any view for: {never_seen}")

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
