"""Run SAM3 on rendered views and save one union mask per prompt (path A, .venv_holo).

    python finetune/sam3_masks.py --views <view_dir> [<view_dir> ...] [--threshold 0.3]

Each view dir must hold render.png and prompts.json (list of noun phrases). Writes
sam3_masks.npz {"masks": bool[K,H,W], "prompts": [...], "scores": [...]} into the same dir.
Only depends on transformers / torch / PIL so it can live in the SAM3 environment.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
from PIL import Image

from sam3_to_2dmap import load_concept_bank, load_sam3, segment_prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--views", nargs="+", required=True)
    parser.add_argument("--model", default="facebook/sam3")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--concept_bank", default=None, help="Stage-B bank.pt (finetune/concept_bank.py)")
    parser.add_argument("--no_e0", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = load_sam3(args.model, device)
    bank = load_concept_bank(args.concept_bank, device)
    for vdir in args.views:
        out = os.path.join(vdir, "sam3_masks.npz")
        if os.path.exists(out) and not args.force:
            print(f"skip {vdir}: masks exist")
            continue
        render = os.path.join(vdir, "render.png")
        with open(os.path.join(vdir, "prompts.json"), "r", encoding="utf-8") as f:
            prompts = json.load(f)
        image = Image.open(render)
        print(f"{vdir}: {len(prompts)} prompts")
        found = segment_prompts(processor, model, image, prompts, args.threshold, device,
                                bank=bank, use_e0=not args.no_e0)
        by_prompt = {p["prompt"]: p for p in found}
        h, w = image.size[1], image.size[0]
        masks = np.zeros((len(prompts), h, w), dtype=bool)
        scores = np.zeros(len(prompts), dtype=np.float32)
        dim = len(found[0]["text_vec"]) if found else 256
        text_vecs = np.zeros((len(prompts), dim), dtype=np.float32)
        for i, prompt in enumerate(prompts):
            if prompt in by_prompt:
                masks[i] = by_prompt[prompt]["mask"]
                scores[i] = by_prompt[prompt]["score"]
                text_vecs[i] = by_prompt[prompt]["text_vec"]
        np.savez_compressed(out, masks=masks, prompts=np.array(prompts, dtype=object), scores=scores,
                            text_vecs=text_vecs, concept_bank=np.array(args.concept_bank or "", dtype=object))
        print(f"  saved {out}: {int((scores > 0).sum())}/{len(prompts)} prompts detected")


if __name__ == "__main__":
    main()
