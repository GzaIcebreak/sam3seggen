"""Turn a rendered view into a SegviGen 2D part-color map using SAM3.

Run with .venv_holo (transformers 5.x). Do not mix with the SegviGen venv.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

from prompt_specs import normalize_part_specs, part_names, validate_target_name

PALETTE = [
    (220, 40, 40),
    (40, 90, 230),
    (30, 180, 70),
    (240, 210, 30),
    (40, 200, 210),
    (230, 70, 180),
    (140, 50, 200),
    (240, 130, 30),
    (20, 120, 120),
    (180, 180, 40),
    (80, 40, 160),
    (40, 160, 40),
]


def pick_separated_colors(count: int) -> list[tuple[int, int, int]]:
    """Greedy colours with the largest remaining distance to those already chosen."""
    if count <= 0:
        return []
    remaining = list(PALETTE)
    chosen = [remaining.pop(0)]
    while len(chosen) < count:
        if not remaining:
            remaining = list(PALETTE)
        best = max(
            range(len(remaining)),
            key=lambda index: min(
                np.linalg.norm(np.array(remaining[index], dtype=np.float64) - np.array(color, dtype=np.float64))
                for color in chosen
            ),
        )
        chosen.append(remaining.pop(best))
    return chosen

# Foreground pixels no prompt claimed. Keeping them distinct from the white
# background stops rembg/DINO from reading the silhouette as empty space.
UNASSIGNED = (150, 150, 150)


def _to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def _remap_detector_keys(state):
    """facebook/sam3 is a video ckpt; Sam3Model only needs detector_model.*."""
    remapped = {}
    for key, value in state.items():
        if key.startswith("tracker_model.") or key.startswith("tracker_neck."):
            continue
        if key.startswith("detector_model."):
            key = key[len("detector_model.") :]
        remapped[key] = value
    return remapped


def load_sam3(model_id: str, device: str):
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import Sam3Config, Sam3Model, Sam3Processor

    local = snapshot_download(model_id, local_files_only=True)
    processor = Sam3Processor.from_pretrained(local)
    config = Sam3Config.from_pretrained(local)
    model = Sam3Model(config)
    raw = load_file(os.path.join(local, "model.safetensors"))
    mapped = _remap_detector_keys(raw)
    incompatible = model.load_state_dict(mapped, strict=False)
    missing = [k for k in incompatible.missing_keys if "tracker" not in k]
    if missing:
        print(f"warning: {len(missing)} missing keys, e.g. {missing[:5]}")
    extra = incompatible.unexpected_keys
    if extra:
        print(f"warning: {len(extra)} unexpected keys, e.g. {extra[:5]}")
    model.to(device).eval()
    return processor, model


def load_concept_bank(path: str | None, device: str):
    """Optional Stage-B bank (finetune/concept_bank.py); None when no path is given."""
    if not path:
        return None
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetune"))
    import sam3_bank
    bank = sam3_bank.ConceptBank.load(path, device)
    print(f"concept bank: {len(bank.names)} names, |E_0|={bank.E_0.norm().item():.3f}, template {bank.template!r}")
    return bank


@torch.no_grad()
def segment_prompts(processor, model, image: Image.Image, prompts: list[str], threshold: float, device: str,
                    bank=None, use_e0: bool = True):
    """One union mask per prompt. With `bank`, the learned text offsets are applied (frozen SAM3).

    Each returned part also carries `text_vec`: the mean-pooled projected text features (+ offset),
    the 256-d vector SegviGen's legend tokens are built from.
    """
    w, h = image.size
    img_inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    pixel_values = img_inputs["pixel_values"].to(device)
    vision_embeds = model.get_vision_features(pixel_values=pixel_values)

    parts = []
    for prompt in prompts:
        text_inputs = _to_device(processor(text=prompt, return_tensors="pt"), device)
        text_out = model.get_text_features(input_ids=text_inputs["input_ids"],
                                           attention_mask=text_inputs.get("attention_mask"), return_dict=True)
        offset = bank.offset(prompt, use_e0=use_e0) if bank is not None else None
        pooled = text_out.pooler_output
        if offset is not None:
            pooled = pooled + offset.to(pooled.dtype).view(1, 1, -1)
        am = text_inputs.get("attention_mask")
        m = am.float().unsqueeze(-1) if am is not None else torch.ones_like(pooled[..., :1])
        text_vec = ((pooled.float() * m).sum(1) / m.sum(1).clamp(min=1))[0].cpu().numpy()
        text_out.pooler_output = pooled
        outputs = model(
            vision_embeds=vision_embeds,
            text_embeds=text_out,
            attention_mask=am,
        )
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=0.5,
            target_sizes=[(h, w)],
        )[0]
        masks = results.get("masks")
        scores = results.get("scores")
        if masks is None or len(masks) == 0:
            print(f"  [{prompt}] no instance")
            continue
        union = torch.zeros((h, w), dtype=torch.bool, device=masks.device)
        best = float(scores.max().item()) if scores is not None and len(scores) else 0.0
        for m in masks:
            union |= m.bool()
        area = int(union.sum().item())
        print(f"  [{prompt}] instances={len(masks)} best={best:.3f} pixels={area}")
        if area <= 0:
            continue
        parts.append({"prompt": prompt, "mask": union.cpu().numpy(), "score": best, "text_vec": text_vec})
    return parts


def segment_parts(processor, model, image: Image.Image, specs, threshold: float, device: str,
                  allow_missing: bool = False, bank=None, use_e0: bool = True):
    """One mask per concept; grouped concepts stay separate and remember their part name.

    SegviGen copies 2D colours onto 3D. If a grouped part is painted as one colour
    before that lift, disconnected pieces (a hand next to a staff, a boot under
    armour) get absorbed by the neighbour. Keep concept colours distinct here and
    merge back to the requested part name after the 3D labels exist.
    """
    unique = list(dict.fromkeys(
        prompt for _, prompts in specs for prompt in prompts
    ))
    found = {
        part["prompt"]: part
        for part in segment_prompts(processor, model, image, unique, threshold, device, bank=bank, use_e0=use_e0)
    }

    parts = []
    missing = []
    for name, prompts in specs:
        members = [found[prompt] for prompt in prompts if prompt in found]
        if not members:
            missing.append(name)
            continue
        if len(members) > 1:
            kept = "+".join(member["prompt"] for member in members)
            print(f"  [{name}] concepts {kept} kept separate for 3D lift")
        for member in members:
            parts.append({
                "prompt": member["prompt"],
                "part": name,
                "mask": member["mask"].astype(bool),
                "score": member["score"],
                "text_vec": member.get("text_vec"),
            })
    if missing:
        if allow_missing:
            # Probing mode (front-view selection): an undetected part simply scores 0
            # for this view instead of aborting the whole probe.
            print(f"  warning: no mask for component(s): {missing}")
        else:
            raise ValueError(f"SAM3 produced no mask for requested component(s): {missing}")
    return parts


def foreground_mask(image: Image.Image, alpha_threshold: int = 16) -> np.ndarray:
    """Silhouette of the rendered object; the render is RGBA on a transparent film."""
    if image.mode in ("RGBA", "LA"):
        return np.array(image.split()[-1]) > alpha_threshold
    return np.array(image.convert("RGB")).max(axis=2) > 8


def colorize(image: Image.Image, parts: list[dict], instance: bool,
             unassigned_to: str | None = None) -> tuple[Image.Image, list[dict]]:
    fg = foreground_mask(image)
    h, w = fg.shape
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    canvas[fg] = UNASSIGNED
    legend = []
    occupied = np.zeros((h, w), dtype=bool)

    # Paint small / specific parts first so "wall"/"house" cannot swallow the rest.
    # Colours are assigned up front so adjacent concepts (hand vs staff, base vs
    # armour) are not given two close pinks just because they were painted first.
    paintable = []
    for part in sorted(parts, key=lambda p: int(p["mask"].sum())):
        # Clipping to the silhouette matters: SAM3 masks bleed a few pixels past
        # the object, and colour outside the silhouette drags the attribute-guided
        # remesh into growing spikes out to the aabb faces.
        mask = part["mask"].astype(bool) & fg
        if instance:
            # already unioned per prompt; keep semantic coloring
            pass
        free = mask & ~occupied
        if free.sum() == 0:
            continue
        occupied |= free
        paintable.append((part, free))

    colors = pick_separated_colors(len(paintable))
    for (part, free), color in zip(paintable, colors):
        canvas[free] = color
        legend.append({
            "prompt": part["prompt"],
            "part": part.get("part", part["prompt"]),
            "color": list(color),
            "pixels": int(free.sum()),
            "score": part["score"],
            "text_vec": [round(float(x), 5) for x in part["text_vec"]] if part.get("text_vec") is not None else None,
        })

    leftover = fg & ~occupied
    if leftover.any() and unassigned_to:
        # Folding the leftovers into a named part keeps the part count equal to the number
        # of parts asked for; otherwise they survive as a separate grey part downstream.
        entry = next(
            (row for row in legend if row.get("part", row["prompt"]) == unassigned_to),
            None,
        )
        if entry is None:
            raise SystemExit(f"--unassigned_to {unassigned_to!r} is not one of the parts that got a mask")
        canvas[leftover] = entry["color"]
        entry["pixels"] += int(leftover.sum())
        occupied |= leftover
        leftover = fg & ~occupied

    unassigned = int(leftover.sum())
    if unassigned:
        legend.append({
            "prompt": "<unassigned>",
            "part": "<unassigned>",
            "color": list(UNASSIGNED),
            "pixels": unassigned,
            "score": None,
            "text_vec": None,
        })
    return Image.fromarray(canvas, mode="RGB"), legend


def main():
    parser = argparse.ArgumentParser(description="SAM3 -> SegviGen 2D part-color map")
    parser.add_argument("--image", required=True, help="Rendered conditioning view")
    parser.add_argument("--out", required=True, help="Output 2d_map.png")
    parser.add_argument("--prompts", nargs="+", required=True,
                        help="One entry per output part, e.g. 'roof door window'. Join several "
                             "concepts with '+' to merge them into one part, optionally naming "
                             "it: 'armor staff base body=head+face+hand+boot'.")
    parser.add_argument("--model", default="facebook/sam3")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--legend", default=None, help="Optional JSON path for color legend")
    parser.add_argument("--unassigned_to", default=None,
                        help="Name of the part that absorbs foreground pixels no prompt claimed, "
                             "instead of leaving them as a separate grey <unassigned> part.")
    parser.add_argument("--allow_missing", action="store_true",
                        help="Tolerate prompts with no detection (they are just absent from the "
                             "legend) instead of failing. Used when probing candidate views.")
    parser.add_argument("--concept_bank", default=None,
                        help="Stage-B bank.pt: adds learned offsets to the text features (SAM3 stays frozen)")
    parser.add_argument("--no_e0", action="store_true", help="With --concept_bank: per-name offsets only")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = Image.open(os.path.abspath(args.image))
    specs = normalize_part_specs(args.prompts)
    expected_names = part_names(specs)
    validate_target_name(args.unassigned_to, expected_names)
    print(f"SAM3 device={device} image={image.size} mode={image.mode}")
    print("parts: " + ", ".join(f"{name}({'+'.join(prompts)})" for name, prompts in specs))

    processor, model = load_sam3(args.model, device)
    bank = load_concept_bank(args.concept_bank, device)
    parts = segment_parts(processor, model, image, specs, args.threshold, device,
                          allow_missing=args.allow_missing, bank=bank, use_e0=not args.no_e0)
    if not parts and not args.allow_missing:
        raise SystemExit("SAM3 produced no masks. Try different --prompts or a lower --threshold.")

    colored, legend = colorize(image, parts, instance=False, unassigned_to=args.unassigned_to)
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    colored.save(out)
    legend_path = args.legend or os.path.splitext(out)[0] + "_legend.json"
    with open(legend_path, "w", encoding="utf-8") as f:
        json.dump(legend, f, ensure_ascii=False, indent=2)
    print(f"saved map {out}")
    print(f"saved legend {legend_path}")
    for row in legend:
        print(f"  {row['prompt']}: rgb={row['color']} pixels={row['pixels']}")

if __name__ == "__main__":
    main()
