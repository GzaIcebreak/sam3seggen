"""Turn a rendered view into a SegviGen 2D part-color map using SAM3.

Run with .venv_holo (transformers 5.x). Do not mix with the SegviGen venv.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

PALETTE = [
    (255, 80, 160),
    (80, 200, 180),
    (160, 80, 220),
    (255, 220, 60),
    (40, 140, 80),
    (220, 120, 40),
    (80, 140, 255),
    (200, 60, 60),
    (120, 220, 80),
    (90, 90, 200),
    (255, 160, 200),
    (40, 180, 220),
]

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


@torch.no_grad()
def segment_prompts(processor, model, image: Image.Image, prompts: list[str], threshold: float, device: str):
    w, h = image.size
    img_inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    pixel_values = img_inputs["pixel_values"].to(device)
    vision_embeds = model.get_vision_features(pixel_values=pixel_values)

    parts = []
    for prompt in prompts:
        text_inputs = _to_device(processor(text=prompt, return_tensors="pt"), device)
        outputs = model(
            vision_embeds=vision_embeds,
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs.get("attention_mask"),
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
        parts.append({"prompt": prompt, "mask": union.cpu().numpy(), "score": best})
    return parts


def foreground_mask(image: Image.Image, alpha_threshold: int = 16) -> np.ndarray:
    """Silhouette of the rendered object; the render is RGBA on a transparent film."""
    if image.mode in ("RGBA", "LA"):
        return np.array(image.split()[-1]) > alpha_threshold
    return np.array(image.convert("RGB")).max(axis=2) > 8


def colorize(image: Image.Image, parts: list[dict], instance: bool) -> tuple[Image.Image, list[dict]]:
    fg = foreground_mask(image)
    h, w = fg.shape
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    canvas[fg] = UNASSIGNED
    legend = []
    occupied = np.zeros((h, w), dtype=bool)

    color_i = 0
    # Paint small / specific parts first so "wall"/"house" cannot swallow the rest.
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
        color = PALETTE[color_i % len(PALETTE)]
        canvas[free] = color
        occupied |= free
        legend.append({
            "prompt": part["prompt"],
            "color": list(color),
            "pixels": int(free.sum()),
            "score": part["score"],
        })
        color_i += 1

    unassigned = int((fg & ~occupied).sum())
    if unassigned:
        legend.append({
            "prompt": "<unassigned>",
            "color": list(UNASSIGNED),
            "pixels": unassigned,
            "score": None,
        })
    return Image.fromarray(canvas, mode="RGB"), legend


def main():
    parser = argparse.ArgumentParser(description="SAM3 -> SegviGen 2D part-color map")
    parser.add_argument("--image", required=True, help="Rendered conditioning view")
    parser.add_argument("--out", required=True, help="Output 2d_map.png")
    parser.add_argument("--prompts", nargs="+", required=True, help="SAM3 text concepts, e.g. roof door window")
    parser.add_argument("--model", default="facebook/sam3")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--legend", default=None, help="Optional JSON path for color legend")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = Image.open(os.path.abspath(args.image))
    print(f"SAM3 device={device} image={image.size} mode={image.mode} prompts={args.prompts}")

    processor, model = load_sam3(args.model, device)
    parts = segment_prompts(processor, model, image, args.prompts, args.threshold, device)
    if not parts:
        raise SystemExit("SAM3 produced no masks. Try different --prompts or a lower --threshold.")

    colored, legend = colorize(image, parts, instance=False)
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
