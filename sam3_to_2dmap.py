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

DEFAULT_SAM3 = os.environ.get(
    "SEGVIGEN_SAM3",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "facebook", "sam3"),
)

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

    if os.path.isdir(model_id):
        local = model_id
    else:
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


def _finetune_on_path():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetune"))


def load_concept_bank(path: str | None, device: str):
    """Optional Stage-B bank (finetune/concept_bank.py); None when no path is given."""
    if not path:
        return None
    _finetune_on_path()
    import sam3_bank
    bank = sam3_bank.ConceptBank.load(path, device)
    print(f"concept bank: {bank.describe()}, template {bank.template!r}")
    return bank


def attach_decoder_lora(model, bank, bank_path: str | None, lora_file: str | None, device: str) -> bool:
    """v5 banks carry `decoder_lora` / `lora_scope` in meta; the weights live next to bank.pt. Without them
    the bank's text offsets sit on a decoder they were not trained with, so this is not optional."""
    meta = getattr(bank, "meta", None) or {}
    r, scope = int(meta.get("decoder_lora", 0) or 0), meta.get("lora_scope", "")
    if not lora_file:
        if r <= 0:
            return False
        stem = os.path.basename(bank_path or "").replace("bank", "decoder_lora")
        cand = os.path.join(os.path.dirname(bank_path or ""), stem)
        if not os.path.exists(cand):
            raise SystemExit(f"bank asks for decoder LoRA r={r} ({scope}) but {cand} is missing")
        lora_file = cand
    _finetune_on_path()
    import sam3_bank
    from lora import load_lora_state_dict
    _, n = sam3_bank.inject_decoder_lora(model, r=r or 8, scope=scope or "mask,text")
    load_lora_state_dict(model, torch.load(lora_file, map_location=device, weights_only=False))
    print(f"decoder LoRA: r={r} scope={scope} {n} layers from {lora_file}")
    return True


@torch.no_grad()
def segment_prompts_argmax(processor, model, image: Image.Image, prompts: list[str], device: str,
                           bank=None, use_e0: bool = True, tau: float = 0.5, chunk: int = 12):
    """v5 deployment operator (PLAN_concept_bank_v5.md section 3): one softmax per pixel over all prompts
    plus the bank's learned background logit, then argmax. Unlike the score-threshold overlay the prompts
    compete, which is the same objective the bank was trained with, so `body` no longer wins `arm`'s pixels
    just by being detected. The returned masks are disjoint by construction, so `colorize` paints them
    unchanged and the legend downstream is identical."""
    fields = argmax_fields(processor, model, image, prompts, device, bank, use_e0, chunk)
    return argmax_parts(fields, prompts, tau)


@torch.no_grad()
def argmax_fields(processor, model, image: Image.Image, prompts: list[str], device: str,
                  bank=None, use_e0: bool = True, chunk: int = 12,
                  gate_rel: float = 0.0, gate_topk: int = 0) -> dict:
    """The expensive half of the operator above: one image encode plus one forward per prompt chunk.
    Everything a tau needs afterwards is in the returned dict, so a tau sweep costs one forward pass.
    `gate_rel` / `gate_topk` drop each prompt's weak queries before they can win pixels (sam3_bank.query_gate)."""
    _finetune_on_path()
    import sam3_bank as sb
    vis = sb.encode_image(processor, model, image, device)
    maps, vecs, scores = [], [], []
    for k in range(0, len(prompts), chunk):
        part = prompts[k:k + chunk]
        tf, am = sb.text_features_batch(processor, model, part, device)
        qv = sb.text_query_vecs(tf, am)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            out, bias = sb.bank_forward(model, vis, tf, am, bank, part, qv, use_e0)
        s = sb.batch_scores(out, bias).float()
        maps.append(sb.name_logit_maps(out, bias, sb.query_gate(s, gate_rel, gate_topk)))
        scores.append(s.max(dim=1).values)
        offs = bank.offsets(part, qv, use_e0).float() if bank is not None else torch.zeros_like(qv)
        vecs.append((qv.float() + offs).cpu().numpy())
    return {
        "maps": torch.cat(maps, 0),
        "scores": torch.cat(scores).tolist(),
        "vecs": np.concatenate(vecs, 0),
        "fg": torch.from_numpy(foreground_mask(image)).to(device),
        "bg_logit": bank.bg_logit() if bank is not None else 0.0,
        "temp": float((getattr(bank, "meta", None) or {}).get("assign_temp", 1.0) or 1.0),
    }


def argmax_parts(fields: dict, prompts: list[str], tau: float, min_comp: float = 0.0) -> list[dict]:
    """The cheap half: turn the per-prompt pixel logits into disjoint masks at one confidence floor.
    `min_comp` re-labels fragments under that share of the silhouette (sam3_bank.clean_components)."""
    _finetune_on_path()
    import sam3_bank as sb
    lab = sb.paint_argmax(fields["maps"], fields["fg"], fields["bg_logit"], tau, fields["temp"])
    if min_comp > 0:
        lab = sb.clean_components(lab, fields["fg"], min_comp)
    parts = []
    for i, prompt in enumerate(prompts):
        mask = (lab == i).cpu().numpy()
        area = int(mask.sum())
        print(f"  [{prompt}] tau={tau:g} argmax pixels={area} best_score={fields['scores'][i]:.3f}")
        if area <= 0:
            continue
        parts.append({"prompt": prompt, "mask": mask, "score": fields["scores"][i], "text_vec": fields["vecs"][i]})
    return parts


@torch.no_grad()
def rank_fields(processor, model, image: Image.Image, prompts: list[str], device: str, rank_model: str,
                bank=None, use_e0: bool = True, chunk: int = 12) -> dict:
    """Mask RankGNN deployment (REPORT_mask_rank_v3.md): one forward pass, then the ranker weighs each
    prompt's top-K queries against each other. Returns the candidate dict sam3_bank's mask-level painters
    consume. The feature path mirrors finetune/mask_rank_feats.py so training and deployment cannot drift."""
    _finetune_on_path()
    import sam3_bank as sb
    from mask_rank import candidate_features, load_model, select_topk
    ranker, cfg = load_model(rank_model, device)
    vis = sb.encode_image(processor, model, image, device)
    fpn = vis.fpn_hidden_states[0][0]
    use_qv = bank is not None and bank.nn_cos > 0 and bank.tvec is not None
    logits, scores, vecs = [], [], []
    for k in range(0, len(prompts), chunk):
        part = prompts[k:k + chunk]
        tf, am = sb.text_features_batch(processor, model, part, device)
        qv = sb.text_query_vecs(tf, am)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            out, bias = sb.bank_forward(model, vis, tf, am, bank, part, qv if use_qv else None, use_e0)
        logits.append(out.pred_masks.clone())
        scores.append(sb.batch_scores(out, bias).float())
        offs = (bank.offsets(part, qv if use_qv else None, use_e0).float() if bank is not None
                else torch.zeros_like(qv.float()))
        vecs.append((qv.float() + offs).detach())
        del out
    logits, scores, tvecs = torch.cat(logits, 0), torch.cat(scores, 0), torch.cat(vecs, 0)
    fg = torch.from_numpy(foreground_mask(image)).to(device)
    h, w = logits.shape[-2:]
    fg_low = torch.nn.functional.interpolate(fg.float()[None, None], size=(h, w), mode="nearest")[0, 0] > 0.5
    idx = select_topk(scores, int(cfg["topk"]))
    f = candidate_features(logits, scores, fpn, fg_low, tvecs, idx)
    keep = ranker(f["node"], f["edge"]).view(idx.shape)
    return {"logits": logits, "scores": scores, "idx": idx, "keep": keep, "fg": fg,
            "best": scores.max(dim=1).values.tolist(), "vecs": tvecs.cpu().numpy()}


def rank_parts(fields: dict, prompts: list[str], score_k: float, drop: float, add: float,
               order: str = "small", min_comp: float = 0.0) -> list[dict]:
    """v3's overlay set edited by the ranker (sam3_bank.paint_hybrid), then the same disjoint part list
    argmax_parts returns. drop = 0 and add > 1 reproduce v3 exactly."""
    _finetune_on_path()
    import sam3_bank as sb
    lab = sb.paint_hybrid(fields, score_k, drop, add, order)
    if min_comp > 0:
        lab = sb.clean_components(lab, fields["fg"], min_comp)
    sc = fields["scores"].gather(1, fields["idx"])
    parts = []
    for i, prompt in enumerate(prompts):
        mask = (lab == i).cpu().numpy()
        area = int(mask.sum())
        n_v3 = int((sc[i] >= score_k).sum())
        n_drop = int(((sc[i] >= score_k) & (fields["keep"][i] < drop)).sum())
        n_add = int(((sc[i] < score_k) & (fields["keep"][i] >= add)).sum())
        print(f"  [{prompt}] rank pixels={area} best_score={fields['best'][i]:.3f} "
              f"v3_masks={n_v3} dropped={n_drop} added={n_add}")
        if area <= 0:
            continue
        parts.append({"prompt": prompt, "mask": mask, "score": fields["best"][i], "text_vec": fields["vecs"][i]})
    return parts


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
        am = text_inputs.get("attention_mask")
        pooled = text_out.pooler_output
        m = am.float().unsqueeze(-1) if am is not None else torch.ones_like(pooled[..., :1])
        qv = (pooled.float() * m).sum(1) / m.sum(1).clamp(min=1)                  # [1, D]
        bias = None
        if bank is not None:
            # legend token = mean-pooled features + the prompt's offset (context tokens are not part of it)
            text_vec = (qv[0] + bank.offset(prompt, use_e0=use_e0, query_vec=qv[0]).float()).cpu().numpy()
            text_out, am, bias = bank.apply(text_out, am, [prompt], qv, use_e0)
        else:
            text_vec = qv[0].cpu().numpy()
        outputs = model(
            vision_embeds=vision_embeds,
            text_embeds=text_out,
            attention_mask=am,
        )
        if bias is not None:
            outputs.pred_logits = outputs.pred_logits + bias.to(outputs.pred_logits.dtype).view(-1, 1)
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


def segment_parts_sweep(processor, model, image: Image.Image, specs, threshold: float, device: str,
                        allow_missing: bool = False, bank=None, use_e0: bool = True,
                        assign: str = "paint", taus=(0.5,),
                        gate_rel: float = 0.0, gate_topk: int = 0, min_comp: float = 0.0,
                        rank: dict | None = None) -> list[list[dict]]:
    """`segment_parts` for a whole confidence sweep: one part list per tau, sharing a single forward pass.
    `paint` and `rank` have no tau, so they return a single list however many were asked for.
    `rank` = {model, drop, add, order} for --assign rank."""
    unique = list(dict.fromkeys(prompt for _, prompts in specs for prompt in prompts))
    if assign == "rank":
        fields = rank_fields(processor, model, image, unique, device, rank["model"], bank=bank, use_e0=use_e0)
        return [group_parts(rank_parts(fields, unique, threshold, rank["drop"], rank["add"], rank["order"],
                                       min_comp), specs, allow_missing)]
    if assign != "argmax":
        got = segment_prompts(processor, model, image, unique, threshold, device, bank=bank, use_e0=use_e0)
        return [group_parts(got, specs, allow_missing)]
    fields = argmax_fields(processor, model, image, unique, device, bank=bank, use_e0=use_e0,
                           gate_rel=gate_rel, gate_topk=gate_topk)
    return [group_parts(argmax_parts(fields, unique, tau, min_comp), specs, allow_missing) for tau in taus]


def segment_parts(processor, model, image: Image.Image, specs, threshold: float, device: str,
                  allow_missing: bool = False, bank=None, use_e0: bool = True,
                  assign: str = "paint", tau: float = 0.5):
    """One mask per concept; grouped concepts stay separate and remember their part name."""
    return segment_parts_sweep(processor, model, image, specs, threshold, device, allow_missing, bank, use_e0,
                               assign, (tau,))[0]


def group_parts(got: list[dict], specs, allow_missing: bool = False) -> list[dict]:
    """Attach the requested part name to each concept mask, keeping grouped concepts separate.

    SegviGen copies 2D colours onto 3D. If a grouped part is painted as one colour
    before that lift, disconnected pieces (a hand next to a staff, a boot under
    armour) get absorbed by the neighbour. Keep concept colours distinct here and
    merge back to the requested part name after the 3D labels exist.
    """
    found = {part["prompt"]: part for part in got}

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
    parser.add_argument("--out", required=True, nargs="+",
                        help="Output 2d_map.png; with several --tau, one path per tau in the same order")
    parser.add_argument("--prompts", nargs="+", required=True,
                        help="One entry per output part, e.g. 'roof door window'. Join several "
                             "concepts with '+' to merge them into one part, optionally naming "
                             "it: 'armor staff base body=head+face+hand+boot'.")
    parser.add_argument("--model", default=DEFAULT_SAM3)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--legend", default=None, nargs="+", help="Optional JSON path for color legend, one per --out")
    parser.add_argument("--unassigned_to", default=None,
                        help="Name of the part that absorbs foreground pixels no prompt claimed, "
                             "instead of leaving them as a separate grey <unassigned> part.")
    parser.add_argument("--allow_missing", action="store_true",
                        help="Tolerate prompts with no detection (they are just absent from the "
                             "legend) instead of failing. Used when probing candidate views.")
    parser.add_argument("--concept_bank", default=None,
                        help="Stage-B bank.pt: adds learned offsets to the text features (SAM3 stays frozen)")
    parser.add_argument("--no_e0", action="store_true", help="With --concept_bank: per-name offsets only")
    parser.add_argument("--assign", choices=["paint", "argmax", "rank"], default="paint",
                        help="paint = score threshold then smallest-mask-first overlay (v3); "
                             "argmax = per-pixel softmax over the prompts + learned background (v5); "
                             "rank = v3's set edited by the Mask RankGNN (--rank_model), same overlay")
    parser.add_argument("--rank_model", default=None, help="--assign rank: mask_rank.py checkpoint")
    parser.add_argument("--rank_drop", type=float, default=0.1,
                        help="--assign rank: drop a v3 mask whose ranker weight is below this")
    parser.add_argument("--rank_add", type=float, default=0.9,
                        help="--assign rank: add a mask v3 skipped whose ranker weight is at least this")
    parser.add_argument("--rank_order", choices=["small", "keep"], default="small",
                        help="--assign rank: overlay order, smallest union first (v3) or by ranker weight")
    parser.add_argument("--tau", type=float, nargs="+", default=[0.5],
                        help="--assign argmax: a pixel stays grey unless its winner's probability exceeds this. "
                             "Several values sweep the threshold off one forward pass and need as many --out paths")
    parser.add_argument("--gate_rel", type=float, default=0.0,
                        help="--assign argmax: a prompt's queries scoring under this x its best sit out the competition")
    parser.add_argument("--gate_topk", type=int, default=0, help="--assign argmax: keep only each prompt's k best queries")
    parser.add_argument("--min_comp", type=float, default=0.0,
                        help="--assign argmax: re-label painted fragments smaller than this share of the silhouette")
    parser.add_argument("--decoder_lora_file", default=None,
                        help="decoder_lora_*.pt for a v5 bank; found next to bank.pt when omitted")
    args = parser.parse_args()

    taus = args.tau if args.assign == "argmax" else args.tau[:1]
    if args.assign == "rank" and not args.rank_model:
        raise SystemExit("--assign rank needs --rank_model")
    if len(args.out) != len(taus):
        raise SystemExit(f"got {len(args.out)} --out paths for {len(taus)} threshold(s)")
    if args.legend and len(args.legend) != len(args.out):
        raise SystemExit(f"got {len(args.legend)} --legend paths for {len(args.out)} --out paths")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = Image.open(os.path.abspath(args.image))
    specs = normalize_part_specs(args.prompts)
    expected_names = part_names(specs)
    validate_target_name(args.unassigned_to, expected_names)
    print(f"SAM3 device={device} image={image.size} mode={image.mode}")
    print("parts: " + ", ".join(f"{name}({'+'.join(prompts)})" for name, prompts in specs))

    processor, model = load_sam3(args.model, device)
    bank = load_concept_bank(args.concept_bank, device)
    if bank is not None:
        attach_decoder_lora(model, bank, args.concept_bank, args.decoder_lora_file, device)
    sweep = segment_parts_sweep(processor, model, image, specs, args.threshold, device,
                                allow_missing=args.allow_missing, bank=bank, use_e0=not args.no_e0,
                                assign=args.assign, taus=taus,
                                gate_rel=args.gate_rel, gate_topk=args.gate_topk, min_comp=args.min_comp,
                                rank={"model": args.rank_model, "drop": args.rank_drop, "add": args.rank_add,
                                      "order": args.rank_order})
    for i, parts in enumerate(sweep):
        if not parts and not args.allow_missing:
            raise SystemExit("SAM3 produced no masks. Try different --prompts or a lower --threshold.")
        colored, legend = colorize(image, parts, instance=False, unassigned_to=args.unassigned_to)
        out = os.path.abspath(args.out[i])
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        colored.save(out)
        legend_path = args.legend[i] if args.legend else os.path.splitext(out)[0] + "_legend.json"
        with open(legend_path, "w", encoding="utf-8") as f:
            json.dump(legend, f, ensure_ascii=False, indent=2)
        print(f"saved map {out}")
        print(f"saved legend {legend_path}")
        for row in legend:
            print(f"  {row['prompt']}: rgb={row['color']} pixels={row['pixels']}")

if __name__ == "__main__":
    main()
