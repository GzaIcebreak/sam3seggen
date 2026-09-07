"""SAM3 tracker as the multi-view propagator (the "swap SAM2 for SAM3" ablation).

GeoSAM2 = SAM2 (Hiera) video predictor + LoRA + residual depth/normal fusion, trained on 3D parts. Its
weights are tied to SAM2's feature layout, so SAM3 cannot be dropped into GeoSAM2's LoRA/fusion modules
without retraining. What CAN be swapped without training is the propagator itself: SAM3 ships a SAM2-style
tracker (memory attention + mask decoder on the Perception-Encoder backbone, `Sam3TrackerVideoModel`).
This script feeds the same SAM3 label maps GeoSAM2 gets as prompts into that tracker, propagates them
around the 12-view turntable (RGB only, no geometry) and writes one label map per view. The lifting to
the mesh is then done by `geosam2_ablate.py --mode lift`, identical to the other ablation rows.

    .venv_holo\Scripts\python finetune\sam3_track.py --renders E:\...\geosam2\renders --out E:\...\geosam2\results
        --objects <id> ... --view match --ref_render_pattern "E:\...\pv\{obj}\views\az0\render.png" --n_prompts 1

Prompt views: n_prompts views spaced 12/n apart starting at the anchor (match/best/<int>). Propagation
runs in two half-turn sessions (anchor -> anchor+6 forward, anchor -> anchor-6 backward) so no view is
more than 180 deg of memory away from the anchor; a frame present in both sessions averages the logits.
Prompted views keep their SAM3 masks verbatim (same convention as GeoSAM2's conditioning frames).

Output <out>/<obj>/sam3track_p<n>_<tag>/
    masks/view_XXXX.npy   int32 label map per view, value = SAM3 prompt index + 1, 0 = no part
    info.json             ids -> names, prompt views, seconds
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geosam2_run import MIN_AREA_PX, match_view  # noqa: E402

NUM_VIEWS = 12


def load_tracker(device: str):
    from huggingface_hub import snapshot_download
    from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
    local = snapshot_download("facebook/sam3", local_files_only=True)
    proc = Sam3TrackerVideoProcessor.from_pretrained(local)
    model = Sam3TrackerVideoModel.from_pretrained(local, dtype=torch.bfloat16).to(device).eval()
    return proc, model


def load_frames(rdir: str) -> list[np.ndarray]:
    frames = []
    for v in range(NUM_VIEWS):
        im = Image.open(os.path.join(rdir, f"color_{v:04d}.webp")).convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        frames.append(np.asarray(Image.alpha_composite(bg, im).convert("RGB")))
    return frames


def prompt_views(anchor: int, n: int) -> list[int]:
    step = NUM_VIEWS // n
    return [(anchor + i * step) % NUM_VIEWS for i in range(n)]


def resolve_anchor(rdir: str, view: str, ref_pattern: str | None, obj: str) -> int:
    if view == "best":
        with open(os.path.join(rdir, "sam3", "summary.json"), "r", encoding="utf-8") as f:
            return int(json.load(f)["best_view"])
    if view == "match":
        if not ref_pattern:
            raise SystemExit("--view match needs --ref_render_pattern")
        return match_view(rdir, ref_pattern.format(obj=obj))[0]
    return int(view)


def propagate(proc, model, frames: list[np.ndarray], maps: dict[int, np.ndarray], anchor: int, device: str):
    """maps: prompted view -> label map. Returns per-view summed logits [n_ids, H, W] and the id list."""
    ids = sorted({int(k) for m in maps.values() for k in np.unique(m) if k > 0 and (m == k).sum() >= MIN_AREA_PX})
    H, W = frames[0].shape[:2]
    acc = {v: np.zeros((len(ids), H, W), np.float32) for v in range(NUM_VIEWS)}
    cnt = {v: 0 for v in range(NUM_VIEWS)}
    for direction in (1, -1):
        order = [(anchor + direction * i) % NUM_VIEWS for i in range(NUM_VIEWS // 2 + 1)]
        sess = proc.init_video_session(video=[frames[v] for v in order], inference_device=device, dtype=torch.bfloat16)
        prompted = False
        for fi, v in enumerate(order):
            if v not in maps:
                continue
            present = [k for k in ids if (maps[v] == k).sum() >= MIN_AREA_PX]
            if not present:
                continue
            proc.add_inputs_to_inference_session(sess, frame_idx=fi, obj_ids=present,
                                                 input_masks=[maps[v] == k for k in present])
            with torch.inference_mode():  # encode the conditioning frame (memory) before propagating
                model(inference_session=sess, frame_idx=fi)
            prompted = True
        if not prompted:
            continue
        # forward from the anchor and backward from the far end: an object prompted only on a later view
        # has no memory on the frames before it in a single forward pass
        passes = [dict(start_frame_idx=0, reverse=False)]
        if len(maps) > 1:
            passes.append(dict(start_frame_idx=len(order) - 1, reverse=True))
        with torch.inference_mode():
            for kw in passes:
                for out in model.propagate_in_video_iterator(sess, **kw):
                    logits = proc.post_process_masks([out.pred_masks], original_sizes=[(H, W)], binarize=False)[0]
                    logits = logits[:, 0].float().cpu().numpy()  # [n_obj_in_session, H, W]
                    v = order[out.frame_idx]
                    for row, oid in enumerate(sess.obj_ids):
                        acc[v][ids.index(int(oid))] += logits[row]
                    cnt[v] += 1
    for v in acc:
        if cnt[v] > 1:
            acc[v] /= cnt[v]
    return acc, ids


def to_label_maps(acc, ids, maps) -> dict[int, np.ndarray]:
    out = {}
    for v, logit in acc.items():
        if v in maps:  # prompted view: SAM3 mask verbatim
            lab = maps[v].astype(np.int32)
            lab[~np.isin(lab, ids)] = 0
            out[v] = lab
            continue
        lab = np.zeros(logit.shape[1:], np.int32)
        if len(ids):
            best = logit.argmax(0)
            pos = logit.max(0) > 0
            lab[pos] = np.asarray(ids, np.int32)[best[pos]]
        out[v] = lab
    return out


def run_object(proc, model, rdir: str, out_dir: str, anchor: int, n: int, device: str) -> dict:
    with open(os.path.join(rdir, "sam3", "summary.json"), "r", encoding="utf-8") as f:
        prompts = json.load(f)["prompts"]
    views = prompt_views(anchor, n)
    maps = {v: np.load(os.path.join(rdir, "sam3", f"view_{v:04d}.npy")) for v in views}
    frames = load_frames(rdir)
    t0 = time.time()
    acc, ids = propagate(proc, model, frames, maps, anchor, device)
    labs = to_label_maps(acc, ids, maps)
    os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)
    for v, lab in labs.items():
        np.save(os.path.join(out_dir, "masks", f"view_{v:04d}.npy"), lab)
    fg = [float((lab > 0).mean()) for lab in labs.values()]
    info = {"anchor": anchor, "prompt_views": views, "ids": {str(k): prompts[k - 1] for k in ids},
            "seconds": round(time.time() - t0, 1), "labelled_share_per_view": [round(x, 4) for x in fg],
            "propagator": "facebook/sam3 Sam3TrackerVideoModel (RGB only, two half-turn sessions)"}
    with open(os.path.join(out_dir, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--objects", nargs="*", default=None)
    ap.add_argument("--objects_file", default=None)
    ap.add_argument("--view", default="match", help="match | best | <int>")
    ap.add_argument("--ref_render_pattern", default=None, help="for --view match, e.g. E:\\...\\pv\\{obj}\\views\\az0\\render.png")
    ap.add_argument("--n_prompts", type=int, nargs="+", default=[1])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    objects = list(args.objects or [])
    if args.objects_file:
        with open(args.objects_file, "r", encoding="utf-8") as f:
            objects += [l.strip() for l in f if l.strip()]
    tag = args.view if args.view in ("best", "match") else f"view{int(args.view):02d}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc, model = load_tracker(device)
    for obj in objects:
        rdir = os.path.join(args.renders, obj)
        anchor = resolve_anchor(rdir, args.view, args.ref_render_pattern, obj)
        for n in args.n_prompts:
            out_dir = os.path.join(args.out, obj, f"sam3track_p{n}_{tag}")
            if os.path.exists(os.path.join(out_dir, "info.json")) and not args.force:
                print(f"[skip] {obj} sam3track_p{n}_{tag}", flush=True)
                continue
            try:
                info = run_object(proc, model, rdir, out_dir, anchor, n, device)
                print(f"[ok] {obj} sam3track_p{n}_{tag} anchor={anchor} views={info['prompt_views']} "
                      f"ids={len(info['ids'])} {info['seconds']}s", flush=True)
            except Exception as e:
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, "info.json"), "w", encoding="utf-8") as f:
                    json.dump({"error": f"{type(e).__name__}: {e}"}, f, indent=2)
                print(f"[FAIL] {obj} sam3track_p{n}_{tag}: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
