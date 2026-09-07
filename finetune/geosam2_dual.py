"""GeoSAM2 with SAM3 label maps from TWO opposite views as prompts (no automatic segmentation).

    E:\...\.venv_geosam2\Scripts\python finetune\geosam2_dual.py --renders E:\...\geosam2\renders
        --out E:\...\geosam2\results --objects_file E:\...\pv_hard.txt --dataset_root E:\...\pv --view match

GeoSAM2's CLI takes one prompt view and fills the rest of the object with SAM2 automatic masks from the
opposite view; with name prompts that auto pass is what leaves parts unnamed (and its area filter drops
most masks when the anchor is a side view). Here both views v and v+6 get SAM3 masks, each part keeps
one id across the two passes (id = label-map value = prompt index + 1, ids of the second pass are
offset by 1000 and merged back by name afterwards), and faces GeoSAM2 leaves unlabelled (0 / 999) are
filled from the nearest labelled face centroid. Writes <out>/<obj>/dual_<tag>/
    dual.npy         merged face labels (0 = still unlabelled), dual_filled.npy after the fill
    dual.glb         the mesh GeoSAM2 exported (its normalised frame; face order matches the npy)
    labels.json      {"ids": {"1": "helmet", ...}, "views": [v, v2], ...}
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

GEOSAM2_ROOT = os.environ.get("GEOSAM2_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "GeoSAM2"))
sys.path.insert(0, GEOSAM2_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.chdir(GEOSAM2_ROOT)

import torch  # noqa: E402
import trimesh  # noqa: E402

import inference as gi  # noqa: E402  GeoSAM2's inference.py
from sam2.automatic_mask_generator_geosam2 import SAM2AutomaticMaskGenerator  # noqa: E402
from sam2.build_sam import build_sam2, build_sam2_video_predictor_geosam2  # noqa: E402
from geosam2_run import MIN_AREA_PX, fill_unlabelled, match_view  # noqa: E402

OFFSET = 1000


def label_masks(label_map: np.ndarray, offset: int = 0) -> dict[int, np.ndarray]:
    out = {}
    for k in np.unique(label_map):
        if k <= 0:
            continue
        m = label_map == k
        if m.sum() >= MIN_AREA_PX:
            out[int(k) + offset] = m
    return out


def run_object(predictor, mask_generator, rdir: str, out_dir: str, views: tuple[int, int], pa: float) -> dict:
    with open(os.path.join(rdir, "sam3", "summary.json"), "r", encoding="utf-8") as f:
        summary = json.load(f)
    prompts = summary["prompts"]
    data = gi.read_data(rdir.replace("\\", "/"))
    v1, v2 = views
    m1 = label_masks(np.load(os.path.join(rdir, "sam3", f"view_{v1:04d}.npy")))
    m2 = label_masks(np.load(os.path.join(rdir, "sam3", f"view_{v2:04d}.npy")), OFFSET)
    data["prompt_masks"] = {v1: m1, v2: m2}
    data["gt_masks"] = data["prompt_masks"]
    starts = [v for v, m in ((v1, m1), (v2, m2)) if m]
    if not starts:
        return {"error": "no SAM3 masks in either view"}
    work = os.path.join(out_dir, "raw")
    os.makedirs(work, exist_ok=True)
    t0 = time.time()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        res = gi.segment_with_mask_prompts(
            predictor=predictor, mask_generator=mask_generator, data=data,
            opposite_auto_segmentation=False, enable_postprocess=True, postprocess_pa=pa,
            output_dir=work.replace("\\", "/"), start_frames=starts,
            start_to_seed_views={s: [s] for s in starts})
    last = starts[-1]
    tag = f"segmentation_postprocessed_promptView{last:02d}_pa{pa:g}"
    lab = np.rint(np.load(os.path.join(work, tag + ".npy"))).astype(np.int64)
    # merge the two passes by name: id k and k+OFFSET are the same prompt
    merged = np.where((lab >= OFFSET) & (lab < 2 * OFFSET), lab - OFFSET, lab)
    ids = {str(k): prompts[k - 1] for k in sorted({int(x) for x in np.unique(merged) if 0 < x <= len(prompts)})}
    shutil.copy(os.path.join(work, tag + ".glb"), os.path.join(out_dir, "dual.glb"))
    np.save(os.path.join(out_dir, "dual.npy"), merged.astype(np.float32))
    mesh = res["mesh"]
    filled = fill_unlabelled(mesh, merged, {0, 999})
    np.save(os.path.join(out_dir, "dual_filled.npy"), filled.astype(np.float32))
    info = {"views": list(views), "starts": starts, "ids": ids, "pa": pa, "seconds": round(time.time() - t0, 1),
            "unlabelled_faces_raw": int(np.isin(merged, [0, 999]).sum()), "n_faces": int(len(merged)),
            "prompted_view1": len(m1), "prompted_view2": len(m2)}
    with open(os.path.join(out_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset_root", default=None)
    ap.add_argument("--objects_file", default=None)
    ap.add_argument("--objects", nargs="*", default=None)
    ap.add_argument("--view", default="match", help="best | match | <view id>; the second view is +6")
    ap.add_argument("--pa", type=float, default=0.02)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    objects = list(args.objects or [])
    if args.objects_file:
        with open(args.objects_file, "r", encoding="utf-8") as f:
            objects += [l.strip() for l in f if l.strip()]
    tag = args.view if args.view in ("best", "match") else f"view{int(args.view):02d}"

    device = gi.init_env()
    ckpt, cfg = "ckpt/geosam2.pt", "configs/geosam2.yaml"
    sam2 = build_sam2(cfg, ckpt, device=device, apply_postprocessing=False)
    predictor = build_sam2_video_predictor_geosam2(cfg, ckpt, device=device)
    mask_generator = SAM2AutomaticMaskGenerator(model=sam2, points_per_side=64, points_per_batch=128,
                                                pred_iou_thresh=0.7, stability_score_thresh=0.7,
                                                stability_score_offset=0.7, crop_n_layers=0, box_nms_thresh=0.7,
                                                crop_n_points_downscale_factor=2, min_mask_region_area=25.0,
                                                use_m2m=True)
    for obj in objects:
        rdir = os.path.join(args.renders, obj)
        out_dir = os.path.join(args.out, obj, f"dual_{tag}")
        if os.path.exists(os.path.join(out_dir, "dual_filled.npy")) and not args.force:
            print(f"[skip] {obj} dual_{tag}", flush=True)
            continue
        if args.view == "best":
            with open(os.path.join(rdir, "sam3", "summary.json"), "r", encoding="utf-8") as f:
                v = int(json.load(f)["best_view"])
        elif args.view == "match":
            v, iou = match_view(rdir, os.path.join(args.dataset_root, obj, "views", "az0", "render.png"))
        else:
            v = int(args.view)
        try:
            info = run_object(predictor, mask_generator, rdir, out_dir, (v, (v + 6) % 12), args.pa)
        except Exception as e:  # keep the batch going; the object is reported as failed
            info = {"error": f"{type(e).__name__}: {e}"}
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "labels.json"), "w", encoding="utf-8") as f:
                json.dump(info, f, indent=2)
        status = "FAIL" if "error" in info else "ok"
        print(f"[{status}] {obj} dual_{tag} views={v},{(v + 6) % 12} " +
              (info.get("error", "") if status == "FAIL" else
               f"prompted={info['prompted_view1']}+{info['prompted_view2']} unlabelled_raw="
               f"{info['unlabelled_faces_raw']}/{info['n_faces']} {info['seconds']}s"), flush=True)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
