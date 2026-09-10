"""Lift ext_bench 2D maps through SegviGen and write upright GLBs.

Loads the DiT once. Voxelises each asset once. Samples every pending (asset, tag) pair.
Same export path as inference_full.py --two_d_map, then the same upright() as ext_bench.

    python finetune/lift_maps_segvigen.py --keys dog --tags v5
    python finetune/lift_maps_segvigen.py --tags ease_v6 --ckpt <v6> --seeds 0,1,2,3,4
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("SEGVIGEN_DINOV3", os.path.join(ROOT, "weights", "facebook", "dinov3-vitl16-pretrain-lvd1689m"))
os.environ.setdefault("SEGVIGEN_RMBG", os.path.join(ROOT, "weights", "ZhengPeng7", "BiRefNet"))

from ext_bench import ASSETS, CKPT, OUT, PY, TRANSFORMS, front_json, map_paths, run, src_glb, upright, wd  # noqa: E402

TAG_MAP = {
    "v5": "map_v5",
    "v5_t6": "map_v5_t6",
    "v5_t7": "map_v5_t7",
    "v5_clean": "map_v5_clean",
    "v5_rel5": "map_v5_rel5",
    "v5_rel7": "map_v5_rel7",
    "base": "map",
    "rank": "map_rank",
    "ease": "map_ease",
    "venice": "map_venice",
    # same 2D map as "ease", lifted through the v6 LoRA merged into the base ckpt
    # (Zaun1996/segvigen-lora v6/lora_last.pt -> merge_lora.py -> ckpt/full_seg_v6.ckpt)
    "ease_v6": "map_ease",
    # v3 overlay map lifted through the v6 ckpt (pair with --ckpt ckpt/full_seg_v6.ckpt)
    "v3_v6": "map",
}


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def freeze(module):
    module.train(False)
    return module


def cond_img(key: str, tag: str) -> str:
    variant = TAG_MAP[tag]
    d = wd(key)
    if variant == "map":
        return os.path.join(d, "map.png")
    return map_paths(key, variant, "front")[0]


def seed_export(d: str, tag: str, seed: int | None) -> str:
    if seed is None:
        return os.path.join(d, f"seg_{tag}.glb")
    return os.path.join(d, f"seg_{tag}_s{seed}.glb")


def collect(keys, tags, seeds, force: bool):
    jobs = []
    for key in keys:
        d = wd(key)
        glb = src_glb(key)
        for tag in tags:
            img = cond_img(key, tag)
            if not os.path.exists(img):
                say(f"skip {key} {tag}: no {img}")
                continue
            if seeds:
                pending = [s for s in seeds if force or not os.path.exists(seed_export(d, tag, s))]
                if not pending:
                    say(f"skip sample {key} {tag}: all {len(seeds)} seed GLBs exist")
                    continue
                jobs.append({
                    "key": key, "tag": tag, "img": img, "glb": glb,
                    "vxz": os.path.join(d, "input.vxz"), "seeds": pending,
                })
            else:
                export = seed_export(d, tag, None)
                if os.path.exists(export) and not force:
                    continue
                jobs.append({
                    "key": key, "tag": tag, "img": img, "glb": glb,
                    "vxz": os.path.join(d, "input.vxz"), "export": export, "seeds": None,
                })
    return jobs


def infer_batch(ckpt, items):
    import torch
    import inference_full as inf
    from collections import defaultdict

    say(f"load {os.path.basename(ckpt)} ({len(items)} jobs)")
    with open(os.path.join(ROOT, "microsoft", "TRELLIS.2-4B", "pipeline.json"), "r") as f:
        pipeline_args = json.load(f)["args"]
    tex_flow = inf.models.from_pretrained(
        "microsoft/TRELLIS.2-4B/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16")
    gen3dseg = inf.Gen3DSeg(tex_flow)
    state = torch.load(ckpt, map_location="cpu")["state_dict"]
    state = inf.OrderedDict([(k.replace("gen3dseg.", ""), v) for k, v in state.items()])
    gen3dseg.load_state_dict(state)
    freeze(gen3dseg).cuda()
    sampler = inf.Sampler()
    shape_encoder = freeze(inf.models.from_pretrained(
        "microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16").cuda())
    tex_encoder = freeze(inf.models.from_pretrained(
        "microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16").cuda())
    shape_decoder = freeze(inf.models.from_pretrained(
        "microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16").cuda())
    tex_decoder = freeze(inf.models.from_pretrained(
        "microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16").cuda())
    rembg = inf.BiRefNet(model_name="briaai/RMBG-2.0")
    rembg.cuda()
    dino = inf.DinoV3FeatureExtractor(model_name="facebook/dinov3-vitl16-pretrain-lvd1689m")
    dino.cuda()

    by_key = defaultdict(list)
    for it in items:
        by_key[it["key"]].append(it)
    for key, group in by_key.items():
        it0 = group[0]
        if not os.path.exists(it0["vxz"]):
            say(f"voxelize {key}")
            inf.process_glb_to_vxz(it0["glb"], it0["vxz"])
        shape_slat, meshes, subs, tex_slat = inf.vxz_to_latent_slat(
            shape_encoder, shape_decoder, tex_encoder, it0["vxz"])
        for it in group:
            image = inf.preprocess_image(rembg, inf.Image.open(it["img"]))
            cond = inf.get_cond(dino, [image])
            runs = [(None, it["export"])] if not it.get("seeds") else [
                (s, seed_export(wd(it["key"]), it["tag"], s)) for s in it["seeds"]
            ]
            for seed, export in runs:
                t0 = time.time()
                label = f"{it['key']} {it['tag']}" + (f" seed={seed}" if seed is not None else "")
                say(f"sample {label}")
                try:
                    if seed is not None:
                        torch.manual_seed(seed)
                        torch.cuda.manual_seed_all(seed)
                    out_slat = inf.tex_slat_sample_single(
                        gen3dseg, sampler, pipeline_args, shape_slat, tex_slat, cond)
                    with torch.no_grad():
                        tex_voxels = tex_decoder(out_slat, guide_subs=subs) * 0.5 + 0.5
                    inf.slat_to_glb(meshes, tex_voxels).export(export)
                except Exception as exc:
                    say(f"  FAILED {label}: {exc}")
                dt = time.time() - t0
                stamp = os.path.join(wd(it["key"]), "times.json")
                times = json.load(open(stamp, encoding="utf-8")) if os.path.exists(stamp) else {}
                times[f"segvigen_{it['tag']}" + (f"_s{seed}" if seed is not None else "")] = round(dt, 1)
                with open(stamp, "w", encoding="utf-8") as f:
                    json.dump(times, f, ensure_ascii=False, indent=1)
                if os.path.exists(export):
                    say(f"  wrote {export} ({dt:.0f}s)")
        del shape_slat, meshes, subs, tex_slat
        torch.cuda.empty_cache()


def load_rgb_white(path: str) -> np.ndarray:
    from PIL import Image
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    return np.asarray(Image.alpha_composite(bg, im).convert("RGB"), dtype=np.float32)


def map_agreement(pred_png: str, map_png: str, legend_json: str) -> float:
    """Nearest-palette label agreement on the map's non-white pixels (eval_fidelity style)."""
    from PIL import Image
    from eval_parts import legend_palette

    pred = load_rgb_white(pred_png)
    ref = load_rgb_white(map_png)
    if pred.shape[:2] != ref.shape[:2]:
        pred = np.asarray(
            Image.fromarray(pred.astype(np.uint8)).resize((ref.shape[1], ref.shape[0]), Image.NEAREST),
            dtype=np.float32)
    palette, _ = legend_palette(legend_json)
    pal = np.asarray(palette, dtype=np.float32)
    fg = ~((ref.min(axis=-1) > 248) & (ref.max(axis=-1) > 248))
    if int(fg.sum()) < 50:
        return 0.0
    pr = pred[fg]
    rf = ref[fg]
    pred_lab = ((pr[:, None, :] - pal[None, :, :]) ** 2).sum(-1).argmin(1)
    ref_lab = ((rf[:, None, :] - pal[None, :, :]) ** 2).sum(-1).argmin(1)
    return float((pred_lab == ref_lab).mean())


def render_az(glb: str, out_png: str, az: float, log: str, samples: int | None = None) -> str | None:
    target = out_png.replace(".png", f"_{az:g}.png")
    if os.path.exists(target):
        return target
    cmd = [PY, os.path.join(ROOT, "data_toolkit", "render_cond_view.py"),
           "--glb", glb, "--transforms", TRANSFORMS, "--out", target, "--azimuths", f"{az:g}"]
    if samples is not None:
        cmd += ["--samples", str(samples)]
    run(cmd, log)
    return target if os.path.exists(target) else None


def pick_closest(keys, tags, seeds, preview_samples: int):
    """Upright every seed, score the front render against the 2D map, promote the winner."""
    for key in keys:
        d = wd(key)
        glb = src_glb(key)
        az = float(front_json(key)["azimuth"])
        log = os.path.join(d, "log.txt")
        vis = os.path.join(d, "vis")
        os.makedirs(vis, exist_ok=True)
        for tag in tags:
            variant = TAG_MAP[tag]
            map_png, legend = map_paths(key, variant, "front") if variant != "map" else (
                os.path.join(d, "map.png"), os.path.join(d, "legend.json"))
            rows = []
            for seed in seeds:
                seg = seed_export(d, tag, seed)
                up = os.path.join(d, f"seg_{tag}_s{seed}_upright.glb")
                if not os.path.exists(seg):
                    say(f"  missing {seg}")
                    continue
                if not os.path.exists(up):
                    frame, cov = upright(seg, glb, up)
                    say(f"  upright {key} {tag} s{seed}: frame={frame} coverage={cov:.3f}")
                preview = render_az(up, os.path.join(vis, f"{tag}_s{seed}.png"), az, log, preview_samples)
                if preview is None:
                    say(f"  no preview {key} {tag} s{seed}")
                    continue
                score = map_agreement(preview, map_png, legend)
                rows.append({"seed": seed, "score": round(score, 4), "seg": seg, "upright": up, "preview": preview})
                say(f"  {key} {tag} s{seed}: map_agree={score:.4f}")
            if not rows:
                say(f"  no seeds to pick for {key} {tag}")
                continue
            best = max(rows, key=lambda r: r["score"])
            dest_seg = os.path.join(d, f"seg_{tag}.glb")
            dest_up = os.path.join(d, f"seg_{tag}_upright.glb")
            shutil.copyfile(best["seg"], dest_seg)
            shutil.copyfile(best["upright"], dest_up)
            pick = {"tag": tag, "map": map_png, "winner": best["seed"], "scores": rows}
            with open(os.path.join(d, f"{tag}_seed_pick.json"), "w", encoding="utf-8") as f:
                json.dump(pick, f, ensure_ascii=False, indent=1)
            say(f"PICK {key} {tag}: seed={best['seed']} score={best['score']:.4f} -> {dest_up}")


def render_winners(keys, tags, col: str):
    """Full-quality front/back Cycles of the promoted upright GLB. Deletes stale vis tiles first."""
    for key in keys:
        d = wd(key)
        az = float(front_json(key)["azimuth"])
        back = (az + 180) % 360
        log = os.path.join(d, "log.txt")
        vis = os.path.join(d, "vis")
        os.makedirs(vis, exist_ok=True)
        for tag in tags:
            up = os.path.join(d, f"seg_{tag}_upright.glb")
            if not os.path.exists(up):
                say(f"skip render {key} {tag}: no {up}")
                continue
            stem = os.path.join(vis, f"{col}.png")
            for a in (az, back):
                stale = stem.replace(".png", f"_{a:g}.png")
                if os.path.exists(stale):
                    os.remove(stale)
            front_png = render_az(up, stem, az, log, 128)
            back_png = render_az(up, stem, back, log, 128)
            say(f"  render {key} {tag}: front={front_png} back={back_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", nargs="*", default=None)
    ap.add_argument("--tags", default="v5,v5_t6,v5_t7")
    ap.add_argument("--ckpt", default=CKPT["base"])
    ap.add_argument("--seeds", default=None,
                    help="Comma-separated ints. Sample each, pick the one closest to the 2D map.")
    ap.add_argument("--force", action="store_true", help="Re-sample even if a seed GLB already exists.")
    ap.add_argument("--preview-samples", type=int, default=32,
                    help="Cycles samples for the seed-picking front preview.")
    ap.add_argument("--skip-sample", action="store_true", help="Only upright / pick / render existing GLBs.")
    ap.add_argument("--skip-pick", action="store_true")
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--col", default="lift_ease_v6",
                    help="vis/ stem written for the winner (must match ext_bench COLUMNS).")
    args = ap.parse_args()
    keys = args.keys or list(ASSETS)
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    for t in tags:
        if t not in TAG_MAP:
            raise SystemExit(f"unknown tag {t}, want {sorted(TAG_MAP)}")
    seeds = None
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip() != ""]
    os.chdir(ROOT)
    if not args.skip_sample:
        jobs = collect(keys, tags, seeds, args.force)
        if jobs:
            infer_batch(args.ckpt, jobs)
        else:
            say("nothing to sample")
    if seeds and not args.skip_pick:
        pick_closest(keys, tags, seeds, args.preview_samples)
    else:
        for key in keys:
            glb = src_glb(key)
            d = wd(key)
            for tag in tags:
                seg = os.path.join(d, f"seg_{tag}.glb")
                up = os.path.join(d, f"seg_{tag}_upright.glb")
                if os.path.exists(seg) and not os.path.exists(up):
                    frame, cov = upright(seg, glb, up)
                    say(f"upright {key} {tag}: frame={frame} coverage={cov:.3f}")
    if not args.skip_render:
        render_winners(keys, tags, args.col)
    say("LIFT DONE")


if __name__ == "__main__":
    main()
