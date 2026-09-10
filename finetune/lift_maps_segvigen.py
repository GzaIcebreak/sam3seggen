"""Lift ext_bench 2D maps through SegviGen (base 2D-map ckpt) and write upright GLBs.

Loads the DiT once. Voxelises each asset once. Samples every pending (asset, tag) pair.
Same export path as inference_full.py --two_d_map, then the same upright() as ext_bench.

    python finetune/lift_maps_segvigen.py --keys dog --tags v5
    python finetune/lift_maps_segvigen.py --tags v5,v5_t6,v5_t7
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("SEGVIGEN_DINOV3", os.path.join(ROOT, "weights", "facebook", "dinov3-vitl16-pretrain-lvd1689m"))
os.environ.setdefault("SEGVIGEN_RMBG", os.path.join(ROOT, "weights", "ZhengPeng7", "BiRefNet"))

from ext_bench import ASSETS, CKPT, OUT, map_paths, src_glb, upright, wd  # noqa: E402

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
}


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def freeze(module):
    module.train(False)
    return module


def collect(keys, tags):
    jobs = []
    for key in keys:
        d = wd(key)
        glb = src_glb(key)
        for tag in tags:
            variant = TAG_MAP[tag]
            img = map_paths(key, variant, "front")[0] if variant != "map" else os.path.join(d, "map.png")
            if variant == "map":
                img = os.path.join(d, "map.png")
            seg = os.path.join(d, f"seg_{tag}.glb")
            if os.path.exists(seg):
                continue
            if not os.path.exists(img):
                say(f"skip {key} {tag}: no {img}")
                continue
            jobs.append({
                "key": key, "tag": tag, "img": img, "glb": glb,
                "vxz": os.path.join(d, "input.vxz"), "export": seg,
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
            t0 = time.time()
            say(f"sample {it['key']} {it['tag']}")
            try:
                image = inf.preprocess_image(rembg, inf.Image.open(it["img"]))
                cond = inf.get_cond(dino, [image])
                out_slat = inf.tex_slat_sample_single(
                    gen3dseg, sampler, pipeline_args, shape_slat, tex_slat, cond)
                with torch.no_grad():
                    tex_voxels = tex_decoder(out_slat, guide_subs=subs) * 0.5 + 0.5
                inf.slat_to_glb(meshes, tex_voxels).export(it["export"])
            except Exception as exc:
                say(f"  FAILED {it['key']} {it['tag']}: {exc}")
            dt = time.time() - t0
            stamp = os.path.join(wd(it["key"]), "times.json")
            times = json.load(open(stamp, encoding="utf-8")) if os.path.exists(stamp) else {}
            times[f"segvigen_{it['tag']}"] = round(dt, 1)
            with open(stamp, "w", encoding="utf-8") as f:
                json.dump(times, f, ensure_ascii=False, indent=1)
            if os.path.exists(it["export"]):
                say(f"  wrote {it['export']} ({dt:.0f}s)")
        del shape_slat, meshes, subs, tex_slat
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", nargs="*", default=None)
    ap.add_argument("--tags", default="v5,v5_t6,v5_t7")
    ap.add_argument("--ckpt", default=CKPT["base"])
    args = ap.parse_args()
    keys = args.keys or list(ASSETS)
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    for t in tags:
        if t not in TAG_MAP:
            raise SystemExit(f"unknown tag {t}, want {sorted(TAG_MAP)}")
    os.chdir(ROOT)
    jobs = collect(keys, tags)
    if jobs:
        infer_batch(args.ckpt, jobs)
    else:
        say("nothing to sample")
    for key in keys:
        glb = src_glb(key)
        d = wd(key)
        for tag in tags:
            seg = os.path.join(d, f"seg_{tag}.glb")
            up = os.path.join(d, f"seg_{tag}_upright.glb")
            if os.path.exists(seg) and not os.path.exists(up):
                frame, cov = upright(seg, glb, up)
                say(f"upright {key} {tag}: frame={frame} coverage={cov:.3f}")
    say("LIFT DONE")


if __name__ == "__main__":
    main()
