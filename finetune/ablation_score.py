"""Score and tabulate the GeoSAM2 / SAM3 propagation ablation (geosam2_ablate.py + sam3_track.py outputs).

    finetune\run_ft.bat ablation_score.py hard     20 hard PartVerse objects (GT): eval_parts on every abl_* row
                                                    -> datasets/geosam2/eval/ablation.json
    finetune\run_ft.bat ablation_score.py ext      10 external assets (no GT): label-free stats + agreement with
                                                    SegviGen v6 / GeoSAM2 dual, parts GLBs, front/back montage
                                                    -> datasets/ext_bench/ablation_scores.json, ablation_*.png
    finetune\run_ft.bat ablation_score.py tables   markdown tables of both -> datasets/geosam2/eval/ablation_tables.md
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

DS = r"E:\AI_New\ModelGen\datasets"
HARD_LIST = os.path.join(DS, "pv_hard.txt")
PV = os.path.join(DS, "pv")
HARD_RESULTS = os.path.join(DS, "geosam2", "results")
HARD_EVAL = os.path.join(DS, "geosam2", "eval")
EXT = os.path.join(DS, "ext_bench")
EXT_RESULTS = os.path.join(EXT, "geosam2", "results")

ROWS = [  # (dir stem under abl_<stem>_match, label)
    ("sam3_p1", "SAM3 1 视角，直接提升（无传播）"),
    ("sam3_p2", "SAM3 2 视角，直接提升"),
    ("sam3_p4", "SAM3 4 视角，直接提升"),
    ("sam3_p12", "SAM3 12 视角，直接提升（SAM3 完全取代 SAM2）"),
    ("geo_p1", "GeoSAM2 传播（SAM2+LoRA+几何），1 提示视角"),
    ("geo_p2", "GeoSAM2 传播，2 提示视角"),
    ("geo_p4", "GeoSAM2 传播，4 提示视角"),
    ("lift_sam3track_p1_match", "SAM3 tracker 传播（无几何），1 提示视角"),
    ("lift_sam3track_p2_match", "SAM3 tracker 传播，2 提示视角"),
    ("lift_sam3track_p4_match", "SAM3 tracker 传播，4 提示视角"),
]
BASELINES = [  # rows from the earlier comparison (geosam2/eval/results.json), for reference
    ("segvigen_base", "SegviGen base + SAM3"),
    ("segvigen_v6", "我们的方案 (v6 + SAM3)"),
    ("single_match", "GeoSAM2 原版 single（1 提示视角 + 对面自动分割，raw）"),
    ("single_match_filled", "GeoSAM2 原版 single（同上，最近面填充）"),
    ("dual_match_filled", "GeoSAM2 dual（2 提示视角，旧流程，最近面填充）"),
]
HARD_COLS = [("miou", "mIoU"), ("miou_matched", "mIoU匹配"), ("sem_miou", "语义mIoU"), ("name_acc", "命名正确率"),
             ("boundary_f1", "边界F1"), ("over_seg_parts", "过分割件"), ("under_seg_segments", "欠分割段"),
             ("small_part_recall", "小件召回"), ("unlabelled_share", "留白"), ("n_pred_segments", "段数")]


def read(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def write(p, o):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(o, f, ensure_ascii=False, indent=1)


def objects(list_file):
    with open(list_file, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def row_dir(results_root, obj, stem):
    return os.path.join(results_root, obj, f"abl_{stem}_match")


def row_ok(d):
    lj = os.path.join(d, "labels.json")
    return os.path.exists(lj) and "error" not in read(lj) and os.path.exists(os.path.join(d, "faces.npy"))


# ----------------------------------------------------------------------------- hard (GT)
def score_hard():
    from common import ObjectDir
    from eval_parts import FaceLabelPred, evaluate
    out_path = os.path.join(HARD_EVAL, "ablation.json")
    scores = read(out_path) if os.path.exists(out_path) else {}
    for obj in objects(HARD_LIST):
        od = ObjectDir(os.path.join(PV, obj))
        for stem, _ in ROWS:
            d = row_dir(HARD_RESULTS, obj, stem)
            if not row_ok(d):
                continue
            info = read(os.path.join(d, "labels.json"))
            # faces_inst.npy is identical to faces.npy in practice: GeoSAM2's component renumbering in
            # complete_labels compares 0-d tensors by identity, so it never fires. Only faces.npy is scored.
            for variant, npy, ids_key in (("", "faces.npy", "ids"),):
                key = stem + variant
                if scores.get(key, {}).get(obj, {}).get("_mtime") == os.path.getmtime(os.path.join(d, npy)):
                    continue
                names = {int(k): v for k, v in info[ids_key].items()}
                pred = FaceLabelPred(os.path.join(d, "mesh.glb"), os.path.join(d, npy), names, {0, 999})
                t0 = time.time()
                res = evaluate(od, pred)
                s = res["summary"]
                s["seconds_method"] = info.get("seconds")
                s["_mtime"] = os.path.getmtime(os.path.join(d, npy))
                scores.setdefault(key, {})[obj] = s
                print(f"{obj[:8]} {key:32s} miou={s['miou']:.3f} matched={s['miou_matched']:.3f} sem={s['sem_miou'] or 0:.3f} "
                      f"name={s['name_acc']:.2f} bF1={s['boundary_f1'] or 0:.2f} segs={s['n_pred_segments']} ({time.time() - t0:.0f}s)",
                      flush=True)
                write(out_path, scores)
    return scores


# ----------------------------------------------------------------------------- ext (no GT)
PALETTE = [(228, 26, 28), (55, 126, 184), (77, 175, 74), (152, 78, 163), (255, 127, 0), (255, 255, 51),
           (166, 86, 40), (247, 129, 191), (0, 158, 115), (86, 180, 233), (213, 94, 0), (204, 121, 167),
           (120, 94, 240), (230, 159, 0), (0, 114, 178), (176, 122, 161), (60, 180, 75), (145, 30, 180),
           (245, 130, 48), (70, 240, 240)]


def parts_glb(d: str, ref_glb: str, out: str):
    """faces.npy -> one flat-coloured sub-mesh per label in the asset frame; colour fixed by prompt index."""
    import trimesh
    from common import GREY
    from geosam2_to_glb import load_mesh, to_reference_frame
    mesh = to_reference_frame(load_mesh(os.path.join(d, "mesh.glb")), load_mesh(ref_glb))
    labels = np.rint(np.load(os.path.join(d, "faces.npy"))).astype(np.int64)
    names = {int(k): v for k, v in read(os.path.join(d, "labels.json"))["ids"].items()}
    scene = trimesh.Scene()
    for k in [int(k) for k in np.unique(labels)]:
        sub = mesh.submesh([np.nonzero(labels == k)[0]], append=True)
        if k in (0, 999):
            name, color = "unlabelled", GREY
        else:
            name, color = names.get(k, f"label_{k}"), PALETTE[(k - 1) % len(PALETTE)]
        mat = trimesh.visual.material.PBRMaterial(baseColorFactor=[*color, 255], metallicFactor=0.0, roughnessFactor=0.9)
        sub.visual = trimesh.visual.TextureVisuals(material=mat)
        scene.add_geometry(sub, node_name=f"{name}_{k}", geom_name=f"{name}_{k}")
    scene.export(out)


def score_ext():
    import ext_bench as eb
    from eval_parts import FaceLabelPred
    out_path = os.path.join(EXT, "ablation_scores.json")
    scores = read(out_path) if os.path.exists(out_path) else {}
    for k in eb.ASSETS:
        rows_here = [(stem, row_dir(EXT_RESULTS, k, stem)) for stem, _ in ROWS]
        rows_here = [(s, d) for s, d in rows_here if row_ok(d)]
        if not rows_here:
            continue
        eb.say(f"ext score {k}")
        preds = eb.load_preds(k)  # reference methods from the main benchmark
        ref = {m: preds[m] for m in ("segvigen_v6", "geosam2_dual_match_filled", "segvigen_native") if m in preds}
        for stem, d in rows_here:
            info = read(os.path.join(d, "labels.json"))
            ref[f"abl_{stem}"] = FaceLabelPred(os.path.join(d, "mesh.glb"), os.path.join(d, "faces.npy"),
                                                {int(i): n for i, n in info["ids"].items()}, {0, 999})
            glb = os.path.join(eb.wd(k), f"abl_{stem}_parts.glb")
            if not os.path.exists(glb):
                parts_glb(d, eb.src_glb(k), glb)
        pts, spacing = eb.surface_samples(eb.src_glb(k))
        labs = {m: eb.labels_on_samples(p, pts)[0] for m, p in ref.items()}
        entry = scores.setdefault(k, {"methods": {}, "agreement": {}})
        for stem, d in rows_here:
            m = f"abl_{stem}"
            lab = labs[m]
            fr = eb.fragments(pts, lab, spacing)
            segs = [s for s in fr if s >= 0]
            info = read(os.path.join(d, "labels.json"))
            entry["methods"][m] = {
                "n_segments": len(segs), "names": sorted({ref[m].name_of(s) for s in segs if ref[m].name_of(s)}),
                "fragments_per_segment": float(np.mean([fr[s] for s in segs])) if segs else None,
                "segments_fragmented": int(sum(fr[s] >= 2 for s in segs)),
                "unlabelled_share": float((lab == eb.UNLABELLED).mean()),
                "unlabelled_raw_faces": info["unlabelled_faces_raw"] / max(1, info["n_faces"]),
                "boundary_share": eb.boundary_share(pts, lab, spacing),
                "seconds": info.get("seconds"),
            }
            for r in ("segvigen_v6", "geosam2_dual_match_filled", "segvigen_native"):
                if r in labs:
                    wm, mm = eb.matched_miou(lab, labs[r])
                    e = {"miou_weighted": wm, "miou_mean": mm}
                    if r != "segvigen_native":
                        ni = eb.name_iou(ref[m], lab, ref[r], labs[r])
                        vals = [v for v in ni.values() if v is not None]
                        e["name_miou"] = float(np.mean(vals)) if vals else None
                    entry["agreement"][f"{m}|{r}"] = e
            eb.say(f"  {m:30s} segs={len(segs):2d} frag/seg={entry['methods'][m]['fragments_per_segment'] or 0:.2f} "
                   f"unlab={entry['methods'][m]['unlabelled_share']:.2f} bnd={entry['methods'][m]['boundary_share'] or 0:.3f}")
        write(out_path, scores)
    montage_ext(scores)
    return scores


MONTAGE_COLS = [("input", "Input"), ("map", "SAM3 map"), ("segvigen_v6", "v6 + SAM3"),
                ("geosam2_dual_match_filled", "GeoSAM2 dual 填充 (旧)"),
                ("abl_sam3_p12", "SAM3 x12 直接提升"), ("abl_geo_p1", "GeoSAM2 传播 p1"), ("abl_geo_p2", "GeoSAM2 传播 p2"),
                ("abl_geo_p4", "GeoSAM2 传播 p4"), ("abl_lift_sam3track_p1_match", "SAM3 tracker p1"),
                ("abl_lift_sam3track_p2_match", "SAM3 tracker p2"), ("abl_lift_sam3track_p4_match", "SAM3 tracker p4")]


def montage_ext(scores):
    import ext_bench as eb
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 15)
    except OSError:
        font = ImageFont.load_default()
    keys = [k for k in eb.ASSETS if k in scores]
    for side in ("front", "back"):
        S = 256
        W, H = len(MONTAGE_COLS) * S, len(keys) * (S + 22) + 22
        canvas = Image.new("RGB", (W, H), "white")
        draw = ImageDraw.Draw(canvas)
        for c, (_, label) in enumerate(MONTAGE_COLS):
            draw.text((c * S + 4, 3), label, fill="black", font=font)
        for r, k in enumerate(keys):
            d = eb.wd(k)
            az = eb.front_json(k)["azimuth"]
            if side == "back":
                az = (az + 180) % 360
            y = 22 + r * (S + 22)
            draw.text((4, y + 3), f"{k} ({eb.ASSETS[k][0]})", fill="black", font=font)
            for c, (col, _) in enumerate(MONTAGE_COLS):
                if col == "input":
                    t = os.path.join(d, "render.png" if side == "front" else "render_back.png")
                elif col == "map":
                    t = os.path.join(d, "map.png") if side == "front" else None
                else:
                    glb = os.path.join(d, "seg_v6_upright.glb" if col == "segvigen_v6" else f"{col}_parts.glb")
                    t = eb.render_view(glb, os.path.join(d, "vis", f"{col}.png"), az, os.path.join(d, "log.txt")) \
                        if os.path.exists(glb) else None
                if t and os.path.exists(t):
                    im = Image.open(t).convert("RGBA")
                    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
                    canvas.paste(Image.alpha_composite(bg, im).convert("RGB").resize((S, S)), (c * S, y + 22))
                else:
                    draw.text((c * S + S // 2 - 12, y + 22 + S // 2), "n/a", fill="grey", font=font)
        out = os.path.join(EXT, f"ablation_{side}.png")
        canvas.save(out)
        eb.say(f"saved {out}")


# ----------------------------------------------------------------------------- tables
def fmt(v, nd=3):
    return "–" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def mean_of(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def tables():
    L = []
    hard = read(os.path.join(HARD_EVAL, "ablation.json")) if os.path.exists(os.path.join(HARD_EVAL, "ablation.json")) else {}
    base = read(os.path.join(HARD_EVAL, "results.json")) if os.path.exists(os.path.join(HARD_EVAL, "results.json")) else {}
    objs = objects(HARD_LIST)
    L.append("### hard 集（20 个 PartVerse 物体，有 GT；均值）\n")
    L.append("| 行 | 物体数 | " + " | ".join(c for _, c in HARD_COLS) + " | 秒 |")
    L.append("|---|---|" + "---|" * (len(HARD_COLS) + 1))
    for stem, label in BASELINES:
        rows = [base[stem][o] for o in objs if o in base.get(stem, {})]
        if rows:
            L.append(f"| {label} | {len(rows)} | " + " | ".join(fmt(mean_of(rows, k), 3 if k not in ('over_seg_parts', 'under_seg_segments', 'n_pred_segments') else 1) for k, _ in HARD_COLS) + " | – |")
    for stem, label in ROWS:
        rows = [hard[stem][o] for o in objs if o in hard.get(stem, {})]
        if rows:
            L.append(f"| {label} | {len(rows)} | " + " | ".join(
                fmt(mean_of(rows, k), 3 if k not in ('over_seg_parts', 'under_seg_segments', 'n_pred_segments') else 1)
                for k, _ in HARD_COLS) + f" | {fmt(mean_of(rows, 'seconds_method'), 1)} |")
    L.append("")
    ext = read(os.path.join(EXT, "ablation_scores.json")) if os.path.exists(os.path.join(EXT, "ablation_scores.json")) else {}
    main = read(os.path.join(EXT, "scores.json")) if os.path.exists(os.path.join(EXT, "scores.json")) else {}
    if ext:
        L.append("### 外部资产（10 个，无 GT；均值）\n")
        L.append("| 行 | 资产数 | 段数 | 碎片/段 | 碎段 | 留白(填充后) | 留白(提升后原始) | 边界密度 | 与 v6 一致(类无关/按名) | 与 GeoSAM2 dual 一致(类无关/按名) | 与原生一致 | 秒 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for m, label in (("segvigen_v6", "我们的方案 (v6 + SAM3)"), ("geosam2_dual_match_filled", "GeoSAM2 dual（填充，旧流程）"),
                         ("geosam2_single_match", "GeoSAM2 single（raw，旧流程）")):
            rows = [main[k]["methods"][m] for k in main if m in main[k]["methods"]]
            if rows:
                L.append(f"| {label} | {len(rows)} | {fmt(mean_of(rows, 'n_segments'), 1)} | {fmt(mean_of(rows, 'fragments_per_segment'), 2)} | "
                         f"{fmt(mean_of(rows, 'segments_fragmented'), 1)} | {fmt(mean_of(rows, 'unlabelled_share'), 2)} | – | "
                         f"{fmt(mean_of(rows, 'boundary_share'), 3)} | – | – | – | – |")
        for stem, label in ROWS:
            m = f"abl_{stem}"
            rows = [ext[k]["methods"][m] for k in ext if m in ext[k]["methods"]]
            if not rows:
                continue
            def agree(r, key):
                es = [ext[k]["agreement"].get(f"{m}|{r}") for k in ext if f"{m}|{r}" in ext[k]["agreement"]]
                return mean_of(es, key)
            L.append(f"| {label} | {len(rows)} | {fmt(mean_of(rows, 'n_segments'), 1)} | {fmt(mean_of(rows, 'fragments_per_segment'), 2)} | "
                     f"{fmt(mean_of(rows, 'segments_fragmented'), 1)} | {fmt(mean_of(rows, 'unlabelled_share'), 2)} | "
                     f"{fmt(mean_of(rows, 'unlabelled_raw_faces'), 2)} | {fmt(mean_of(rows, 'boundary_share'), 3)} | "
                     f"{fmt(agree('segvigen_v6', 'miou_weighted'), 2)} / {fmt(agree('segvigen_v6', 'name_miou'), 2)} | "
                     f"{fmt(agree('geosam2_dual_match_filled', 'miou_weighted'), 2)} / {fmt(agree('geosam2_dual_match_filled', 'name_miou'), 2)} | "
                     f"{fmt(agree('segvigen_native', 'miou_weighted'), 2)} | {fmt(mean_of(rows, 'seconds'), 1)} |")
        L.append("")
    out = os.path.join(HARD_EVAL, "ablation_tables.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))
    print(f"saved {out}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "tables"
    {"hard": score_hard, "ext": score_ext, "tables": tables}[what]()
