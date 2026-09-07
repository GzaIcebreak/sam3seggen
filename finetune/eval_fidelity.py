"""Colour fidelity of an inference result against the 2D map that conditioned it.

For every GT part: sample surface points, look up the colour the output GLB baked there,
and compare with the colour the map assigned to that part (grey_parts expect GREY).

Scoring is nearest-palette *assignment*, not colour equality. SegviGen's decoder reproduces
a palette colour with a drift of tens of RGB units, so an absolute distance threshold rejects
parts that carry the right colour: expected (210,242,63) coming out as (146,239,81) is the
same yellow-green, 66 units away, and a 60-unit cutoff scores it 0. Since what a segmentation
needs is the right palette *entry*, each sample goes to its nearest entry with no cutoff, and
the drift is reported separately as a distance so the two failure modes stay distinguishable.

Per part: fidelity = share of samples whose nearest palette entry is the expected one;
dist_median = how far those samples sit from it; margin = distance to the expected entry minus
distance to the closest competing entry (negative = assigned correctly, near 0 = about to flip);
purity = share carrying the part's dominant entry (1.0 = one solid colour, low = fragmented).
The old threshold-based number is kept as fidelity_snap60 so earlier reports stay comparable.

    finetune\run_ft.bat eval_fidelity.py --object <root>/<obj> --variant sam3_az0 --output_glb out.glb
    finetune\run_ft.bat eval_fidelity.py --object <root>/<obj> --variant sam3_az0 --run_inference --ckpt ckpt/x.ckpt
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import trimesh
from PIL import Image

import common
from common import ObjectDir, GREY


def texture_lookup(mesh: trimesh.Trimesh, face_idx: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Baked base colour at surface points via barycentric UV interpolation."""
    tri = mesh.triangles[face_idx]
    bary = trimesh.triangles.points_to_barycentric(tri, points)
    # degenerate (zero-area) faces give NaN barycentrics -> fall back to the first vertex
    bad = ~np.isfinite(bary).all(1)
    bary[bad] = [1.0, 0.0, 0.0]
    uv = mesh.visual.uv[mesh.faces[face_idx]]
    p_uv = np.nan_to_num((bary[:, :, None] * uv).sum(1), nan=0.0)
    mat = mesh.visual.material
    img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    if img is None:
        color = np.asarray(getattr(mat, "baseColorFactor", [255, 255, 255, 255]))[:3]
        return np.tile(color, (len(points), 1)).astype(np.float32)
    img = np.asarray(img.convert("RGB"))
    h, w = img.shape[:2]
    x = np.clip((p_uv[:, 0] % 1.0) * (w - 1), 0, w - 1).round().astype(int)
    y = np.clip(((1 - p_uv[:, 1]) % 1.0) * (h - 1), 0, h - 1).round().astype(int)
    return img[y, x].astype(np.float32)


def load_output(glb_path: str) -> trimesh.Trimesh:
    loaded = trimesh.load(glb_path, force="scene")
    meshes = [m for m in loaded.dump() if isinstance(m, trimesh.Trimesh)]
    if len(meshes) != 1:
        raise ValueError(f"{glb_path}: expected one textured mesh, got {len(meshes)}")
    return meshes[0]


def normalize_like_vxz(mesh: trimesh.Trimesh, aabb: np.ndarray) -> trimesh.Trimesh:
    center = (aabb[0] + aabb[1]) / 2
    scale = 0.99999 / float((aabb[1] - aabb[0]).max())
    m = mesh.copy()
    m.apply_translation(-center)
    m.apply_scale(scale)
    return m


# slat_to_glb exports in TRELLIS' Y-up convention (a 90 deg turn about X w.r.t. the voxel frame);
# pick whichever candidate frame actually lands on the input geometry.
_FRAMES = {
    "identity": np.eye(4),
    "x+90": np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float),
    "x-90": np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float),
}


def align_output(out: trimesh.Trimesh, reference_pts: np.ndarray) -> tuple[trimesh.Trimesh, str, float]:
    best = None
    for name, T in _FRAMES.items():
        cand = out.copy()
        cand.apply_transform(T)
        _, dist, _ = trimesh.proximity.ProximityQuery(cand).on_surface(reference_pts)
        coverage = float((dist < 0.02).mean())
        if best is None or coverage > best[2]:
            best = (cand, name, coverage)
    return best


def components_per_label(pts: np.ndarray, labels: np.ndarray, n_labels: int, spacing: float,
                         min_share: float = 0.01) -> dict[int, int]:
    """Connected regions per colour over uniform-density surface samples: points of one colour are
    joined when closer than 2 sample spacings. Specks under min_share of that colour's points are
    ignored. Done on samples of the *input* parts, so the output mesh's topology (double shells,
    UV seams, broken extraction) cannot inflate the count."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    pairs = cKDTree(pts).query_pairs(2.0 * spacing, output_type="ndarray")
    same = labels[pairs[:, 0]] == labels[pairs[:, 1]]
    pairs = pairs[same]
    n = len(pts)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, comp = connected_components(g, directed=False)
    out = {}
    for k in range(n_labels):
        sel = labels == k
        if not sel.any():
            out[k] = 0
            continue
        sizes = np.bincount(comp[sel])
        sizes = sizes[sizes > 0]
        out[k] = int((sizes / sel.sum() >= min_share).sum())
    return out


def boundary_f1(pts: np.ndarray, gt: np.ndarray, pred: np.ndarray, spacing: float, tol: float = 0.01) -> dict:
    """Boundary points = points with a differently-labelled neighbour within 1.5 sample spacings.
    F1 of the predicted (colour-group) boundary against the expected one, matched within tol
    (normalised frame: the object fits the unit cube, so 0.01 = 1% of its largest extent)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    pairs = tree.query_pairs(1.5 * spacing, output_type="ndarray")
    if len(pairs) == 0:
        return {"boundary_f1": None}

    def boundary(lab):
        diff = lab[pairs[:, 0]] != lab[pairs[:, 1]]
        idx = np.unique(pairs[diff].ravel())
        return pts[idx]

    b_gt, b_pred = boundary(gt), boundary(pred)
    if len(b_gt) == 0 or len(b_pred) == 0:
        return {"boundary_f1": None, "boundary_gt_pts": int(len(b_gt)), "boundary_pred_pts": int(len(b_pred))}
    d_p, _ = cKDTree(b_gt).query(b_pred)
    d_r, _ = cKDTree(b_pred).query(b_gt)
    prec, rec = float((d_p < tol).mean()), float((d_r < tol).mean())
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"boundary_f1": f1, "boundary_precision": prec, "boundary_recall": rec, "boundary_tol": tol,
            "boundary_gt_pts": int(len(b_gt)), "boundary_pred_pts": int(len(b_pred))}


def boundary_samples(parts, aabb, expected: dict, palette: list, query, out, pal, n_total: int = 40000):
    """Uniform-density surface samples over the whole object (per-part caps would make the
    boundary test depend on part size): points, expected colour index, predicted colour index."""
    parts_n = [normalize_like_vxz(p, aabb) for p in parts]
    total = sum(p.area for p in parts_n)
    density = n_total / max(total, 1e-9)
    pts_l, gt_l, pred_l = [], [], []
    for p, part_n in enumerate(parts_n):
        if expected.get(p) not in palette:
            continue
        n = max(30, int(part_n.area * density))
        pts, _ = trimesh.sample.sample_surface(part_n, n)
        closest, dist, face_idx = query.on_surface(pts)
        keep = dist < 0.02
        if not keep.any():
            continue
        colors = texture_lookup(out, face_idx[keep], closest[keep])
        pred = np.sqrt(((colors[:, None, :] - pal[None]) ** 2).sum(-1)).argmin(1)
        pts_l.append(pts[keep])
        gt_l.append(np.full(int(keep.sum()), palette.index(expected[p])))
        pred_l.append(pred)
    if not pts_l:
        return None
    return np.concatenate(pts_l), np.concatenate(gt_l), np.concatenate(pred_l), float(np.sqrt(1.0 / density))


def evaluate(obj: ObjectDir, variant: str, output_glb: str, samples_per_part: int = 3000, snap_dist: float = 60.0) -> dict:
    with open(os.path.join(obj.variant_dir(variant), "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(obj.ids_meta, "r", encoding="utf-8") as f:
        aabb = np.asarray(json.load(f)["aabb"])
    expected: dict[int, tuple] = {}
    for g, members in enumerate(meta["groups"]):
        for p in members:
            expected[p] = tuple(meta["colors"][g])
    for p in meta["grey_parts"]:
        expected[p] = GREY
    palette = [tuple(c) for c in meta["colors"]] + [GREY]
    pal = np.asarray(palette, dtype=np.float32)

    parts = common.load_parts(obj)
    names = obj.names() or [str(i) for i in range(len(parts))]
    whole_n = normalize_like_vxz(trimesh.util.concatenate(parts), aabb)
    ref_pts, _ = trimesh.sample.sample_surface(whole_n, 2000)
    out, frame, coverage = align_output(load_output(output_glb), ref_pts)
    query = trimesh.proximity.ProximityQuery(out)
    rows = []
    for p, part in enumerate(parts):
        part_n = normalize_like_vxz(part, aabb)
        n = min(samples_per_part, max(200, int(part_n.area * 2e5)))
        pts, _ = trimesh.sample.sample_surface(part_n, n)
        closest, dist, face_idx = query.on_surface(pts)
        keep = dist < 0.02
        if keep.sum() < 20:
            rows.append({"part": p, "name": names[p], "samples": int(keep.sum()), "note": "not covered by output"})
            continue
        colors = texture_lookup(out, face_idx[keep], closest[keep])
        cdist = np.sqrt(((colors[:, None, :] - pal[None]) ** 2).sum(-1))
        nearest = cdist.argmin(1)
        drift = cdist[np.arange(len(nearest)), nearest]
        counts = np.bincount(nearest, minlength=len(palette))
        assigned = int(counts.argmax())
        exp = expected.get(p)
        exp_idx = palette.index(exp) if exp in palette else None
        row = {
            "part": p, "name": names[p], "samples": int(keep.sum()),
            "expected": list(exp) if exp else None,
            "expected_kind": "grey" if exp == GREY else ("colour" if exp else "none"),
            "assigned": list(palette[assigned]),
            "purity": float(counts.max() / counts.sum()),
            "other_share": float((drift > snap_dist).mean()),  # drift past the old cutoff
        }
        if exp_idx is not None:
            d_exp = cdist[:, exp_idx]
            d_rival = np.delete(cdist, exp_idx, axis=1).min(1)
            snapped = np.where(drift > snap_dist, -1, nearest)
            snap_counts = np.bincount(snapped + 1, minlength=len(palette) + 1)
            row.update({
                "fidelity": float(counts[exp_idx] / counts.sum()),
                "correct": assigned == exp_idx,
                "dist_median": float(np.median(d_exp)),
                "dist_p90": float(np.percentile(d_exp, 90)),
                "margin": float(np.median(d_exp - d_rival)),
                "fidelity_snap60": float(snap_counts[exp_idx + 1] / snap_counts.sum()),
            })
        else:
            row["fidelity"] = None
        rows.append(row)
    hidden = set(meta.get("hidden_parts", []))
    for r in rows:
        r["hidden"] = r["part"] in hidden
    scored = [r for r in rows if r.get("fidelity") is not None and not r["hidden"]]
    hidden_rows = [r for r in rows if r.get("fidelity") is not None and r["hidden"]]

    # fragmentation + boundary: uniform-density samples over the input parts, coloured by the output
    bs = boundary_samples(parts, aabb, expected, palette, query, out, pal)
    group_rows = []
    bnd = {"boundary_f1": None}
    comps_pred = comps_gt = {}
    if bs:
        b_pts, b_gt, b_pred, spacing = bs
        comps_gt = components_per_label(b_pts, b_gt, len(palette), spacing)
        comps_pred = components_per_label(b_pts, b_pred, len(palette), spacing)
        bnd = boundary_f1(b_pts, b_gt, b_pred, spacing)
    for g, members in enumerate(meta["groups"]):
        group_rows.append({"group": g, "color": list(palette[g]), "parts": members,
                           "expected_components": comps_gt.get(g, 0), "output_components": comps_pred.get(g, 0),
                           "fragments": max(0, comps_pred.get(g, 0) - comps_gt.get(g, 0))})

    summary = {
        # fragments: extra connected regions of a colour beyond what the GT group itself has
        "fragments_total": int(sum(r["fragments"] for r in group_rows)),
        "groups_fragmented": int(sum(r["fragments"] > 0 for r in group_rows)),
        "grey_components": int(comps_pred.get(len(palette) - 1, 0)),
        "grey_components_expected": int(comps_gt.get(len(palette) - 1, 0)),
        **bnd,
        "metric": "nearest-palette",  # tells these reports apart from the pre-threshold-fix ones
        "variant": variant, "kind": meta["kind"], "n_parts": len(parts), "n_visible_scored": len(scored),
        "output_frame": frame, "surface_coverage": coverage,
        "mean_fidelity": float(np.mean([r["fidelity"] for r in scored])) if scored else None,
        "parts_correct(>=0.8)": int(sum(r["fidelity"] >= 0.8 for r in scored)),
        "parts_assigned_correct": int(sum(r["correct"] for r in scored)),
        "median_dist": float(np.median([r["dist_median"] for r in scored])) if scored else None,
        "median_margin": float(np.median([r["margin"] for r in scored])) if scored else None,
        "mean_purity": float(np.mean([r["purity"] for r in scored])) if scored else None,
        "mean_other_share": float(np.mean([r["other_share"] for r in scored])) if scored else None,
        "snap_dist": snap_dist,
        "mean_fidelity_snap60": float(np.mean([r["fidelity_snap60"] for r in scored])) if scored else None,
        "parts_correct_snap60(>=0.8)": int(sum(r["fidelity_snap60"] >= 0.8 for r in scored)),
        "grey_parts_expected": len(meta["grey_parts"]),
        "hidden_parts": sorted(hidden),
        "hidden_mean_purity": float(np.mean([r["purity"] for r in hidden_rows])) if hidden_rows else None,
    }
    return {"summary": summary, "parts": rows, "groups": group_rows}


def write_legend(obj: ObjectDir, variant: str, text_cache: str, shuffle: bool = False, seed: int = 0,
                 swap: tuple[str, str] | None = None) -> str:
    """Legend json for inference_full.py --legend from the variant's meta (groups, colors) and the
    Stage-B text cache, built exactly like finetune/dataset.py does for training.
    shuffle permutes all group names (control run); swap exchanges the names of two groups
    (semantic-conflict test: do the boundaries follow the words?)."""
    import torch
    from dataset import TextCache, majority_name, object_name
    with open(os.path.join(obj.variant_dir(variant), "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    names = obj.names() or []
    cache = TextCache(text_cache)
    gnames = [majority_name(g, names) for g in meta["groups"]]
    if shuffle and len(gnames) > 1:
        perm = torch.randperm(len(gnames), generator=torch.Generator().manual_seed(seed)).tolist()
        gnames = [gnames[i] for i in perm]
    if swap:
        a, b = swap
        if a not in gnames or b not in gnames:
            raise SystemExit(f"--swap_names: {a!r}/{b!r} not both among the group names {gnames}")
        gnames = [b if n == a else a if n == b else n for n in gnames]
    entries = []
    for gname, color in zip(gnames, meta["colors"]):
        v = cache.get(gname)
        if v is not None:
            entries.append({"name": gname, "color": color, "text_vec": [round(float(x), 5) for x in v]})
    ov = cache.get(object_name(obj))
    stem = "legend" + ("_shuffled" if shuffle else "") + (f"_swap_{swap[0]}_{swap[1]}".replace(" ", "-") if swap else "")
    path = os.path.join(obj.variant_dir(variant), stem + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"entries": entries, "object_text_vec": [round(float(x), 5) for x in ov] if ov is not None else None,
                   "object": object_name(obj), "names": gnames}, f, ensure_ascii=False)
    return path


def partner_map(obj: ObjectDir, variant: str) -> str | None:
    with open(os.path.join(obj.variant_dir(variant), "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    if not meta.get("pair"):
        return None
    prefix = variant.rsplit("_", 1)[0]
    for vname in sorted(os.listdir(obj.variants_dir)):
        if vname == variant or not vname.startswith(prefix + "_"):
            continue
        mp = os.path.join(obj.variant_dir(vname), "meta.json")
        if os.path.exists(mp):
            with open(mp, encoding="utf-8") as f:
                if json.load(f).get("pair") == meta["pair"]:
                    return os.path.join(obj.variant_dir(vname), "map.png")
    return None


def run_inference(obj: ObjectDir, variant: str, ckpt: str, out_glb: str, guidance: float | None,
                  legend_ckpt: str | None = None, text_cache: str | None = None, pair: bool = False,
                  no_legend: bool = False, shuffle: bool = False, swap: tuple[str, str] | None = None) -> None:
    py = sys.executable
    cmd = [py, os.path.join(common.ROOT, "inference_full.py"), "--ckpt_path", ckpt, "--glb", obj.input_glb,
           "--input_vxz", obj.input_vxz, "--img", os.path.join(obj.variant_dir(variant), "map.png"),
           "--export_glb", out_glb, "--two_d_map"]
    if legend_ckpt:
        cmd += ["--legend_ckpt", legend_ckpt]
        if text_cache and not no_legend:
            cmd += ["--legend", write_legend(obj, variant, text_cache, shuffle=shuffle, swap=swap)]
        if pair:
            img2 = partner_map(obj, variant)
            if img2:
                cmd += ["--img2", img2]
    subprocess.run(cmd, check=True, cwd=common.ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--object", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output_glb", default=None)
    parser.add_argument("--run_inference", action="store_true")
    parser.add_argument("--ckpt", default=os.path.join(common.ROOT, "ckpt", "full_seg_w_2d_map.ckpt"))
    parser.add_argument("--report", default=None)
    parser.add_argument("--legend_ckpt", default=None, help="v3: lora_*.pt payload with the legend encoder")
    parser.add_argument("--text_cache", default=None, help="v3: Stage-B text_cache.pt (legend tokens from meta.json)")
    parser.add_argument("--pair", action="store_true", help="v3: also feed the partner view's map")
    parser.add_argument("--no_legend", action="store_true", help="v3: images only (ablation)")
    parser.add_argument("--shuffle_legend", action="store_true", help="v3 control: permuted names")
    parser.add_argument("--swap_names", default=None,
                        help="v3 semantic-conflict test: 'hand,staff' exchanges the two groups' names in the legend")
    args = parser.parse_args()

    os.chdir(common.ROOT)
    obj = ObjectDir(args.object)
    swap = tuple(s.strip() for s in args.swap_names.split(",")) if args.swap_names else None
    if swap and len(swap) != 2:
        parser.error("--swap_names needs exactly two comma-separated names")
    suffix = ("_leg" if args.legend_ckpt and args.text_cache and not args.no_legend else "") + \
             ("_shuf" if args.shuffle_legend else "") + ("_swap" if swap else "") + ("_pair" if args.pair else "")
    out_glb = args.output_glb or os.path.join(
        obj.variant_dir(args.variant), f"infer_{os.path.splitext(os.path.basename(args.ckpt))[0]}{suffix}.glb")
    if args.run_inference or not os.path.exists(out_glb):
        run_inference(obj, args.variant, args.ckpt, out_glb, None, legend_ckpt=args.legend_ckpt,
                      text_cache=args.text_cache, pair=args.pair, no_legend=args.no_legend,
                      shuffle=args.shuffle_legend, swap=swap)
    result = evaluate(obj, args.variant, out_glb)
    print(json.dumps(result["summary"], indent=2))
    for r in result["parts"]:
        if r.get("fidelity") is None:
            print(f"  {r['part']:>2d} {r['name']:<20s} {r.get('note', '')}")
        else:
            tag = " (hidden, not scored)" if r["hidden"] else ""
            verdict = "ok" if r["correct"] else f"-> {r['assigned']}"
            print(f"  {r['part']:>2d} {r['name']:<20s} expect={r['expected_kind']:<6s} "
                  f"fidelity={r['fidelity']:.2f} {verdict:<20s} dist={r['dist_median']:5.1f} "
                  f"margin={r['margin']:+6.1f} purity={r['purity']:.2f} snap60={r['fidelity_snap60']:.2f}{tag}")
    report = args.report or os.path.splitext(out_glb)[0] + "_fidelity.json"
    with open(report, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
