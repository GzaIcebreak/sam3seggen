"""Colour fidelity of an inference result against the 2D map that conditioned it.

For every GT part: sample surface points, look up the colour the output GLB baked there,
snap it to the variant's palette (+ GREY, + "other"), and compare with the colour the map
assigned to that part (grey_parts expect GREY). Also reports purity = share of a part's
samples carrying its dominant colour (1.0 = one solid colour, low = fragmented).

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
    uv = mesh.visual.uv[mesh.faces[face_idx]]
    p_uv = (bary[:, :, None] * uv).sum(1)
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
        d2 = ((colors[:, None, :] - pal[None]) ** 2).sum(-1)
        idx = d2.argmin(1)
        idx[np.sqrt(d2[np.arange(len(idx)), idx]) > snap_dist] = -1  # "other": not a palette colour
        counts = np.bincount(idx + 1, minlength=len(palette) + 1)
        dominant = int(counts.argmax()) - 1
        exp = expected.get(p)
        exp_idx = palette.index(exp) if exp in palette else None
        fidelity = float(counts[exp_idx + 1] / counts.sum()) if exp_idx is not None else None
        rows.append({
            "part": p, "name": names[p], "samples": int(keep.sum()),
            "expected": list(exp) if exp else None, "expected_kind": "grey" if exp == GREY else ("colour" if exp else "none"),
            "dominant": list(palette[dominant]) if dominant >= 0 else "other",
            "fidelity": fidelity, "purity": float(counts.max() / counts.sum()),
            "other_share": float(counts[0] / counts.sum()),
        })
    hidden = set(meta.get("hidden_parts", []))
    for r in rows:
        r["hidden"] = r["part"] in hidden
    scored = [r for r in rows if r.get("fidelity") is not None and not r["hidden"]]
    hidden_rows = [r for r in rows if r.get("fidelity") is not None and r["hidden"]]
    summary = {
        "variant": variant, "kind": meta["kind"], "n_parts": len(parts), "n_visible_scored": len(scored),
        "output_frame": frame, "surface_coverage": coverage,
        "mean_fidelity": float(np.mean([r["fidelity"] for r in scored])) if scored else None,
        "parts_correct(>=0.8)": int(sum(r["fidelity"] >= 0.8 for r in scored)),
        "mean_purity": float(np.mean([r["purity"] for r in scored])) if scored else None,
        "mean_other_share": float(np.mean([r["other_share"] for r in scored])) if scored else None,
        "grey_parts_expected": len(meta["grey_parts"]),
        "hidden_parts": sorted(hidden),
        "hidden_mean_purity": float(np.mean([r["purity"] for r in hidden_rows])) if hidden_rows else None,
    }
    return {"summary": summary, "parts": rows}


def run_inference(obj: ObjectDir, variant: str, ckpt: str, out_glb: str, guidance: float | None) -> None:
    py = sys.executable
    cmd = [py, os.path.join(common.ROOT, "inference_full.py"), "--ckpt_path", ckpt, "--glb", obj.input_glb,
           "--input_vxz", obj.input_vxz, "--img", os.path.join(obj.variant_dir(variant), "map.png"),
           "--export_glb", out_glb, "--two_d_map"]
    subprocess.run(cmd, check=True, cwd=common.ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--object", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output_glb", default=None)
    parser.add_argument("--run_inference", action="store_true")
    parser.add_argument("--ckpt", default=os.path.join(common.ROOT, "ckpt", "full_seg_w_2d_map.ckpt"))
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    os.chdir(common.ROOT)
    obj = ObjectDir(args.object)
    out_glb = args.output_glb or os.path.join(obj.variant_dir(args.variant), f"infer_{os.path.splitext(os.path.basename(args.ckpt))[0]}.glb")
    if args.run_inference or not os.path.exists(out_glb):
        run_inference(obj, args.variant, args.ckpt, out_glb, None)
    result = evaluate(obj, args.variant, out_glb)
    print(json.dumps(result["summary"], indent=2))
    for r in result["parts"]:
        if r.get("fidelity") is None:
            print(f"  {r['part']:>2d} {r['name']:<20s} {r.get('note', '')}")
        else:
            tag = " (hidden, not scored)" if r["hidden"] else ""
            print(f"  {r['part']:>2d} {r['name']:<20s} expect={r['expected_kind']:<6s} fidelity={r['fidelity']:.2f} "
                  f"purity={r['purity']:.2f} other={r['other_share']:.2f} dominant={r['dominant']}{tag}")
    report = args.report or os.path.splitext(out_glb)[0] + "_fidelity.json"
    with open(report, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
