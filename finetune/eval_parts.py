"""Class-agnostic 3D part segmentation metric shared by SegviGen outputs and external face labels.

    finetune\run_ft.bat eval_parts.py --object E:\...\pv\<id> --segvigen <infer.glb> --variant sam3_az0 [--report r.json]
    finetune\run_ft.bat eval_parts.py --object E:\...\pv\<id> --segvigen <infer.glb> --legend <map_legend.json>
    finetune\run_ft.bat eval_parts.py --object E:\...\pv\<id> --faces <geosam2.glb> --face_labels <geosam2.npy>
                                      --labels_json <labels.json> [--unlabelled 0 999] [--report r.json]

Why a new metric: eval_fidelity scores "did the output keep the 2D map's colours" on visible parts only.
The goal is part ownership, so here every GT part is scored, hidden ones included, and a prediction is
just a partition of the surface (what a cutter would get), name optional:

  * Uniform-density surface samples over the GT parts (each part >= 40 pts) carry the GT part index;
    the prediction is read at the nearest predicted-surface point (dist <= 0.02 in the unit-cube frame),
    as a face label (external) or the nearest palette colour of the baked texture (SegviGen).
  * mIoU: for every GT part, the best IoU with any predicted segment (unlabelled / grey counts as a
    segment too: an unnamed blob that exactly covers a part is still a correct cut); mIoU_matched uses
    a one-to-one Hungarian assignment so merging two parts into one segment is paid for twice.
  * over_seg: GT parts split over >= 2 segments each holding >= 20% of the part; under_seg: segments
    spanning >= 2 GT parts each >= 20% of the segment.
  * boundary_f1: eval_fidelity's point-based boundary F1, on labels (class-agnostic).
  * name_acc: GT parts whose majority segment carries the GT name (case-insensitive); name_acc_named
    restricts to parts whose majority segment has a name at all. unlabelled_share: sample share whose
    prediction is grey / unlabelled / uncovered.
  * sem_miou: per unique GT name, IoU of the GT region (all parts of that name, so two boots are one
    region) against the union of predicted segments carrying that name; unnamed predictions are
    background. This is the semantic-segmentation view and the one a name-prompted pipeline is fair on.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common import GREY, ObjectDir  # noqa: E402
from eval_fidelity import align_output, boundary_f1, load_output, normalize_like_vxz, texture_lookup  # noqa: E402

UNCOVERED = -1
UNLABELLED = -2


def gt_samples(obj: ObjectDir, n_total: int = 60000, min_per_part: int = 40):
    with open(obj.ids_meta, "r", encoding="utf-8") as f:
        aabb = np.asarray(json.load(f)["aabb"])
    parts = [normalize_like_vxz(p, aabb) for p in common.load_parts(obj)]
    total = sum(p.area for p in parts)
    density = n_total / max(total, 1e-9)
    pts, lab = [], []
    for i, p in enumerate(parts):
        n = max(min_per_part, int(p.area * density))
        s, _ = trimesh.sample.sample_surface(p, n)
        pts.append(s)
        lab.append(np.full(len(s), i))
    whole = trimesh.util.concatenate(parts)
    ref, _ = trimesh.sample.sample_surface(whole, 2000)
    return np.concatenate(pts), np.concatenate(lab), float(np.sqrt(1.0 / density)), ref, len(parts)


class FaceLabelPred:
    """External method: mesh + one integer label per face (+ optional id -> name)."""

    def __init__(self, glb: str, labels_npy: str, names: dict[int, str], unlabelled: set[int]):
        loaded = trimesh.load(glb, force="scene")
        meshes = [m for m in loaded.dump() if isinstance(m, trimesh.Trimesh)]
        self.mesh = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
        self.face_labels = np.rint(np.load(labels_npy)).astype(np.int64)
        if len(self.face_labels) != len(self.mesh.faces):
            raise ValueError(f"{len(self.face_labels)} labels for {len(self.mesh.faces)} faces")
        self.names = names
        self.unlabelled = unlabelled

    def labels_at(self, mesh: trimesh.Trimesh, face_idx: np.ndarray, closest: np.ndarray) -> np.ndarray:
        lab = self.face_labels[face_idx].copy()
        for u in self.unlabelled:
            lab[lab == u] = UNLABELLED
        return lab

    def name_of(self, label: int) -> str | None:
        return self.names.get(int(label))


class TexturePred:
    """SegviGen: textured mesh, label = nearest palette colour; grey = unlabelled."""

    def __init__(self, glb: str, palette: list[tuple[int, int, int]], names: dict[int, str]):
        self.mesh = load_output(glb)
        self.pal = np.asarray(palette, dtype=np.float32)
        self.grey = palette.index(GREY) if GREY in palette else None
        self.names = names

    def labels_at(self, mesh: trimesh.Trimesh, face_idx: np.ndarray, closest: np.ndarray) -> np.ndarray:
        colors = texture_lookup(mesh, face_idx, closest)
        lab = np.sqrt(((colors[:, None, :] - self.pal[None]) ** 2).sum(-1)).argmin(1)
        if self.grey is not None:
            lab[lab == self.grey] = UNLABELLED
        return lab

    def name_of(self, label: int) -> str | None:
        return self.names.get(int(label))


def segvigen_palette(obj: ObjectDir, variant: str):
    """Colour index -> name from the variant meta (group -> member parts -> GT names)."""
    with open(os.path.join(obj.variant_dir(variant), "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    gt_names = obj.names() or []
    palette = [tuple(c) for c in meta["colors"]] + [GREY]
    names = {}
    for g, members in enumerate(meta["groups"]):
        member_names = sorted({gt_names[p] for p in members if p < len(gt_names)})
        names[g] = member_names[0] if len(member_names) == 1 else "+".join(member_names)
    return palette, names, meta


def legend_palette(legend_json: str):
    """Colour index -> name from a sam3_to_2dmap legend (deployment path, no GT binding)."""
    with open(legend_json, "r", encoding="utf-8") as f:
        legend = json.load(f)
    palette, names = [], {}
    for row in legend:
        c = tuple(int(x) for x in row["color"])
        if c == GREY or row.get("part") == "<unassigned>":
            continue
        names[len(palette)] = row.get("part") or row["prompt"]
        palette.append(c)
    palette.append(GREY)
    return palette, names


def norm_name(s: str | None) -> str:
    return (s or "").strip().lower()


def name_matches(pred_name: str | None, gt_name: str) -> bool:
    """Path A groups may carry several GT names joined by '+'; any member counts."""
    if not pred_name:
        return False
    return norm_name(gt_name) in {norm_name(n) for n in pred_name.split("+")}


def semantic_iou(pred, pred_lab: np.ndarray, gt: np.ndarray, names: list[str]) -> dict:
    """Per unique GT name: IoU between the GT region (all parts with that name) and the union of predicted
    segments carrying that name. Duplicate parts (two boots) are one region here, so a single 'boot' mask
    is not penalised; unnamed predictions count as background."""
    seg_names = {}
    for s in np.unique(pred_lab):
        if s >= 0:
            n = pred.name_of(int(s))
            seg_names[int(s)] = {norm_name(x) for x in n.split("+")} if n else set()
    out = {}
    for name in sorted({norm_name(n) for n in names}):
        gt_region = np.isin(gt, [i for i, n in enumerate(names) if norm_name(n) == name])
        pred_region = np.isin(pred_lab, [s for s, ns in seg_names.items() if name in ns])
        union = (gt_region | pred_region).sum()
        out[name] = float((gt_region & pred_region).sum() / union) if union else 0.0
    return out


def evaluate(obj: ObjectDir, pred, seed: int = 0) -> dict:
    np.random.seed(seed)
    pts, gt, spacing, ref, n_parts = gt_samples(obj)
    mesh, frame, coverage = align_output(pred.mesh, ref)
    closest, dist, face_idx = trimesh.proximity.ProximityQuery(mesh).on_surface(pts)
    pred_lab = pred.labels_at(mesh, face_idx, closest)
    pred_lab = np.where(dist <= 0.02, pred_lab, UNCOVERED)

    segs = [s for s in np.unique(pred_lab) if s != UNCOVERED]
    seg_index = {s: j for j, s in enumerate(segs)}
    # contingency table GT part x predicted segment
    table = np.zeros((n_parts, len(segs)), dtype=np.int64)
    for s, j in seg_index.items():
        sel = pred_lab == s
        table[:, j] = np.bincount(gt[sel], minlength=n_parts)
    gt_size = np.bincount(gt, minlength=n_parts)
    seg_size = table.sum(0)
    union = gt_size[:, None] + seg_size[None, :] - table
    iou = np.where(union > 0, table / np.maximum(union, 1), 0.0)

    best_iou = iou.max(1) if len(segs) else np.zeros(n_parts)
    if len(segs):
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(-iou)
        matched = np.zeros(n_parts)
        matched[r] = iou[r, c]
    else:
        matched = np.zeros(n_parts)

    names = obj.names() or [str(i) for i in range(n_parts)]
    rows = []
    for p in range(n_parts):
        share = table[p] / max(1, gt_size[p])
        maj = int(share.argmax()) if len(segs) else None
        maj_seg = segs[maj] if maj is not None else None
        pred_name = pred.name_of(maj_seg) if maj_seg is not None and maj_seg >= 0 else None
        rows.append({
            "part": p, "name": names[p], "samples": int(gt_size[p]),
            "best_iou": float(best_iou[p]), "matched_iou": float(matched[p]),
            "majority_segment": None if maj_seg is None else int(maj_seg), "majority_share": float(share.max()) if len(segs) else 0.0,
            "majority_name": pred_name,
            "name_correct": name_matches(pred_name, names[p]),
            "n_segments_20pct": int((share >= 0.2).sum()),
            "uncovered_share": float((pred_lab[gt == p] == UNCOVERED).mean()) if gt_size[p] else 1.0,
            "unlabelled_share": float((pred_lab[gt == p] == UNLABELLED).mean()) if gt_size[p] else 1.0,
        })
    seg_share = table / np.maximum(seg_size, 1)[None, :]
    under = int(((seg_share >= 0.2).sum(0) >= 2).sum()) if len(segs) else 0
    # small parts: under 1% of the sampled surface (the audit's "面积<1%的小件召回")
    area_share = gt_size / max(1, gt_size.sum())
    small = np.nonzero(area_share < 0.01)[0]
    for p in range(n_parts):
        rows[p]["area_share"] = float(area_share[p])
        rows[p]["small"] = bool(area_share[p] < 0.01)
    named_rows = [r for r in rows if r["majority_name"]]
    bnd = boundary_f1(pts, gt, pred_lab, spacing) if len(pts) else {"boundary_f1": None}
    real_segs = [s for s in segs if s >= 0]
    sem = semantic_iou(pred, pred_lab, gt, names)
    summary = {
        "n_parts": n_parts, "n_pred_segments": len(real_segs), "n_names": len(sem),
        "miou": float(best_iou.mean()), "miou_matched": float(matched.mean()),
        "sem_miou": float(np.mean(list(sem.values()))) if sem else None,
        "names_iou>=0.5": int(sum(v >= 0.5 for v in sem.values())),
        "parts_iou>=0.5": int((best_iou >= 0.5).sum()), "parts_iou>=0.75": int((best_iou >= 0.75).sum()),
        "n_small_parts": int(len(small)),
        "small_part_recall": float((best_iou[small] >= 0.5).mean()) if len(small) else None,
        "over_seg_parts": int(sum(r["n_segments_20pct"] >= 2 for r in rows)),
        "under_seg_segments": under,
        "name_acc": float(np.mean([r["name_correct"] for r in rows])),
        "name_acc_named": float(np.mean([r["name_correct"] for r in named_rows])) if named_rows else None,
        "parts_named": len(named_rows),
        "unlabelled_share": float((pred_lab == UNLABELLED).mean()),
        "uncovered_share": float((pred_lab == UNCOVERED).mean()),
        "boundary_f1": bnd.get("boundary_f1"), "boundary_precision": bnd.get("boundary_precision"),
        "boundary_recall": bnd.get("boundary_recall"),
        "output_frame": frame, "surface_coverage": coverage, "n_samples": int(len(pts)),
    }
    return {"summary": summary, "parts": rows, "semantic": sem}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--object", required=True)
    ap.add_argument("--segvigen", default=None, help="SegviGen output GLB (textured)")
    ap.add_argument("--variant", default=None, help="Variant whose meta gives the palette / names for --segvigen")
    ap.add_argument("--legend", default=None, help="sam3_to_2dmap legend JSON giving the palette / names instead")
    ap.add_argument("--faces", default=None, help="External labelled mesh (GLB)")
    ap.add_argument("--face_labels", default=None, help="npy with one label per face of --faces")
    ap.add_argument("--labels_json", default=None, help='{"ids": {"1": "helmet", ...}} (geosam2_run.py)')
    ap.add_argument("--unlabelled", type=int, nargs="*", default=[0, 999], help="Face label values meaning 'no label'")
    ap.add_argument("--report", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    obj = ObjectDir(args.object)
    if args.segvigen:
        if args.legend:
            palette, names = legend_palette(args.legend)
        elif args.variant:
            palette, names, _ = segvigen_palette(obj, args.variant)
        else:
            ap.error("--segvigen needs --variant or --legend")
        pred = TexturePred(args.segvigen, palette, names)
    elif args.faces:
        names = {}
        if args.labels_json:
            with open(args.labels_json, "r", encoding="utf-8") as f:
                names = {int(k): v for k, v in json.load(f)["ids"].items()}
        pred = FaceLabelPred(args.faces, args.face_labels, names, set(args.unlabelled))
    else:
        ap.error("give --segvigen or --faces")
    result = evaluate(obj, pred)
    if not args.quiet:
        print(json.dumps(result["summary"], indent=2))
        for r in result["parts"]:
            print(f"  {r['part']:>2d} {r['name']:<18s} iou={r['best_iou']:.2f} matched={r['matched_iou']:.2f} "
                  f"maj={str(r['majority_name']):<18s} {'ok' if r['name_correct'] else '  '} "
                  f"split={r['n_segments_20pct']} unlab={r['unlabelled_share']:.2f}")
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
