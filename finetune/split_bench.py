"""Which (2D map painter x SegviGen ckpt x split mode) lands closest to a clean part split?

Runs entirely on the ext_bench lifts that already exist (datasets/ext_bench/<key>/seg_<tag>*.glb,
see lift_maps_segvigen.py) -- no GPU. For every asset, combo and seed it labels the SegviGen
colouring with each split mode of data_toolkit/parts_rebake.py and scores the result without
any ground truth:

    front_agree   pixel agreement with the conditioning-view guide map (the map SegviGen saw)
    back_agree    the same against the opposite view's map, which nothing downstream ever saw
    pieces        connected same-part pieces beyond one per part (fragmentation)
    frag_share    faces outside their part's largest piece
    stain_dev     faces whose label differs from SegviGen's raw colouring
    parts_found   prompts that end up with >= 1% of the faces / prompts asked for

    python finetune/split_bench.py                       # every asset, combo, seed, mode
    python finetune/split_bench.py --assets robot sword --modes stain weld
Writes datasets/ext_bench/split_bench.json and prints the per-combo mean table.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ext_bench import ASSETS, OUT, TRANSFORMS, front_json, map_paths, read_json, wd  # noqa: E402
from data_toolkit.parts_rebake import (  # noqa: E402
    SPLIT_MODES, _face_labels, assign_to_palette, face_base_colors, load_single_mesh,
    palette_from_legend, welded_face_adjacency,
)
from data_toolkit.project_2d import load_camera, project_points, segvigen_to_render_frame  # noqa: E402
from PIL import Image  # noqa: E402

# combo -> (seg tag written by lift_maps_segvigen.py, map variant it was conditioned on)
COMBOS = {
    "v3+base": ("base", "map"),
    "v3+v6": ("v3_v6", "map"),
    "ease+base": ("ease", "map_ease"),
    "ease+v6": ("ease_v6", "map_ease"),
    "gnn+base": ("rank", "map_rank"),
}
# segment_api --assign auto: the ranker's map unless it deleted a prompt v3 kept
AUTO = {"auto+base": ("v3+base", "ease+base"), "auto+v6": ("v3+v6", "ease+v6")}
UNASSIGNED = "<unassigned>"


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def seg_files(key: str, tag: str) -> list[tuple[str, str]]:
    d = wd(key)
    out = []
    if os.path.exists(os.path.join(d, f"seg_{tag}.glb")):
        out.append(("-", os.path.join(d, f"seg_{tag}.glb")))
    for p in sorted(glob.glob(os.path.join(d, f"seg_{tag}_s[0-9]*.glb"))):
        if not p.endswith("_upright.glb"):
            out.append((os.path.basename(p)[len(f"seg_{tag}_s"):-4], p))
    return out


def auto_pick(key: str) -> str:
    """'ease' when the ranker kept every prompt v3 painted, else 'v3'."""
    v3 = {e["prompt"] for e in read_json(map_paths(key, "map", "front")[1]) if e["prompt"] != UNASSIGNED}
    ease_legend = map_paths(key, "map_ease", "front")[1]
    if not os.path.exists(ease_legend):
        return "v3"
    ease = {e["prompt"] for e in read_json(ease_legend) if e["prompt"] != UNASSIGNED}
    return "ease" if v3 <= ease else "v3"


def pixel_winners(centroids, canvas_shape, cam):
    cam_pos, axes, focal, principal = cam
    height, width = canvas_shape[:2]
    u, v, depth = project_points(centroids, cam_pos, axes, focal, principal)
    inside = (depth > 1e-6) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    winner = np.full(height * width, -1, dtype=np.int64)
    if inside.any():
        ui = np.floor(u[inside]).astype(np.int64)
        vi = np.floor(v[inside]).astype(np.int64)
        fi = np.nonzero(inside)[0]
        order = np.argsort(-depth[inside])
        winner[(vi * width + ui)[order]] = fi[order]
    return winner.reshape(height, width)


def map_agreement(labels_part, part_names, centroids, map_png, legend_json, azimuth):
    """Share of guide-map pixels painted with a named prompt whose z-buffer face carries that part.

    Each map is read with its own legend (sam3_to_2dmap hands colours out in prompt order,
    so the back view's colours differ from the front's) and matched to the split by prompt
    name. Pixels the mesh does not cover count as disagreement (the map claims a part
    there); background and <unassigned> pixels are left out of the denominator.
    """
    palette, concepts, _ = palette_from_legend(legend_json)
    concept_to_part = np.array([part_names.index(c) if c in part_names else -1 for c in concepts])
    canvas = np.asarray(Image.open(map_png).convert("RGB"))
    cam = load_camera(TRANSFORMS, azimuth=azimuth, resolution=canvas.shape[0])
    winner = pixel_winners(centroids, canvas.shape, cam)
    rgb = canvas.reshape(-1, 3).astype(np.float64)
    foreground = ~np.all(rgb >= 250, axis=1)
    concept = np.argmin(np.linalg.norm(rgb[:, None, :] - palette[None], axis=2), axis=1)
    keep = foreground & (concept_to_part[concept] >= 0)
    if not keep.any():
        return float("nan")
    want = concept_to_part[concept[keep]]
    face = winner.reshape(-1)[keep]
    got = np.where(face >= 0, labels_part[np.maximum(face, 0)], -1)
    return float((got == want).mean())


def fragmentation(adjacency, labels, small=0.01):
    """(pieces beyond one per present part, share of faces sitting in pieces under `small`).

    A part legitimately comes in several pieces (two legs, four wheels), so only pieces
    too small to be a part of their own are what we call fragmentation.
    """
    from trimesh.graph import connected_components
    same = labels[adjacency[:, 0]] == labels[adjacency[:, 1]]
    pieces = connected_components(adjacency[same], nodes=np.arange(len(labels)))
    tiny = sum(len(p) for p in pieces if len(p) < small * len(labels))
    return len(pieces) - len(np.unique(labels)), tiny / len(labels)


def evaluate(key: str, seg: str, variant: str, modes: list[str], args) -> list[dict]:
    d = wd(key)
    az = float(front_json(key)["azimuth"])
    map_png, legend = map_paths(key, variant, "front")
    back_png, back_legend = map_paths(key, variant, "back")
    palette, concepts, parts = palette_from_legend(legend)
    part_names = list(dict.fromkeys(parts))
    concept_to_part = np.array([part_names.index(p) for p in parts])
    prompts = [p for p in ASSETS[key][1]]

    mesh = load_single_mesh(seg)
    centroids = segvigen_to_render_frame(np.asarray(mesh.vertices))[np.asarray(mesh.faces)].mean(axis=1)
    adjacency = np.asarray(welded_face_adjacency(mesh))
    stain = concept_to_part[assign_to_palette(face_base_colors(mesh), palette)]

    rows = []
    for mode in modes:
        labels, centers, names = _face_labels(
            mesh, palette=legend, two_d_map=map_png, transforms=TRANSFORMS, azimuth=az,
            split_mode=mode, min_fragment_faces=args.min_fragment_faces,
            min_island_ratio=args.min_island_ratio,
            weld_min_visible=args.weld_min_visible, weld_min_agreement=args.weld_min_agreement,
        )
        labels = np.asarray(labels)
        counts = np.bincount(labels, minlength=len(names))
        found = sum(1 for p in prompts if p in names and counts[names.index(p)] >= 0.01 * len(labels))
        extra_pieces, frag_share = fragmentation(adjacency, labels)
        rows.append({
            "mode": mode,
            "front_agree": map_agreement(labels, list(names), centroids, map_png, legend, az),
            "back_agree": map_agreement(labels, list(names), centroids, back_png, back_legend,
                                        (az + 180.0) % 360.0) if os.path.exists(back_legend) else float("nan"),
            "pieces": int(extra_pieces),
            "frag_share": float(frag_share),
            "stain_dev": float((labels != stain).mean()),
            "parts_found": found / len(prompts),
            "faces": int(len(labels)),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", nargs="*", default=None)
    ap.add_argument("--combos", nargs="*", default=list(COMBOS))
    ap.add_argument("--modes", nargs="*", default=list(SPLIT_MODES))
    ap.add_argument("--min_fragment_faces", type=int, default=100)
    ap.add_argument("--min_island_ratio", type=float, default=0.2)
    ap.add_argument("--weld_min_visible", type=float, default=0.25)
    ap.add_argument("--weld_min_agreement", type=float, default=0.6)
    ap.add_argument("--out", default=os.path.join(OUT, "split_bench.json"))
    args = ap.parse_args()
    keys = args.assets or list(ASSETS)

    rows = []
    for key in keys:
        pick = auto_pick(key)
        for combo in args.combos:
            tag, variant = COMBOS[combo]
            if not os.path.exists(map_paths(key, variant, "front")[1]):
                say(f"skip {key} {combo}: no {variant} legend")
                continue
            files = seg_files(key, tag)
            if not files:
                say(f"skip {key} {combo}: no seg_{tag}*.glb")
                continue
            for seed, seg in files:
                say(f"{key} {combo} seed={seed}")
                for r in evaluate(key, seg, variant, args.modes, args):
                    rows.append({"asset": key, "combo": combo, "seed": seed, "auto_pick": pick, **r})

    # synthesise the auto combos from the rows of the painter auto would have chosen
    for name, (v3_combo, ease_combo) in AUTO.items():
        for r in list(rows):
            if r["combo"] == (ease_combo if r["auto_pick"] == "ease" else v3_combo):
                rows.append({**r, "combo": name})

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "rows": rows}, f, ensure_ascii=False, indent=1)
    say(f"wrote {args.out} ({len(rows)} rows)")
    print(summary(rows))


METRICS = ("front_agree", "back_agree", "pieces", "frag_share", "stain_dev", "parts_found")


def summary(rows) -> str:
    """Mean over assets of the per-asset mean over seeds, one line per combo x mode."""
    out = ["", f"{'combo':<10} {'mode':<7} {'n':>3} " + " ".join(f"{m:>11}" for m in METRICS)]
    for combo in dict.fromkeys(r["combo"] for r in rows):
        for mode in dict.fromkeys(r["mode"] for r in rows):
            sel = [r for r in rows if r["combo"] == combo and r["mode"] == mode]
            if not sel:
                continue
            per_asset = {}
            for r in sel:
                per_asset.setdefault(r["asset"], []).append(r)
            means = []
            for m in METRICS:
                vals = [np.nanmean([r[m] for r in rs]) for rs in per_asset.values()]
                means.append(float(np.nanmean(vals)))
            out.append(f"{combo:<10} {mode:<7} {len(per_asset):>3} " + " ".join(f"{v:>11.3f}" for v in means))
    return "\n".join(out)


if __name__ == "__main__":
    main()
