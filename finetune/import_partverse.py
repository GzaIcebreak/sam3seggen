"""PartVerse -> sample object directories for make_samples_a/b.

Expected extracted layout (adjust with --glb_dir/--parts_dir/--anno_dir):
    <partverse>/normalized_glbs/<object_id>.glb
    <partverse>/anno_infos/<object_id>/<object_id>_{face2label.json,info.json,segmented.glb}
    <partverse>/textured_part_glbs/<object_id>/<part_id>.glb        (optional, 80 GB)
    <partverse>/text_captions.json     {object_id: {part_id: [short_caption, long_caption]}}

    python finetune/import_partverse.py --partverse E:/datasets/partverse --out E:/datasets/pv_samples \
        --limit 1500 --min_parts 3 --max_parts 24 --seed 0

Parts only need geometry (they are repainted with ID colours), so by default they are cut out of
the textured whole with anno_infos' face labels; labels are transferred by nearest face centroid
from segmented.glb when the face order does not match. Pass --use_part_glbs to copy the textured
part GLBs instead. SAM3 prompts are the head noun phrase of the short caption
("A metal spike extracted from a baseball bat." -> "metal spike").
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from common import natural_key


def whole_mesh(glb_path: str) -> trimesh.Trimesh:
    """Every geometry of the GLB baked into world space and concatenated (geometry only)."""
    scene = trimesh.load(glb_path, force="scene")
    meshes = []
    for node in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node]
        geom = scene.geometry[geom_name]
        if not isinstance(geom, trimesh.Trimesh) or len(geom.faces) == 0:
            continue
        m = trimesh.Trimesh(vertices=geom.vertices.copy(), faces=geom.faces.copy(), process=False)
        m.apply_transform(transform)
        meshes.append(m)
    if not meshes:
        raise ValueError(f"{glb_path}: no geometry")
    return trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]


def split_by_anno(glb_path: str, anno_dir: str, oid: str, max_centroid_dist: float = 0.02):
    """Return ({label: part_mesh}, stats) using anno_infos face labels."""
    prefix = os.path.join(anno_dir, oid, oid)
    with open(prefix + "_face2label.json", "r") as f:
        f2l = json.load(f)
    seg_labels = np.zeros(len(f2l), dtype=np.int64)
    for k, v in f2l.items():
        seg_labels[int(k)] = int(v)
    mesh = whole_mesh(glb_path)
    stats = {"faces": int(len(mesh.faces)), "anno_faces": int(len(f2l))}
    seg = trimesh.load(prefix + "_segmented.glb", force="mesh")
    seg_centroids = seg.triangles_center
    centroids = mesh.triangles_center
    same_order = len(mesh.faces) == len(seg_labels) and np.allclose(centroids, seg_centroids, atol=1e-4)
    if same_order:
        labels = seg_labels
        stats["mode"] = "direct"
    else:
        # Same normalised frame, different tessellation/order: nearest face centroid carries the label.
        dist, idx = cKDTree(seg_centroids).query(centroids, k=1)
        labels = seg_labels[idx]
        stats.update({"mode": "nearest", "median_dist": float(np.median(dist)), "max_dist": float(dist.max()),
                      "far_faces": int((dist > max_centroid_dist).sum())})
        if np.median(dist) > max_centroid_dist:
            raise ValueError(f"{oid}: segmented.glb does not align with the whole mesh (median centroid dist {np.median(dist):.4f})")
    parts = {}
    for label in np.unique(labels):
        sub = mesh.submesh([np.flatnonzero(labels == label)], append=True)
        parts[int(label)] = sub
    return parts, stats

_PREFIXES = tuple(sorted((
    "a close-up of the ", "a close-up of a ", "a close-up of an ", "a close-up view of the ", "a close-up view of a ",
    "a close-up view of an ", "a detailed view of the ", "a detailed view of a ", "a detailed close-up of the ",
    "a detailed close-up of a ", "a view of the ", "a view of a ", "a top view of the ", "a side view of the ",
    "a front view of the ", "a rear view of the ", "a bottom view of the ", "close-up of the ", "close-up view of the ",
    "the image shows ", "the image depicts ", "an image of ", "a rendering of ", "a 3d model of ", "a 3d rendering of ",
    "a pair of ", "a set of ", "a piece of ", "a portion of ", "a section of ", "a segment of ", "a fragment of ",
    "a part of ", "part of ", "the object is ", "this is ", "a ", "an ", "the ",
), key=len, reverse=True))
_CUTS = (" extracted from", " detached from", " removed from", " taken from", " separated from", " isolated from",
         " from ", " for ", " attached to", " belonging to", " on ", " that ", " which ", " with ", " in ", " used ",
         " highlighted", " highlighting", " showing", " featuring", " displaying", " revealing", " indicating",
         " emphasizing", " held ", " positioned", " located", " seen ", " viewed ", " likely", " possibly", " as ",
         ".", ";", ":")
_STOPWORDS = {"of", "and", "by", "its", "the", "a", "an", "with", "in", "on", "or", "to", "for", "from"}
_JUNK = {"component", "components", "close-up view", "close-up", "view", "structure", "representation", "object",
         "part", "piece", "section", "portion", "element", "item", "shape", "detail", "image", "model", "rendering",
         "geometric representation", "low-poly representation", "simplified geometric representation"}


def _noun_phrase(text: str, max_words: int) -> str:
    s = text.strip().lower()
    for pre in _PREFIXES:
        if s.startswith(pre):
            s = s[len(pre):]
            break
    m = _CLAUSE_AFTER_COMMA.search(s)
    if m:
        s = s[:m.start()]
    s = s.replace(",", "")
    cut = len(s)
    for c in _CUTS:
        i = s.find(c)
        if 0 < i < cut:
            cut = i
    s = s[:cut].strip()
    s = _TRAIL_OF.sub("", s).strip()
    s = re.sub(r"^(single|detached|isolated|individual|small|large|piece of|part of|one of the)\s+", "", s).strip()
    words = s.split()
    if " and " in f" {s} " and len(words) > max_words:
        words = s.split(" and ")[-1].split()
    if len(words) > max_words:
        words = words[-max_words:]
    while words and words[0] in _STOPWORDS:
        words = words[1:]
    while words and words[-1] in _STOPWORDS:
        words = words[:-1]
    return " ".join(words)
# A comma starts a new clause only when followed by one of these; otherwise it separates adjectives.
_CLAUSE_AFTER_COMMA = re.compile(
    r",\s*(which|that|likely|one\b|indicating|highlighting|shown|isolated|specifically|possibly|probably|"
    r"as\b|featuring|extracted|detached|removed|with|showing|part\b|highlighted|positioned|located|used|"
    r"designed|serving|typically|often|perhaps|and\b|or\b)")
_TRAIL_OF = re.compile(r"\s+of\s+(a|an|the|its|this)\b.*$", re.IGNORECASE)


def caption_to_prompt(short_caption: str, long_caption: str | None = None, max_words: int = 3) -> str:
    """Head noun phrase for SAM3; falls back to the long caption's first sentence when the short one is junk."""
    candidates = [short_caption]
    if long_caption:
        candidates.append(long_caption.split(". ")[0])
    for text in candidates:
        if not text:
            continue
        phrase = _noun_phrase(text, max_words)
        if phrase and phrase not in _JUNK and len(phrase) > 2:
            return phrase
    return "part"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--partverse", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--glb_dir", default=None)
    parser.add_argument("--parts_dir", default=None)
    parser.add_argument("--anno_dir", default=None)
    parser.add_argument("--captions", default=None)
    parser.add_argument("--use_part_glbs", action="store_true", help="Copy textured_part_glbs instead of cutting by face labels")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min_parts", type=int, default=2)
    parser.add_argument("--max_parts", type=int, default=32)
    parser.add_argument("--min_part_faces", type=int, default=8, help="Drop objects with a part smaller than this")
    parser.add_argument("--max_far_frac", type=float, default=0.1,
                        help="Skip objects where more than this fraction of faces has no nearby segmented.glb face")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ids", nargs="*", default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    glb_dir = args.glb_dir or os.path.join(args.partverse, "normalized_glbs")
    parts_dir = args.parts_dir or os.path.join(args.partverse, "textured_part_glbs")
    anno_dir = args.anno_dir or os.path.join(args.partverse, "anno_infos")
    captions_path = args.captions or os.path.join(args.partverse, "text_captions.json")
    with open(captions_path, "r", encoding="utf-8") as f:
        captions = json.load(f)

    whole = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(glb_dir, "**", "*.glb"), recursive=True)}
    ids = args.ids or sorted(set(whole) & set(captions))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(ids)
    print(f"{len(whole)} whole GLBs, {len(captions)} captioned, {len(ids)} usable")

    os.makedirs(args.out, exist_ok=True)
    kept, skipped, failed = 0, 0, 0
    report = {}
    for oid in ids:
        if args.limit and kept >= args.limit:
            break
        cap = captions[oid] if isinstance(captions[oid], dict) else {}
        odir = os.path.join(args.out, oid)
        if os.path.exists(os.path.join(odir, "names.json")):
            kept += 1
            continue
        try:
            if args.use_part_glbs:
                part_files = sorted(glob.glob(os.path.join(parts_dir, oid, "*.glb")), key=lambda p: natural_key(os.path.basename(p)))
                if not (args.min_parts <= len(part_files) <= args.max_parts):
                    skipped += 1
                    continue
                pids = [os.path.splitext(os.path.basename(p))[0] for p in part_files]
                parts = {pid: p for pid, p in zip(pids, part_files)}
                stats = {"mode": "part_glbs"}
            else:
                if not os.path.exists(os.path.join(anno_dir, oid, f"{oid}_face2label.json")):
                    skipped += 1
                    continue
                parts_by_label, stats = split_by_anno(whole[oid], anno_dir, oid)
                if not (args.min_parts <= len(parts_by_label) <= args.max_parts):
                    skipped += 1
                    continue
                if stats.get("far_faces", 0) > args.max_far_frac * stats["faces"]:
                    skipped += 1
                    continue
                if min(len(m.faces) for m in parts_by_label.values()) < args.min_part_faces:
                    skipped += 1
                    continue
                pids = [str(k) for k in sorted(parts_by_label)]
                parts = {str(k): m for k, m in parts_by_label.items()}
        except Exception as exc:
            failed += 1
            print(f"  fail {oid}: {exc!r}")
            continue

        names = []
        for pid in pids:
            entry = cap.get(pid) or [None, None]
            names.append(caption_to_prompt(entry[0], entry[1] if len(entry) > 1 else None) if entry[0] else "part")
        if args.dry_run:
            print(oid, len(pids), stats, names)
            kept += 1
            continue
        os.makedirs(os.path.join(odir, "parts"), exist_ok=True)
        shutil.copyfile(whole[oid], os.path.join(odir, "input.glb"))
        for k, pid in enumerate(pids):
            dst = os.path.join(odir, "parts", f"{k}.glb")
            if isinstance(parts[pid], str):
                shutil.copyfile(parts[pid], dst)
            else:
                parts[pid].export(dst)
        with open(os.path.join(odir, "names.json"), "w", encoding="utf-8") as f:
            json.dump(names, f, ensure_ascii=False, indent=2)
        with open(os.path.join(odir, "captions.json"), "w", encoding="utf-8") as f:
            json.dump({"source_part_ids": pids, "captions": {pid: cap.get(pid) for pid in pids}, "split": stats},
                      f, ensure_ascii=False, indent=2)
        report[oid] = stats
        kept += 1
        if kept % 50 == 0:
            print(f"  imported {kept} (skipped {skipped}, failed {failed})", flush=True)
    with open(os.path.join(args.out, "import_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"imported {kept} objects -> {args.out} (skipped {skipped}, failed {failed})")


if __name__ == "__main__":
    main()
