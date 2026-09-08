"""Build relabel input batches: per object, the PartVerse captions plus the current placeholder name.

Each batch is a self-contained JSON of ~50 objects; the labelling agent reads one batch and writes
one output file. The batch carries no image or mesh data -- captions and names only -- so a batch is
about 200 KB and the labelling step needs nothing but text.

    python finetune/relabel/prep_batches.py --root /data/pv_new --out /data/relabel_new --chunk 50
"""
from __future__ import annotations

import argparse
import json
import os


def load(root: str, oid: str) -> dict | None:
    odir = os.path.join(root, oid)
    try:
        with open(os.path.join(odir, "captions.json"), encoding="utf-8") as f:
            cap = json.load(f)
        with open(os.path.join(odir, "names.json"), encoding="utf-8") as f:
            names = json.load(f)
    except FileNotFoundError:
        return None
    pids = cap["source_part_ids"]
    parts = {}
    for i, pid in enumerate(pids):
        c = cap["captions"].get(pid) or ["", ""]
        parts[pid] = {
            "current_name": names[i] if i < len(names) else "",
            "short_caption": c[0] if len(c) > 0 else "",
            "long_caption": c[1] if len(c) > 1 else "",
        }
    return {"part_order": pids, "parts": parts}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset root holding <object_id>/ directories")
    ap.add_argument("--out", required=True, help="where batch_NNN.json go")
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--ids_file", default=None, help="restrict to these object ids (one per line)")
    ap.add_argument("--start", type=int, default=0, help="first batch number, to append to an existing set")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.ids_file:
        with open(args.ids_file, encoding="utf-8") as f:
            ids = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    else:
        ids = sorted(d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d)))

    recs, skipped = {}, []
    for oid in ids:
        r = load(args.root, oid)
        (recs.__setitem__(oid, r) if r else skipped.append(oid))
    ordered = sorted(recs)

    n_batches = 0
    for k in range(0, len(ordered), args.chunk):
        tag = args.start + k // args.chunk
        batch = {o: recs[o] for o in ordered[k:k + args.chunk]}
        with open(os.path.join(args.out, f"batch_{tag:03d}.json"), "w", encoding="utf-8") as f:
            json.dump(batch, f, ensure_ascii=False, indent=1)
        n_batches += 1

    parts = sum(len(r["parts"]) for r in recs.values())
    print(f"{len(ordered)} objects, {parts} parts -> {n_batches} batches of {args.chunk} in {args.out}")
    if skipped:
        print(f"skipped {len(skipped)} objects without captions.json/names.json, e.g. {skipped[:5]}")


if __name__ == "__main__":
    main()
