"""Validate relabel outputs against their batches, and merge the good ones into names_v2.json.

Two modes:
    --batch/--out   one pair, verbose; use while a labelling agent is still iterating.
                    Add --dump for a side-by-side of new name / old name / caption.
    --all           every batch_NNN/out_NNN pair in --dir, then write <dir>/names_v2.json.

Exit code is non-zero when anything failed validation, so a driver script can stop on it.
`problems: 0` and `bad batches: 0` is the bar; do not run apply_names.py before both are met.

    python finetune/relabel/check_names.py --dir /data/relabel_new --all
    python finetune/relabel/check_names.py --dir /data/relabel_new --batch 007 --dump
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rules import check_name, normalise, words, JUNK


def validate(inp: dict, out: dict) -> tuple[list[str], dict]:
    """Rule violations plus counts. `out` is the labelling agent's file, `inp` its batch."""
    problems = []
    stats = {"objects": len(inp), "parts": 0, "uncertain": 0, "junk": 0, "changed": 0}
    for oid in inp:
        if oid not in out:
            problems.append(f"{oid[:8]}: object missing from output")
    for oid in out:
        if oid not in inp:
            problems.append(f"{oid[:8]}: object not in batch")

    for oid, rec in inp.items():
        o = out.get(oid)
        if not isinstance(o, dict) or not isinstance(o.get("parts"), dict):
            continue
        if not isinstance(o.get("object"), str) or not o["object"].strip():
            problems.append(f"{oid[:8]}: missing whole-object name")
        got = o["parts"]
        for pid in rec["part_order"]:
            stats["parts"] += 1
            e = got.get(pid)
            if not isinstance(e, dict):
                problems.append(f"{oid[:8]}/{pid}: part missing in output")
                continue
            if "uncertain" not in e:
                problems.append(f"{oid[:8]}/{pid}: uncertain flag missing")
            if e.get("uncertain"):
                stats["uncertain"] += 1
            name = e.get("name")
            for p in check_name(name):
                problems.append(f"{oid[:8]}/{pid}: {p}")
            if isinstance(name, str):
                if any(w in JUNK for w in words(name)):
                    stats["junk"] += 1
                if normalise(name) != normalise(rec["parts"][pid]["current_name"]):
                    stats["changed"] += 1
        for pid in got:
            if pid not in rec["parts"]:
                problems.append(f"{oid[:8]}/{pid}: extra part in output")
    return problems, stats


def dump(inp: dict, out: dict) -> None:
    print("\n" + "=" * 100)
    for oid, rec in inp.items():
        o = out.get(oid, {})
        print(f"\n### {oid[:8]}  object = {o.get('object')}")
        for pid in rec["part_order"]:
            p = rec["parts"][pid]
            e = o.get("parts", {}).get(pid, {})
            flag = " (?)" if e.get("uncertain") else ""
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", p["long_caption"].strip()) if s.strip()]
            print(f"  [{pid:>3}] {str(e.get('name', '<none>')):<24}{flag:<4} was: {p['current_name']}")
            print(f"        short: {p['short_caption'][:110]}")
            print(f"        last : {(sents[-1] if sents else '')[:150]}")


def read(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="directory holding batch_NNN.json / out_NNN.json")
    ap.add_argument("--batch", default=None, help="single batch tag, e.g. 007")
    ap.add_argument("--all", action="store_true", help="check every pair and merge into names_v2.json")
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--names_out", default=None, help="default <dir>/names_v2.json")
    args = ap.parse_args()

    if args.batch:
        inp = read(os.path.join(args.dir, f"batch_{args.batch}.json"))
        out_path = os.path.join(args.dir, f"out_{args.batch}.json")
        if not os.path.exists(out_path):
            print(f"out_{args.batch}.json missing")
            return 1
        problems, st = validate(inp, read(out_path))
        print(f"batch {args.batch}: {st['objects']} objects, {st['parts']} parts, "
              f"{st['changed']} names changed, {st['uncertain']} uncertain, {st['junk']} junk-word")
        print(f"problems: {len(problems)}")
        for p in problems[:40]:
            print("  " + p)
        if args.dump:
            dump(inp, read(out_path))
        return 1 if problems else 0

    if not args.all:
        print("give --batch NNN or --all")
        return 2

    merged, bad, junk_hits = {}, [], []
    total = {"parts": 0, "uncertain": 0, "changed": 0}
    for bp in sorted(glob.glob(os.path.join(args.dir, "batch_[0-9]*.json"))):
        tag = re.search(r"batch_(\d+)\.json$", bp).group(1)
        op = os.path.join(args.dir, f"out_{tag}.json")
        if not os.path.exists(op):
            bad.append((tag, "missing"))
            continue
        try:
            out = read(op)
        except Exception as e:
            bad.append((tag, f"invalid json: {e}"))
            continue
        inp = read(bp)
        problems, st = validate(inp, out)
        if problems:
            bad.append((tag, f"{len(problems)} problems, e.g. {problems[:3]}"))
            continue
        for oid, rec in inp.items():
            o = out[oid]
            names, unc = [], []
            for pid in rec["part_order"]:
                e = o["parts"][pid]
                nm = normalise(e["name"])
                if any(w in JUNK for w in words(nm)):
                    junk_hits.append((tag, oid[:8], pid, nm))
                names.append(nm)
                unc.append(bool(e.get("uncertain", True)))
            merged[oid] = {"object": normalise(str(o.get("object", ""))), "names": names,
                           "uncertain": unc, "part_order": rec["part_order"]}
        for k in total:
            total[k] += st[k]

    names_out = args.names_out or os.path.join(args.dir, "names_v2.json")
    with open(names_out, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=1)

    print(f"merged {len(merged)} objects, {total['parts']} parts, {total['changed']} names changed, "
          f"{total['uncertain']} uncertain ({100 * total['uncertain'] / max(1, total['parts']):.1f}%)")
    print(f"junk-word names: {len(junk_hits)}  (re-read the caption before changing any of these)")
    for h in junk_hits[:15]:
        print("   ", h)
    print(f"bad batches: {len(bad)}")
    for b in bad:
        print("   ", b)
    print(f"-> {names_out}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
