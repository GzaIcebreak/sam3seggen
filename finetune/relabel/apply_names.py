"""Write names_v2.json back into <root>/<object_id>/names.json, keeping the original as names_v1.json.

Also writes names_meta.json ({source, object, uncertain[]}), which SegviGen's training and the SAM3
prompt builder read. Objects whose part count disagrees with the merged record are skipped and
printed rather than force-written: a mismatch means the object's captions.json and names.json were
already inconsistent at import time, and it should be dropped from the set instead of patched.

names_v1.json is only created when absent, so re-running never destroys the rollback copy.

    python finetune/relabel/apply_names.py --root /data/pv_new --names /data/relabel_new/names_v2.json --dry
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--names", required=True, help="names_v2.json from check_names.py --all")
    ap.add_argument("--source", default="grok-4.6 relabel from PartVerse captions",
                    help="recorded in names_meta.json so a later reader can tell the provenance")
    ap.add_argument("--reset_sam3", action="store_true",
                    help="also delete sam3_* variants and sam3_masks.npz so they regenerate with the new prompts")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    with open(args.names, encoding="utf-8") as f:
        v2 = json.load(f)

    written = changed = mismatch = reset_variants = reset_masks = 0
    with_sam3 = []
    for oid, rec in v2.items():
        odir = os.path.join(args.root, oid)
        np_ = os.path.join(odir, "names.json")
        if not os.path.exists(np_):
            print(f"!! {oid[:8]}: no names.json under {args.root}; skipping")
            mismatch += 1
            continue
        with open(np_, encoding="utf-8") as f:
            old = json.load(f)
        new = rec["names"]
        if len(old) != len(new):
            print(f"!! {oid[:8]}: old has {len(old)} names, new has {len(new)}; skipping")
            mismatch += 1
            continue
        changed += sum(1 for a, b in zip(old, new) if a != b)
        if not args.dry:
            bak = os.path.join(odir, "names_v1.json")
            if not os.path.exists(bak):
                shutil.copyfile(np_, bak)
            with open(np_, "w", encoding="utf-8") as f:
                json.dump(new, f, ensure_ascii=False, indent=2)
            with open(os.path.join(odir, "names_meta.json"), "w", encoding="utf-8") as f:
                json.dump({"source": args.source, "object": rec["object"],
                           "uncertain": rec["uncertain"]}, f, ensure_ascii=False, indent=2)
        written += 1

        sam3 = glob.glob(os.path.join(odir, "variants", "sam3_*"))
        if sam3:
            with_sam3.append(oid)
            if args.reset_sam3:
                for v in sam3:
                    if not args.dry:
                        shutil.rmtree(v)
                    reset_variants += 1
                for m in glob.glob(os.path.join(odir, "views", "*", "sam3_masks.npz")):
                    if not args.dry:
                        os.remove(m)
                    reset_masks += 1

    tag = "[dry] " if args.dry else ""
    print(f"{tag}names.json written: {written}; part names changed: {changed}; skipped: {mismatch}")
    print(f"{tag}objects with sam3 variants: {len(with_sam3)}; "
          f"reset variants: {reset_variants}, reset masks: {reset_masks}")
    if with_sam3 and not args.dry:
        lst = os.path.join(os.path.dirname(os.path.abspath(args.names)), "sam3_objects.txt")
        with open(lst, "w", encoding="utf-8") as f:
            f.write("\n".join(with_sam3) + "\n")
        print(f"{tag}-> {lst}")


if __name__ == "__main__":
    main()
