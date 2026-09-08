"""Pack a relabel work package and optionally push it to a Hugging Face dataset repo.

Two archives, deliberately split by what each stage needs:
  batches.tar.gz      the batch_NNN.json files. Labelling is text-in / text-out, so this is all a
                      labelling agent needs -- 4.4 MB for 5826 objects.
  review_views.tar.gz names.json + captions.json + one view's render.png and ids.npy per object.
                      Only the review stage needs it. az0 alone is what review_sheet.py and
                      screen_large.py read by default, and dropping az135, parts/*.glb and
                      input.glb takes 5826 objects from 39 GB to 1.1 GB packed.

    python finetune/relabel/pack_work_package.py --root .../pv_new --batches .../relabel_new \
        --out .../hf_relabel --wait --repo Zaun1996/segvigen-relabel-work

--wait blocks until every object under --root has the view files, so this can be started while the
renderer is still going and left to finish on its own.
"""
from __future__ import annotations

import argparse
import glob
import os
import tarfile
import time

PER_OBJECT = ["names.json", "names_meta.json", "captions.json"]
PER_VIEW = ["render.png", "ids.npy"]


def objects(root: str) -> list[str]:
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


def ready(root: str, oid: str, az: str) -> bool:
    return os.path.exists(os.path.join(root, oid, "views", az, "ids.npy"))


def wait_for_renders(root: str, az: str, poll_s: int = 60) -> None:
    total = len(objects(root))
    while True:
        done = sum(1 for o in objects(root) if ready(root, o, az))
        print(f"[pack] renders {done}/{total}", flush=True)
        if done >= total:
            return
        time.sleep(poll_s)


def pack_batches(batches: str, out: str) -> str:
    path = os.path.join(out, "batches.tar.gz")
    files = sorted(glob.glob(os.path.join(batches, "batch_[0-9]*.json")))
    with tarfile.open(path, "w:gz", compresslevel=6) as tf:
        for f in files:
            tf.add(f, arcname=os.path.basename(f))
    print(f"[pack] {len(files)} batches -> {path} ({os.path.getsize(path) / 1e6:.1f} MB)", flush=True)
    return path


def pack_views(root: str, out: str, azs: list[str], name: str) -> str:
    """Per-object text plus the listed views. One view is enough to review names; the concept bank
    trains on two, so the same function builds both packages."""
    path = os.path.join(out, name)
    base = os.path.basename(root.rstrip("\\/"))
    n = 0
    with tarfile.open(path, "w:gz", compresslevel=6) as tf:
        for oid in objects(root):
            if not all(ready(root, oid, a) for a in azs):
                continue
            rels = list(PER_OBJECT) + [f"views/{a}/{f}" for a in azs for f in PER_VIEW]
            for rel in rels:
                p = os.path.join(root, oid, *rel.split("/"))
                if os.path.exists(p):
                    tf.add(p, arcname=f"{base}/{oid}/{rel}")
            n += 1
    print(f"[pack] {n} objects, views {','.join(azs)} -> {path} "
          f"({os.path.getsize(path) / 1e9:.2f} GB)", flush=True)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset root, e.g. datasets/pv_new")
    ap.add_argument("--batches", default=None, help="directory with batch_NNN.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--az", default="az0", help="comma-separated views to include")
    ap.add_argument("--name", default="review_views.tar.gz", help="archive filename")
    ap.add_argument("--wait", action="store_true", help="block until every object is rendered")
    ap.add_argument("--repo", default=None, help="HF dataset repo to upload to; needs HF_TOKEN")
    ap.add_argument("--skip_views", action="store_true")
    args = ap.parse_args()

    azs = [a.strip() for a in args.az.split(",") if a.strip()]
    os.makedirs(args.out, exist_ok=True)
    if args.wait:
        for a in azs:
            wait_for_renders(args.root, a)

    files = []
    if args.batches:
        files.append(pack_batches(args.batches, args.out))
    if not args.skip_views:
        files.append(pack_views(args.root, args.out, azs, args.name))

    if args.repo:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
        for f in files:
            print(f"[pack] uploading {os.path.basename(f)}", flush=True)
            api.upload_file(path_or_fileobj=f, path_in_repo=os.path.basename(f),
                            repo_id=args.repo, repo_type="dataset",
                            commit_message=f"Add {os.path.basename(f)}")
        print(f"[pack] uploaded to https://huggingface.co/datasets/{args.repo}", flush=True)
    print("PACK_DONE", flush=True)


if __name__ == "__main__":
    main()
