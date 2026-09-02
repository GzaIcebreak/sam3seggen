"""Resumable download of the PartVerse dataset (dscdyc/partverse, MIT) via HF or a mirror.

    python finetune/download_partverse.py --out E:/datasets/partverse --files captions anno glbs parts

Groups: captions (text_captions.json), anno (anno_infos.tar.gz, 2.8 GB), glbs (normalized_glbs,
20 GB in 2 parts), parts (textured_part_glbs, 80 GB in 8 parts). Split archives are joined
into one .tar.gz once every part is complete; extract with  tar -xzf <name>.tar.gz .
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

import requests

REPO = "datasets/dscdyc/partverse"
GROUPS = {
    "captions": ["text_captions.json", "README.md"],
    "anno": ["anno_infos.tar.gz"],
    "glbs": [f"normalized_glbs.tar.gz.{i:02d}" for i in range(2)],
    "parts": [f"textured_part_glbs.tar.gz.{i:02d}" for i in range(8)],
}


def remote_size(url: str) -> int | None:
    r = requests.head(url, allow_redirects=True, timeout=60)
    if r.status_code >= 400:
        return None
    size = r.headers.get("Content-Length")
    return int(size) if size else None


def download(url: str, dest: str, retries: int = 20, chunk: int = 1 << 20) -> None:
    total = remote_size(url)
    done_marker = dest + ".done"
    if os.path.exists(done_marker) and (total is None or os.path.getsize(dest) == total):
        print(f"skip {os.path.basename(dest)} (complete)")
        return
    part = dest + ".part"
    for attempt in range(retries):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if total is not None and have >= total:
            break
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=120, allow_redirects=True) as r:
                if r.status_code == 416:
                    break
                r.raise_for_status()
                mode = "ab" if have and r.status_code == 206 else "wb"
                if mode == "wb":
                    have = 0
                t0, last = time.time(), have
                with open(part, mode) as f:
                    for block in r.iter_content(chunk_size=chunk):
                        f.write(block)
                        have += len(block)
                        if time.time() - t0 > 30:
                            rate = (have - last) / (time.time() - t0) / 2**20
                            pct = f"{100 * have / total:5.1f}%" if total else "?"
                            print(f"  {os.path.basename(dest)} {pct} {have / 2**30:.2f} GB  {rate:.1f} MB/s", flush=True)
                            t0, last = time.time(), have
            if total is None or have >= total:
                break
        except (requests.RequestException, OSError) as exc:
            wait = min(60, 5 * (attempt + 1))
            print(f"  retry {attempt + 1}/{retries} after error: {exc} (sleep {wait}s)", flush=True)
            time.sleep(wait)
    else:
        raise RuntimeError(f"gave up on {url}")
    os.replace(part, dest)
    open(done_marker, "w").close()
    print(f"DONE {os.path.basename(dest)} {os.path.getsize(dest) / 2**30:.2f} GB", flush=True)


def join_parts(out: str, stem: str, n: int) -> None:
    parts = [os.path.join(out, f"{stem}.tar.gz.{i:02d}") for i in range(n)]
    target = os.path.join(out, f"{stem}.tar.gz")
    if os.path.exists(target) or not all(os.path.exists(p + ".done") for p in parts):
        return
    print(f"joining {n} parts -> {target}", flush=True)
    with open(target + ".tmp", "wb") as w:
        for p in parts:
            with open(p, "rb") as r:
                shutil.copyfileobj(r, w, 1 << 24)
    os.replace(target + ".tmp", target)
    print(f"JOINED {target}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--files", nargs="+", default=["captions", "anno", "glbs", "parts"], choices=list(GROUPS))
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    parser.add_argument("--no_join", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for group in args.files:
        for name in GROUPS[group]:
            url = f"{args.endpoint.rstrip('/')}/{REPO}/resolve/main/{name}"
            download(url, os.path.join(args.out, name))
        if not args.no_join and group == "glbs":
            join_parts(args.out, "normalized_glbs", 2)
        if not args.no_join and group == "parts":
            join_parts(args.out, "textured_part_glbs", 8)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
