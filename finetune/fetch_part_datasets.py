"""Download the two part-dataset expansion sources (see finetune/PLAN_data_expansion.md).

PartNeXt  (AuWang/PartNeXt + AuWang/PartNeXt_mesh, 57 GB): 23 519 textured objects, 350 187 parts,
    50 categories, sourced from Objaverse / ABO / 3D-FUTURE. Its part names are human-annotated and
    hierarchical (`hierarchyList`), so it needs no caption relabelling -- that is the one manual
    bottleneck of the PartVerse route.
PartVerse-XL (dscdyc/partversexl, 106 GB of the 823 GB repo): same directory layout as the PartVerse
    we already have, so import_partverse.py reads it unchanged. Only anno_infos + normalized_glbs +
    text_captions.json are needed; textured_part_glbs (67 archives, 717 GB) is not, because parts are
    cut from the whole mesh by face labels and only need geometry.
    The released split lists 32 659 objects, of which 25 406 are not in the local PartVerse.

Resumable: re-running skips whatever is already in the HF cache / local dir.

    python finetune/fetch_part_datasets.py --root E:\\AI_New\\ModelGen\\datasets --which partnext xl
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from huggingface_hub import snapshot_download

JOBS = {
    "partnext_anno": dict(repo_id="AuWang/PartNeXt", sub="partnext/anno", allow=None),
    "partnext_mesh": dict(repo_id="AuWang/PartNeXt_mesh", sub="partnext/mesh", allow=None),
    "xl": dict(repo_id="dscdyc/partversexl", sub="partverse_xl",
               allow=["anno_infos.tar.gz", "normalized_glbs.tar.gz0*", "text_captions.json",
                      "metadata.csv", "train.csv", "val.csv", "README.md"]),
}
GROUPS = {"partnext": ["partnext_anno", "partnext_mesh"], "xl": ["xl"]}


def disk_size(dst: str) -> int:
    """Bytes under dst including .incomplete parts, so a running download shows as growth."""
    total = 0
    for dp, _, fs in os.walk(dst):
        for f in fs:
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return total


def download(job: str, root: str, workers: int) -> None:
    spec = JOBS[job]
    snapshot_download(repo_id=spec["repo_id"], repo_type="dataset",
                      local_dir=os.path.join(root, spec["sub"]),
                      allow_patterns=spec["allow"], max_workers=workers)


def run(job: str, root: str, workers: int, tries: int, stall_s: int) -> None:
    """One repo, downloaded by a child process that the parent restarts whenever it stalls.

    A 10.7 GB part fetched through a local proxy regularly stops mid-file with the connection dead
    but no exception raised, leaving the process at 0 % CPU indefinitely -- so a plain retry loop
    never fires. The watchdog therefore judges liveness by bytes on disk, not by the child's exit:
    no growth for stall_s seconds means kill and re-enter. snapshot_download resumes from the
    .incomplete file, so a restart costs only the current chunk.
    """
    dst = os.path.join(root, JOBS[job]["sub"])
    os.makedirs(dst, exist_ok=True)
    print(f"[fetch] {job}: {JOBS[job]['repo_id']} -> {dst}", flush=True)
    t0 = time.time()
    start_size = disk_size(dst)

    for attempt in range(1, tries + 1):
        proc = mp.Process(target=download, args=(job, root, workers), daemon=True)
        proc.start()
        last, last_t = disk_size(dst), time.time()
        while proc.is_alive():
            time.sleep(15)
            now = disk_size(dst)
            if now > last:
                last, last_t = now, time.time()
            elif time.time() - last_t > stall_s:
                print(f"[fetch] {job}: stalled at {now / 1e9:.1f} GB for {stall_s}s, restarting "
                      f"(attempt {attempt}/{tries})", flush=True)
                proc.terminate()
                proc.join(30)
                break
        else:
            proc.join()
            if proc.exitcode == 0:
                break
            print(f"[fetch] {job}: child exited {proc.exitcode} at {disk_size(dst) / 1e9:.1f} GB, "
                  f"retrying (attempt {attempt}/{tries})", flush=True)
            time.sleep(min(30, 5 * attempt))

    size = disk_size(dst)
    dt = time.time() - t0
    print(f"[fetch] {job}: done, {size / 1e9:.1f} GB on disk, {dt / 60:.1f} min "
          f"({(size - start_size) / 1e6 / max(dt, 1):.1f} MB/s this run)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"E:\AI_New\ModelGen\datasets")
    ap.add_argument("--which", nargs="+", default=["partnext", "xl"],
                    choices=sorted(set(GROUPS) | set(JOBS)))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tries", type=int, default=40)
    ap.add_argument("--stall_s", type=int, default=120, help="seconds without disk growth before restart")
    args = ap.parse_args()

    jobs: list[str] = []
    for w in args.which:
        for j in GROUPS.get(w, [w]):
            if j not in jobs:
                jobs.append(j)
    for j in jobs:
        run(j, args.root, args.workers, args.tries, args.stall_s)
    print("ALL_FETCHED", flush=True)


if __name__ == "__main__":
    main()
