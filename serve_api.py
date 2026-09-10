"""HTTP wrapper around segment_parts.segment_parts: upload a model, get its parts back.

    POST /segment            multipart upload + options -> manifest and download links
    POST /segment_legacy     the deprecated 2D-map pipeline (segment_api.segment)
    GET  /jobs/{id}/download the result (one glb, or a zip in separate mode)
    GET  /jobs/{id}/parts/*  one part's glb
    GET  /jobs/{id}/atoms    the over-segmented atoms the vote merged (vertex-coloured)
    GET  /jobs/{id}/report   the per-unit vote report
    GET  /jobs/{id}/map      the 2D part map, legacy jobs only
    GET  /health             defaults and whether the GPU is busy

Every stage still runs as a subprocess that loads its own model, and the default pipeline
samples SegviGen several times, so a request costs several minutes and the box has one
GPU: jobs take a lock and a second request gets 409 rather than queueing behind an
invisible wait. This is a test harness, not a throughput service.

Run it with the SegviGen venv and env.sh sourced (see run_serve.sh); interactive docs are
at /docs.
"""
from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import segment_api
import segment_parts

JOBS_DIR = os.path.abspath(os.environ.get(
    "SEGVIGEN_JOBS_DIR", os.path.join(tempfile.gettempdir(), "segvigen_jobs")))
JOB_ID = re.compile(r"\A[0-9a-f]{32}\Z")

_gpu = threading.Lock()
app = FastAPI(title="SegviGen part splitter", version="1", description=__doc__)


def _job_dir(job_id: str) -> str:
    """Resolve a job directory, refusing anything that is not one of our own ids."""
    if not JOB_ID.match(job_id):
        raise HTTPException(404, "no such job")
    path = os.path.join(JOBS_DIR, job_id)
    if not os.path.isdir(path):
        raise HTTPException(404, "no such job")
    return path


def _artifact(job_id: str, *relative: str) -> str:
    path = os.path.join(_job_dir(job_id), *relative)
    if not os.path.isfile(path):
        raise HTTPException(404, f"job {job_id} has no {'/'.join(relative)}")
    return path


def _run_job(job_id: str, upload: bytes, filename: str, options: dict, legacy=False) -> dict:
    job = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job, exist_ok=True)
    source = os.path.join(job, "input" + (os.path.splitext(filename)[1] or ".glb"))
    with open(source, "wb") as file:
        file.write(upload)

    out_glb = os.path.join(job, "parts.glb")
    started = time.time()
    run = segment_api.segment if legacy else segment_parts.segment_parts
    try:
        manifest = run(source, options.pop("prompts"), out_glb,
                       work_dir=os.path.join(job, "work"), **options)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except subprocess.CalledProcessError as exc:
        # The stage printed its own traceback to the server log; the client only needs to
        # know which one gave up and that the intermediates are still on disk.
        stage = os.path.basename(exc.cmd[1]) if len(exc.cmd) > 1 else "pipeline"
        raise HTTPException(
            500, f"{stage} failed (exit {exc.returncode}); intermediates kept in job {job_id}"
        ) from exc

    base = f"/jobs/{job_id}"
    for row in manifest:
        row.pop("file", None)
    result = {
        "job_id": job_id,
        "seconds": round(time.time() - started, 1),
        "parts": manifest,
        "download": f"{base}/download",
        "files": [],
    }
    if legacy:
        result.update({
            "pipeline": "legacy",
            "parts_output": options["parts_output"],
            "split_mode": options["split_mode"],
            "map": f"{base}/map",
            "render": f"{base}/render",
            "files": [f"{base}/parts/{row['node']}.glb" for row in manifest]
                     if options["parts_output"] == "separate" else [],
        })
        return result
    with open(os.path.join(job, "work", "atoms_report.json"), "r", encoding="utf-8") as file:
        atoms = json.load(file)
    result.update({
        "pipeline": "segment_parts",
        "samples": len(atoms["samples"]),
        "atoms": atoms["atoms"],
        "atoms_glb": f"{base}/atoms",
        "report": f"{base}/report",
    })
    return result


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "busy": _gpu.locked(),
        "jobs_dir": JOBS_DIR,
        "defaults": {
            "pipeline": "segment_parts",
            "samples": segment_parts.DEFAULT_SAMPLES,
            "azimuth_jitter": segment_parts.DEFAULT_AZIMUTH_JITTER,
            "checkpoint": segment_parts.DEFAULT_CKPT,
        },
        "legacy_defaults": {
            "assign": "paint",
            "use_v6": True,
            "parts_output": "combined",
            "split_mode": "stain",
            "checkpoint": segment_api._resolve_ckpt(None, True, True),
            "concept_bank": segment_api.DEFAULT_CONCEPT_BANK,
            "rank_model": segment_api.DEFAULT_RANK_MODEL,
        },
    }


@app.post("/segment")
async def segment(
    glb: UploadFile = File(..., description="The model to split."),
    prompts: list[str] = Form(
        ..., description="Repeat once per output part. Join concepts with '+' to merge "
                         "them into one part, optionally named: 'opening=door+window'. "
                         "Never space-separate them -- a concept may contain spaces."),
    unassigned_to: str | None = Form(
        None, description="Part that absorbs atoms no concept claimed, so the output has "
                          "exactly as many parts as were asked for."),
    samples: int = Form(
        segment_parts.DEFAULT_SAMPLES,
        description="full_seg samples to intersect. More samples means finer atoms and a "
                    "longer run: each one is a full flow-model pass."),
    azimuth: float = Form(0.0, description="Degrees to orbit the conditioning camera."),
    azimuth_jitter: float = Form(
        segment_parts.DEFAULT_AZIMUTH_JITTER,
        description="How far the extra samples orbit either side of azimuth. full_seg only "
                    "holds up near the front, so keep this under ~45."),
    with_texture: bool = Form(True, description="Bake the source albedo onto each part."),
    texture_size: int = Form(2048),
    sam3_threshold: float = Form(0.3),
    allow_partial: bool = Form(
        False, description="Accept requested parts that ended up with no faces."),
) -> dict:
    if not _gpu.acquire(blocking=False):
        raise HTTPException(409, "another segmentation is already running on this GPU")
    try:
        return await run_in_threadpool(
            _run_job, uuid.uuid4().hex, await glb.read(), glb.filename or "input.glb",
            {
                "prompts": prompts,
                "unassigned_to": unassigned_to,
                "samples": samples,
                "azimuth": azimuth,
                "azimuth_jitter": azimuth_jitter,
                "with_texture": with_texture,
                "texture_size": texture_size,
                "sam3_threshold": sam3_threshold,
                "strict_parts": not allow_partial,
            },
        )
    finally:
        _gpu.release()


@app.post("/segment_legacy", deprecated=True)
async def segment_legacy(
    glb: UploadFile = File(..., description="The model to split."),
    prompts: list[str] = Form(
        ..., description="Repeat once per output part. Join concepts with '+' to merge "
                         "them into one part, optionally named: 'opening=door+window'. "
                         "Never space-separate them -- a concept may contain spaces."),
    unassigned_to: str | None = Form(
        None, description="Part that absorbs foreground no prompt claimed, so the output "
                          "has exactly as many parts as were asked for."),
    azimuth: float = Form(0.0, description="Degrees to orbit the conditioning camera."),
    front_view: str | None = Form(
        None, description="metric | auto | vlm -- pick the view automatically, ignoring azimuth."),
    assign: str = Form(
        "paint", description="paint (default) | rank (EASE ranker) | auto (score both, keep "
                             "the better map) | argmax."),
    use_v6: bool = Form(True, description="Segment with full_seg_v6.ckpt."),
    with_texture: bool = Form(True, description="Bake the source albedo onto each part."),
    parts_output: str = Form(
        "combined", description="combined (default) = one glb, one node per part. "
                                "separate = additionally one glb per part; /download zips them."),
    split_mode: str = Form(
        "stain", description="stain (default) = cut exactly along SegviGen's colouring, only "
                             "absorbing fragments under 100 faces. weld = same cuts, but whole "
                             "pieces the 2D map clearly paints as another part are renamed. "
                             "refine = overwrite visible faces pixel by pixel and reassign islands."),
    texture_size: int = Form(2048),
    sam3_threshold: float | None = Form(
        None, description="Default is the calibrated value for the painter in use."),
    allow_partial: bool = Form(
        False, description="Accept missing or extra parts instead of failing the request."),
) -> dict:
    if not _gpu.acquire(blocking=False):
        raise HTTPException(409, "another segmentation is already running on this GPU")
    try:
        return await run_in_threadpool(
            _run_job, uuid.uuid4().hex, await glb.read(), glb.filename or "input.glb",
            {
                "prompts": prompts,
                "unassigned_to": unassigned_to,
                "azimuth": azimuth,
                "front_view": front_view,
                "assign": assign,
                "use_v6": use_v6,
                "with_texture": with_texture,
                "parts_output": parts_output,
                "split_mode": split_mode,
                "texture_size": texture_size,
                "sam3_threshold": sam3_threshold,
                "strict_parts": not allow_partial,
            },
            legacy=True,
        )
    finally:
        _gpu.release()


@app.get("/jobs/{job_id}/download")
def download(job_id: str):
    parts_dir = os.path.join(_job_dir(job_id), "parts")
    if not os.path.isdir(parts_dir):
        return FileResponse(_artifact(job_id, "parts.glb"), media_type="model/gltf-binary",
                            filename="parts.glb")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(os.listdir(parts_dir)):
            archive.write(os.path.join(parts_dir, name), name)
    return Response(buffer.getvalue(), media_type="application/zip", headers={
        "content-disposition": f'attachment; filename="{job_id}_parts.zip"'})


@app.get("/jobs/{job_id}/parts/{name}")
def part(job_id: str, name: str):
    if os.path.basename(name) != name:
        raise HTTPException(404, "no such part")
    return FileResponse(_artifact(job_id, "parts", name), media_type="model/gltf-binary",
                        filename=name)


@app.get("/jobs/{job_id}/atoms")
def atoms(job_id: str):
    """The over-segmented atoms the vote merged, one colour each: what to look at first
    when a part came out wrong, since the vote can only merge what this file already cut."""
    return FileResponse(_artifact(job_id, "work", "atoms.glb"),
                        media_type="model/gltf-binary", filename="atoms.glb")


@app.get("/jobs/{job_id}/report")
def report(job_id: str):
    return FileResponse(_artifact(job_id, "work", "vote_report.json"),
                        media_type="application/json")


@app.get("/jobs/{job_id}/map")
def part_map(job_id: str):
    """The 2D part map; only legacy jobs have one."""
    return FileResponse(_artifact(job_id, "work", "sam3_2d_map.png"), media_type="image/png")


@app.get("/jobs/{job_id}/render")
def render(job_id: str):
    return FileResponse(_artifact(job_id, "work", "render.png"), media_type="image/png")


def main():
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8020)
    args = parser.parse_args()
    os.makedirs(JOBS_DIR, exist_ok=True)
    print(f"jobs kept in {JOBS_DIR}; docs at http://{args.host}:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
