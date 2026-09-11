"""HTTP wrapper around segment_parts.segment_parts: upload a model, get its parts back.

    POST /segment            multipart upload + options -> manifest and download links
    POST /segment_legacy     the deprecated 2D-map pipeline (segment_api.segment)
    GET  /jobs/{id}/download the result (one glb, or a zip in separate mode)
    GET  /jobs/{id}/parts/*  one part's glb
    GET  /jobs/{id}/atoms    the over-segmented atoms the vote merged (vertex-coloured)
    GET  /jobs/{id}/report   the per-unit vote report
    GET  /jobs/{id}/complete the X-Part solids (textured, if baked)
    GET  /jobs/{id}/complete_raw the solids before the albedo bake
    GET  /jobs/{id}/guidance/{name} one review overlay from work/guidance/
    GET  /jobs/{id}/map      the 2D part map, legacy jobs only
    GET  /health             stages, switches, current defaults, GPU busy flag

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
from pipeline import (
    COMPLETE_MODES, CONDITION_MODES, FLAT_PAINT_MODES, GRANULARITY,
    MERGE_MODES_ALL, MIRROR_MODES, PipelineOptions,
)

_DEFAULTS = PipelineOptions()

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
    guidance_dir = os.path.join(job, "work", "guidance")
    result.update({
        "pipeline": "segment_parts",
        "samples": len(atoms["samples"]),
        "atoms": atoms["atoms"],
        "atoms_glb": f"{base}/atoms",
        "report": f"{base}/report" if os.path.isfile(
            os.path.join(job, "work", "vote_report.json")) else None,
        "complete": f"{base}/complete" if os.path.isfile(
            os.path.join(job, "complete", "xpart_parts.glb")) else None,
        "complete_raw": f"{base}/complete_raw" if os.path.isfile(
            os.path.join(job, "complete", "xpart_parts_raw.glb")) else None,
        "guidance": [
            f"{base}/guidance/{name}"
            for name in sorted(os.listdir(guidance_dir))
            if name.endswith(".png")
        ] if os.path.isdir(guidance_dir) else [],
    })
    return result


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "busy": _gpu.locked(),
        "jobs_dir": JOBS_DIR,
        "pipeline": "segment_parts",
        "checkpoint": segment_parts.DEFAULT_CKPT,
        **_DEFAULTS.public(),
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
        default=[],
        description="Repeat once per output part. Join concepts with '+' to merge them "
                    "into one part, optionally named: 'opening=door+window'. Empty is "
                    "allowed only with merge=off."),
    unassigned_to: str | None = Form(
        None, description="Part that absorbs units no concept claimed."),
    samples: int = Form(_DEFAULTS.samples),
    azimuth: float = Form(_DEFAULTS.azimuth),
    azimuth_jitter: float = Form(_DEFAULTS.azimuth_jitter),
    granularity: str = Form(
        _DEFAULTS.granularity,
        description="fine 150/300 | medium 300/600 | coarse 800/1600. An explicit floor wins."),
    min_atom_faces: int | None = Form(None),
    min_unit_faces: int | None = Form(None),
    color_tol: float | None = Form(None),
    mirror: str = Form(_DEFAULTS.mirror, description=" | ".join(MIRROR_MODES)),
    min_recall: float | None = Form(None),
    view_azimuths: str = Form(_DEFAULTS.view_azimuths),
    view_elevations: str = Form(_DEFAULTS.view_elevations),
    radius: float = Form(_DEFAULTS.radius),
    resolution: int = Form(_DEFAULTS.resolution),
    flat_paint: str = Form(_DEFAULTS.flat_paint, description=" | ".join(FLAT_PAINT_MODES)),
    merge: str = Form(
        _DEFAULTS.merge,
        description="name = one node per prompt; unit = one node per voted unit; "
                    "off = stop after the units."),
    complete: str = Form(
        _DEFAULTS.complete,
        description="off | boxes (prompts only) | full (regenerate + bake)."),
    condition: str = Form(
        _DEFAULTS.condition,
        description="surface = faces the split assigned; box = whatever is in the box."),
    min_area_share: float = Form(_DEFAULTS.min_area_share),
    redraws: int = Form(_DEFAULTS.redraws),
    octree_resolution: int = Form(_DEFAULTS.octree_resolution),
    seed: int = Form(_DEFAULTS.seed),
    with_texture: bool = Form(_DEFAULTS.with_texture),
    texture_size: int = Form(_DEFAULTS.texture_size),
    reuse: bool = Form(_DEFAULTS.reuse),
    sam3_threshold: float = Form(_DEFAULTS.sam3_threshold),
    concept_bank: str | None = Form(None),
    no_concept_bank: bool = Form(False),
    allow_partial: bool = Form(False, description="Accept requested parts with no faces."),
) -> dict:
    if granularity not in GRANULARITY:
        raise HTTPException(400, f"granularity must be one of {tuple(GRANULARITY)}")
    if merge not in MERGE_MODES_ALL:
        raise HTTPException(400, f"merge must be one of {MERGE_MODES_ALL}")
    if complete not in COMPLETE_MODES:
        raise HTTPException(400, f"complete must be one of {COMPLETE_MODES}")
    if condition not in CONDITION_MODES:
        raise HTTPException(400, f"condition must be one of {CONDITION_MODES}")
    if flat_paint not in FLAT_PAINT_MODES:
        raise HTTPException(400, f"flat_paint must be one of {FLAT_PAINT_MODES}")
    if mirror not in MIRROR_MODES:
        raise HTTPException(400, f"mirror must be one of {MIRROR_MODES}")
    if no_concept_bank and concept_bank:
        raise HTTPException(400, "pass either concept_bank or no_concept_bank, not both")
    options = PipelineOptions.from_mapping({
        "unassigned_to": unassigned_to,
        "samples": samples,
        "azimuth": azimuth,
        "azimuth_jitter": azimuth_jitter,
        "granularity": granularity,
        "min_atom_faces": min_atom_faces,
        "min_unit_faces": min_unit_faces,
        "color_tol": color_tol,
        "mirror": mirror,
        "min_recall": min_recall,
        "view_azimuths": view_azimuths,
        "view_elevations": view_elevations,
        "radius": radius,
        "resolution": resolution,
        "flat_paint": flat_paint,
        "merge": merge,
        "complete": complete,
        "condition": condition,
        "min_area_share": min_area_share,
        "redraws": redraws,
        "octree_resolution": octree_resolution,
        "seed": seed,
        "with_texture": with_texture,
        "texture_size": texture_size,
        "reuse": reuse,
        "sam3_threshold": sam3_threshold,
        "concept_bank": concept_bank,
        "no_concept_bank": no_concept_bank,
        "allow_partial": allow_partial,
    })
    if not _gpu.acquire(blocking=False):
        raise HTTPException(409, "another segmentation is already running on this GPU")
    try:
        result = await run_in_threadpool(
            _run_job, uuid.uuid4().hex, await glb.read(), glb.filename or "input.glb",
            {"prompts": prompts, **options.segment_kwargs()},
        )
    finally:
        _gpu.release()
    result["options"] = options.public()["defaults"]
    return result


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


@app.get("/jobs/{job_id}/complete")
def complete(job_id: str):
    return FileResponse(_artifact(job_id, "complete", "xpart_parts.glb"),
                        media_type="model/gltf-binary", filename="xpart_parts.glb")


@app.get("/jobs/{job_id}/complete_raw")
def complete_raw(job_id: str):
    return FileResponse(_artifact(job_id, "complete", "xpart_parts_raw.glb"),
                        media_type="model/gltf-binary", filename="xpart_parts_raw.glb")


@app.get("/jobs/{job_id}/guidance/{name}")
def guidance(job_id: str, name: str):
    if os.path.basename(name) != name or not name.endswith(".png"):
        raise HTTPException(404, "no such overlay")
    return FileResponse(_artifact(job_id, "work", "guidance", name), media_type="image/png")


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
