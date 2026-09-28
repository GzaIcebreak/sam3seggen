"""HTTP wrapper around segment_parts.segment_parts: upload a model, get its parts back.

    POST /pipeline           one shot: model + prompts -> 202 {job_id, result url}; the job
                             is queued and runs split -> name -> repair -> texture
    GET  /jobs/{id}/result   the finished model itself (closed, textured parts), 409 until done
    POST /segment            the same pipeline, synchronous: manifest and download links
                             `prompts` is one comma-separated sentence, e.g. "head, torso"
    POST /segment_legacy     the deprecated 2D-map pipeline (segment_api.segment)
    GET  /jobs/{id}/download the result (one glb, or a zip in separate mode)
    GET  /jobs/{id}/parts/*  one part's glb
    GET  /jobs/{id}/atoms    the over-segmented atoms the vote merged (vertex-coloured)
    GET  /jobs/{id}/report   the per-unit vote report
    GET  /jobs/{id}/complete the closed solids (hybrid by default; textured, if baked)
    GET  /jobs/{id}/complete_raw the solids before the albedo bake
    GET  /jobs/{id}/guidance/{name} one review overlay from work/guidance/
    GET  /jobs/{id}/map      the 2D part map, legacy jobs only
    GET  /jobs               recent jobs (newest first); use this if the POST timed out
    GET  /jobs/latest        the newest job, running or finished
    GET  /jobs/{id}          one job's state, stage, and download links
    GET  /health             stages, switches, current defaults, GPU busy flag, latest job

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
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BeforeValidator

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import segment_api
import segment_parts
from pipeline import (
    COMPLETE_MODES, CONDITION_MODES, FLAT_PAINT_MODES, GRANULARITY,
    EXPORT_FROM_MODES, HOLOPART_LARGE_MODES, MERGE_MODES_ALL, MIRROR_MODES, MODES,
    PipelineOptions, REFINE_MODES,
)

from merge_parts import parse_part_floors
from smart_prompts import vlm_config, vlm_key_available

_DEFAULTS = PipelineOptions()

JOBS_DIR = os.path.abspath(os.environ.get(
    "SEGVIGEN_JOBS_DIR", os.path.join(tempfile.gettempdir(), "segvigen_jobs")))
JOB_ID = re.compile(r"\A[0-9a-f]{32}\Z")

_gpu = threading.Lock()
_current = {"job_id": None}
# POST /pipeline hands its job to this queue; one worker drains it under the GPU lock, so
# a second one-shot request waits its turn instead of getting the 409 /segment gives.
_queue: "queue.Queue[tuple[str, str, dict]]" = queue.Queue()
_pending: list[str] = []
_pending_lock = threading.Lock()
app = FastAPI(title="SegviGen part splitter", version="1", description=__doc__)

# Newest matching file/dir wins. AutoDL may drop the POST before we can reply,
# so clients recover the job_id from GET /jobs instead of the original response.
_STAGE_MARKERS = (
    ("done", ("complete", "xpart_parts.glb")),
    ("bake", ("complete", "xpart_parts_raw.glb")),
    ("complete", ("complete", "open_instances.glb")),
    ("merge", ("parts.glb",)),
    ("units", ("work", "units.npy")),
    ("split", ("work", "atoms.glb")),
    ("guidance", ("work", "guidance")),
    ("accepted", ("input.glb",)),
)

# /docs leaves the schema type name in empty optional boxes. Treat those as omitted.
_FORM_PLACEHOLDERS = ("", "string", "integer", "number", "null")


def blank_as_none(value):
    if value in (None, * _FORM_PLACEHOLDERS):
        return None
    return value


def parse_optional_int(value):
    value = blank_as_none(value)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, f"expected an integer, got {value!r}") from exc


def parse_optional_float(value):
    value = blank_as_none(value)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, f"expected a number, got {value!r}") from exc


OptionalStr = Annotated[str | None, BeforeValidator(blank_as_none)]


def serializable_errors(exc: RequestValidationError) -> list[dict]:
    """exc.errors() keeps exception objects in ctx; those cannot go into JSON."""
    errors = jsonable_encoder(exc.errors())
    for item in errors:
        loc = item.get("loc") or ()
        if loc and loc[-1] == "glb" and "UploadFile" in str(item.get("msg", "")):
            item["msg"] = (
                "glb must be a file upload (multipart field named glb), not text"
            )
            item["input"] = "<omitted>"
        elif isinstance(item.get("input"), str) and any(
            ord(ch) < 32 for ch in item["input"][:64]
        ):
            item["input"] = "<binary omitted>"
    return errors


@app.exception_handler(RequestValidationError)
async def _invalid_form(request: Request, exc: RequestValidationError):
    detail = serializable_errors(exc)
    print(f"{request.method} {request.url.path} 422 {detail}")
    return JSONResponse(status_code=422, content={"detail": detail})


def _write_json(path: str, payload: dict) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _read_json(path: str):
    try:
        with open(path, encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


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


def _iter_job_ids():
    if not os.path.isdir(JOBS_DIR):
        return
    for name in os.listdir(JOBS_DIR):
        if JOB_ID.match(name) and os.path.isdir(os.path.join(JOBS_DIR, name)):
            yield name


def _latest_job_id():
    newest = None
    newest_mtime = -1.0
    for job_id in _iter_job_ids():
        mtime = os.path.getmtime(os.path.join(JOBS_DIR, job_id))
        if mtime > newest_mtime:
            newest, newest_mtime = job_id, mtime
    return newest


def _infer_stage(path: str) -> str:
    for stage, relative in _STAGE_MARKERS:
        full = os.path.join(path, *relative)
        if os.path.isfile(full) or os.path.isdir(full):
            return stage
    return "accepted"


def _job_links(job_id: str, path: str) -> dict:
    base = f"/jobs/{job_id}"
    mapping = {
        "download": (f"{base}/download", ("parts.glb",)),
        "complete": (f"{base}/complete", ("complete", "xpart_parts.glb")),
        "complete_raw": (f"{base}/complete_raw", ("complete", "xpart_parts_raw.glb")),
        "complete_decisions": (f"{base}/complete_decisions", ("complete", "decisions.json")),
        "atoms": (f"{base}/atoms", ("work", "atoms.glb")),
        "report": (f"{base}/report", ("work", "vote_report.json")),
    }
    return {
        key: href
        for key, (href, relative) in mapping.items()
        if os.path.isfile(os.path.join(path, *relative))
    }


def _queue_position(job_id: str):
    with _pending_lock:
        return _pending.index(job_id) + 1 if job_id in _pending else None


def _summarize_job(job_id: str, include_result=False) -> dict:
    path = os.path.join(JOBS_DIR, job_id)
    meta = _read_json(os.path.join(path, "job.json")) or {}
    stage = _infer_stage(path)
    running = _gpu.locked() and _current["job_id"] == job_id
    position = _queue_position(job_id)
    if meta.get("state") == "error":
        state = "error"
    elif running:
        state = "running"
    elif position is not None:
        state = "queued"
    elif stage == "done" or os.path.isfile(os.path.join(path, "result.json")):
        state = "done"
    elif stage == "accepted" and not meta:
        state = "incomplete"
    else:
        state = "incomplete"
    summary = {
        "job_id": job_id,
        "state": state,
        "stage": stage,
        "filename": meta.get("filename"),
        "started": meta.get("started"),
        "finished": meta.get("finished"),
        "error": meta.get("error"),
        "position": position,
        "mtime": os.path.getmtime(path),
        "links": _job_links(job_id, path),
        "status_url": f"/jobs/{job_id}",
        "result_url": f"/jobs/{job_id}/result",
    }
    if include_result:
        summary["result"] = _read_json(os.path.join(path, "result.json"))
    return summary


def _record_job(job_id: str, filename: str) -> None:
    _current["job_id"] = job_id
    payload = {
        "job_id": job_id,
        "filename": filename,
        "started": time.time(),
        "state": "running",
    }
    _write_json(os.path.join(JOBS_DIR, job_id, "job.json"), payload)
    _write_json(os.path.join(JOBS_DIR, "current.json"), {"job_id": job_id})


def _enqueue_job(upload: bytes, filename: str, options: dict) -> dict:
    """Store the upload, mark the job queued, hand it to the worker; return the ticket."""
    job_id = uuid.uuid4().hex
    job = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job, exist_ok=True)
    with open(os.path.join(job, "input" + (os.path.splitext(filename)[1] or ".glb")), "wb") as file:
        file.write(upload)
    _write_json(os.path.join(job, "job.json"), {
        "job_id": job_id, "filename": filename, "queued": time.time(), "state": "queued",
    })
    with _pending_lock:
        _pending.append(job_id)
        position = len(_pending)
    _queue.put((job_id, filename, options))
    return {
        "job_id": job_id, "state": "queued", "position": position,
        "status_url": f"/jobs/{job_id}", "result_url": f"/jobs/{job_id}/result",
    }


def _worker() -> None:
    """Drain the one-shot queue, one job at a time, under the same GPU lock /segment uses."""
    while True:
        job_id, filename, options = _queue.get()
        job = os.path.join(JOBS_DIR, job_id)
        source = os.path.join(job, "input" + (os.path.splitext(filename)[1] or ".glb"))
        try:
            with open(source, "rb") as file:
                upload = file.read()
        except OSError as exc:
            _finish_job(job_id, error=f"upload lost before the job ran: {exc}")
            with _pending_lock:
                if job_id in _pending:
                    _pending.remove(job_id)
            continue
        with _gpu:
            with _pending_lock:
                if job_id in _pending:
                    _pending.remove(job_id)
            try:
                _run_job(job_id, upload, filename, options)
            except Exception as exc:  # _run_job already recorded the error on the job
                print(f"[pipeline] job {job_id} failed: {exc}")
            finally:
                _current["job_id"] = None


threading.Thread(target=_worker, name="segvigen-pipeline-worker", daemon=True).start()


def _finish_job(job_id: str, *, result=None, error=None) -> None:
    path = os.path.join(JOBS_DIR, job_id)
    meta = _read_json(os.path.join(path, "job.json")) or {
        "job_id": job_id, "started": time.time()}
    meta["finished"] = time.time()
    meta["state"] = "error" if error else "done"
    if error:
        meta["error"] = error
    _write_json(os.path.join(path, "job.json"), meta)
    if result is not None:
        _write_json(os.path.join(path, "result.json"), result)


def _run_job(job_id: str, upload: bytes, filename: str, options: dict, legacy=False) -> dict:
    job = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job, exist_ok=True)
    _record_job(job_id, filename)
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
        _finish_job(job_id, error=str(exc))
        raise HTTPException(400, str(exc)) from exc
    except subprocess.CalledProcessError as exc:
        # The stage printed its own traceback to the server log; the client only needs to
        # know which one gave up and that the intermediates are still on disk.
        stage = os.path.basename(exc.cmd[1]) if len(exc.cmd) > 1 else "pipeline"
        detail = f"{stage} failed (exit {exc.returncode}); intermediates kept in job {job_id}"
        _finish_job(job_id, error=detail)
        raise HTTPException(500, detail) from exc
    except Exception as exc:
        _finish_job(job_id, error=str(exc))
        raise

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
        _finish_job(job_id, result=result)
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
        "complete_decisions": f"{base}/complete_decisions" if os.path.isfile(
            os.path.join(job, "complete", "decisions.json")) else None,
        "guidance": [
            f"{base}/guidance/{name}"
            for name in sorted(os.listdir(guidance_dir))
            if name.endswith(".png")
        ] if os.path.isdir(guidance_dir) else [],
    })
    _finish_job(job_id, result=result)
    return result


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "busy": _gpu.locked(),
        "current_job": _current["job_id"] if _gpu.locked() else None,
        "latest_job": _latest_job_id(),
        "jobs_dir": JOBS_DIR,
        "pipeline": "segment_parts",
        "checkpoint": segment_parts.DEFAULT_CKPT,
        "smart_mode": {"key_configured": vlm_key_available(),
                       "base_url": vlm_config()[0], "model": vlm_config()[1] or "auto"},
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


@app.get("/jobs")
def list_jobs(limit: int = 20) -> dict:
    """Newest jobs first. Use this when AutoDL dropped the POST and the client lost job_id."""
    rows = [_summarize_job(job_id) for job_id in _iter_job_ids()]
    rows.sort(key=lambda row: row["mtime"], reverse=True)
    return {
        "busy": _gpu.locked(),
        "current_job": _current["job_id"] if _gpu.locked() else None,
        "latest_job": rows[0]["job_id"] if rows else None,
        "jobs": rows[: max(1, min(limit, 100))],
    }


@app.get("/jobs/latest")
def latest_job() -> dict:
    job_id = _current["job_id"] if _gpu.locked() and _current["job_id"] else _latest_job_id()
    if not job_id:
        raise HTTPException(404, "no jobs yet")
    return _summarize_job(job_id, include_result=True)


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    _job_dir(job_id)
    return _summarize_job(job_id, include_result=True)


def _check_switches(mapping: dict) -> None:
    """Reject an option outside its switch's range with the same 400 /segment gives."""
    checks = (
        ("granularity", tuple(GRANULARITY)), ("merge", MERGE_MODES_ALL),
        ("complete", COMPLETE_MODES), ("condition", CONDITION_MODES),
        ("holopart_large", HOLOPART_LARGE_MODES), ("refine", REFINE_MODES),
        ("export_from", EXPORT_FROM_MODES), ("flat_paint", FLAT_PAINT_MODES),
        ("mirror", MIRROR_MODES), ("mode", MODES),
    )
    for key, allowed in checks:
        value = mapping.get(key)
        if value is not None and value not in allowed:
            raise HTTPException(400, f"{key} must be one of {tuple(allowed)}")
    if mapping.get("part_min_area_share") is not None:
        try:
            parse_part_floors(mapping["part_min_area_share"])
        except ValueError as error:
            raise HTTPException(400, str(error))
    if mapping.get("mode") == "smart" and not vlm_key_available():
        raise HTTPException(400, "mode=smart (智能分割模式) needs SEGVIGEN_VLM_API_KEY on the server "
                                 "(env var or the repo .env file)")


@app.post("/pipeline", status_code=202)
async def pipeline(
    glb: UploadFile = File(..., description="The model to split and repair."),
    prompts: str = Form(
        default="",
        description="One comma-separated sentence of part names, e.g. 'armor, staff, "
                    "base, body=head+hand'. Empty = 主体, 底座."),
    unassigned_to: OptionalStr = Form(
        _DEFAULTS.unassigned_to,
        description="Part that absorbs faces no prompt claimed; must be one of the prompts."),
    mode: str = Form(
        _DEFAULTS.mode,
        description=" | ".join(MODES) + ". smart = 智能分割模式: with no prompts, Kimi reviews "
        "the concept candidates and the renders and names the parts (needs MOONSHOT_API_KEY)"),
    options: OptionalStr = Form(
        None,
        description="Optional JSON object of PipelineOptions overrides, same names as the "
                    "/segment form fields, e.g. {\"merge_gap\": 0.01, \"complete\": \"off\"}. "
                    "Anything omitted takes GET /health defaults."),
) -> dict:
    """One shot: upload a model and its part names, get a ticket back at once.

    The job is queued and runs split -> name -> repair -> texture on the worker; poll
    `status_url` until `state` is `done`, then download `result_url`, which is the
    finished model itself. Unlike /segment this never blocks past the gateway's patience
    and never answers 409: a second request waits in line.
    """
    overrides = {}
    if options:
        try:
            overrides = json.loads(options)
        except json.JSONDecodeError as exc:
            raise HTTPException(400, f"options must be a JSON object: {exc}")
        if not isinstance(overrides, dict):
            raise HTTPException(400, "options must be a JSON object")
    merged = {"unassigned_to": unassigned_to, "mode": mode, **overrides}
    _check_switches(merged)
    try:
        resolved = PipelineOptions.from_mapping(merged)
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, f"bad option: {exc}")
    return _enqueue_job(await glb.read(), glb.filename or "input.glb",
                        {"prompts": prompts, **resolved.segment_kwargs()})


@app.get("/jobs/{job_id}/result")
def job_result(job_id: str):
    """The finished model: closed, textured parts when repair ran, else the open parts."""
    path = _job_dir(job_id)
    summary = _summarize_job(job_id)
    if summary["state"] in ("queued", "running"):
        raise HTTPException(409, {"state": summary["state"], "stage": summary["stage"],
                                  "position": summary["position"], "status_url": summary["status_url"]})
    if summary["state"] == "error":
        raise HTTPException(500, summary["error"] or "job failed")
    for relative, name in ((("complete", "xpart_parts.glb"), "repaired_parts.glb"),
                           (("parts.glb",), "parts.glb")):
        candidate = os.path.join(path, *relative)
        if os.path.isfile(candidate):
            return FileResponse(candidate, media_type="model/gltf-binary", filename=name)
    raise HTTPException(404, f"job {job_id} finished without a model")


@app.post("/segment")
async def segment(
    glb: UploadFile = File(..., description="The model to split."),
    prompts: str = Form(
        default="",
        description="One comma-separated sentence of part names, e.g. "
                    "'head, torso, arm'. Spaces inside a name are kept. "
                    "'body=head+face' still merges concepts. Empty = 主体, 底座."),
    unassigned_to: OptionalStr = Form(
        _DEFAULTS.unassigned_to,
        description="Part that absorbs units no concept claimed "
                    "(default body). Must be one of the prompts, or it is ignored."),
    samples: int = Form(_DEFAULTS.samples),
    azimuth: float = Form(_DEFAULTS.azimuth),
    azimuth_jitter: float = Form(_DEFAULTS.azimuth_jitter),
    granularity: str = Form(
        _DEFAULTS.granularity,
        description="fine 150/300 | medium 300/600 | coarse 800/1600. An explicit floor wins."),
    min_atom_faces: str | None = Form(None),
    min_unit_faces: str | None = Form(None),
    color_tol: str | None = Form(None),
    mirror: str = Form(_DEFAULTS.mirror, description=" | ".join(MIRROR_MODES)),
    min_recall: str | None = Form(None),
    view_azimuths: str = Form(_DEFAULTS.view_azimuths),
    view_elevations: str = Form(_DEFAULTS.view_elevations),
    radius: float = Form(_DEFAULTS.radius),
    resolution: int = Form(_DEFAULTS.resolution),
    flat_paint: str = Form(_DEFAULTS.flat_paint, description=" | ".join(FLAT_PAINT_MODES)),
    merge: str = Form(
        _DEFAULTS.merge,
        description="name = one node per prompt; unit = one node per voted unit; "
                    "fragments = keep the geometric split, fold only specks; "
                    "off = unnamed units."),
    complete: str = Form(
        _DEFAULTS.complete,
        description="off | boxes (prompts only) | full (X-Part only) | "
                    "hybrid (X-Part, HoloPart on large box-escapees; default)."),
    condition: str = Form(
        _DEFAULTS.condition,
        description="surface = faces the split assigned; box = whatever is in the box."),
    min_area_share: float = Form(_DEFAULTS.min_area_share),
    fragment_share: float = Form(
        _DEFAULTS.fragment_share,
        description="merge=fragments: fold a unit below this share of the surface. "
                    "Smaller keeps more pieces (default 0.01)."),
    redraws: int = Form(_DEFAULTS.redraws),
    holopart_large: str = Form(
        _DEFAULTS.holopart_large,
        description="complete=hybrid: " + " | ".join(HOLOPART_LARGE_MODES)
        + ". score = measure each X-Part solid, HoloPart the low scorers, keep the better"),
    score_candidate: float = Form(
        _DEFAULTS.score_candidate,
        description="score: a large instance below this also gets a HoloPart draw"),
    score_candidate_small: float = Form(
        _DEFAULTS.score_candidate_small, description="score: the same for a small instance"),
    score_floor: float = Form(
        _DEFAULTS.score_floor,
        description="score: both solids below this keep the open surface"),
    part_min_area_share: OptionalStr = Form(
        None, description="Per-part min_area_share: name=share[,name=share]"),
    fold_within_part: bool = Form(
        _DEFAULTS.fold_within_part,
        description="Fold a small component only into its own part (ornaments on a tree)"),
    merge_gap: float = Form(
        _DEFAULTS.merge_gap,
        description="Rejoin same-part pieces within this share of the model diagonal "
                    "(a hand a staff cut in two); 0 = off"),
    merge_max_share: float = Form(
        _DEFAULTS.merge_max_share,
        description="merge_gap: only a piece under this share of the surface joins"),
    refine: str = Form(
        _DEFAULTS.refine,
        description=" | ".join(REFINE_MODES) + ". masks: after the vote, cut a unit where "
        "SAM3's per-face masks name a coherent patch of it differently"),
    refine_min_share: float = Form(
        _DEFAULTS.refine_min_share,
        description="refine=masks: a patch is cut off only above this share of its unit"),
    export_from: str = Form(
        _DEFAULTS.export_from,
        description=" | ".join(EXPORT_FROM_MODES) + ". source: cut the parts from the source "
        "model with its own texture (full resolution); remesh: cut the remesh and bake"),
    auto_prompts: bool = Form(
        _DEFAULTS.auto_prompts,
        description="Empty prompts: ask SAM3 for every bank concept and pick the part names "
                    "(default). false: name the parts 主体 / 底座 instead."),
    mode: str = Form(
        _DEFAULTS.mode,
        description=" | ".join(MODES) + ". smart = 智能分割模式 (Kimi names the parts; needs "
        "MOONSHOT_API_KEY)"),
    octree_resolution: int = Form(_DEFAULTS.octree_resolution),
    seed: int = Form(_DEFAULTS.seed),
    with_texture: bool = Form(_DEFAULTS.with_texture),
    texture_size: int = Form(
        _DEFAULTS.texture_size,
        description="Small-part atlas edge. Parts covering ≥8% / ≥40% of the surface "
                    "bake at 2× / 4× this, capped at 8192."),
    reuse: bool = Form(_DEFAULTS.reuse),
    sam3_threshold: float = Form(_DEFAULTS.sam3_threshold),
    concept_bank: OptionalStr = Form(None),
    no_concept_bank: bool = Form(False),
    allow_partial: bool = Form(
        True, description="Skip a prompt SAM3 never saw and finish the rest (default). "
                          "Set false (or pass strict_parts) to fail the job instead."),
    strict_parts: bool = Form(
        False, description="Fail if a requested name got no mask or no faces."),
) -> dict:
    if granularity not in GRANULARITY:
        raise HTTPException(400, f"granularity must be one of {tuple(GRANULARITY)}")
    if merge not in MERGE_MODES_ALL:
        raise HTTPException(400, f"merge must be one of {MERGE_MODES_ALL}")
    if complete not in COMPLETE_MODES:
        raise HTTPException(400, f"complete must be one of {COMPLETE_MODES}")
    if condition not in CONDITION_MODES:
        raise HTTPException(400, f"condition must be one of {CONDITION_MODES}")
    if holopart_large not in HOLOPART_LARGE_MODES:
        raise HTTPException(400, f"holopart_large must be one of {HOLOPART_LARGE_MODES}")
    if refine not in REFINE_MODES:
        raise HTTPException(400, f"refine must be one of {REFINE_MODES}")
    if export_from not in EXPORT_FROM_MODES:
        raise HTTPException(400, f"export_from must be one of {EXPORT_FROM_MODES}")
    if mode not in MODES:
        raise HTTPException(400, f"mode must be one of {MODES}")
    if mode == "smart" and not vlm_key_available():
        raise HTTPException(400, "mode=smart (智能分割模式) needs SEGVIGEN_VLM_API_KEY on the server")
    if part_min_area_share in ("", "string"):
        part_min_area_share = None
    if part_min_area_share is not None:
        try:
            parse_part_floors(part_min_area_share)
        except ValueError as error:
            raise HTTPException(400, str(error))
    if flat_paint not in FLAT_PAINT_MODES:
        raise HTTPException(400, f"flat_paint must be one of {FLAT_PAINT_MODES}")
    if mirror not in MIRROR_MODES:
        raise HTTPException(400, f"mirror must be one of {MIRROR_MODES}")
    if no_concept_bank and concept_bank:
        raise HTTPException(400, "pass either concept_bank or no_concept_bank, not both")
    min_atom_faces = parse_optional_int(min_atom_faces)
    min_unit_faces = parse_optional_int(min_unit_faces)
    color_tol = parse_optional_float(color_tol)
    min_recall = parse_optional_float(min_recall)
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
        "fragment_share": fragment_share,
        "redraws": redraws,
        "holopart_large": holopart_large,
        "score_candidate": score_candidate,
        "score_candidate_small": score_candidate_small,
        "score_floor": score_floor,
        "part_min_area_share": part_min_area_share,
        "fold_within_part": fold_within_part,
        "merge_gap": merge_gap,
        "merge_max_share": merge_max_share,
        "refine": refine,
        "refine_min_share": refine_min_share,
        "export_from": export_from,
        "auto_prompts": auto_prompts,
        "mode": mode,
        "octree_resolution": octree_resolution,
        "seed": seed,
        "with_texture": with_texture,
        "texture_size": texture_size,
        "reuse": reuse,
        "sam3_threshold": sam3_threshold,
        "concept_bank": concept_bank,
        "no_concept_bank": no_concept_bank,
        "allow_partial": allow_partial,
        "strict_parts": strict_parts,
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
    prompts: str = Form(
        ..., description="One comma-separated sentence of part names, e.g. "
                        "'head, torso, arm'. '+' still merges concepts."),
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


@app.get("/jobs/{job_id}/complete_decisions")
def complete_decisions(job_id: str):
    return FileResponse(_artifact(job_id, "complete", "decisions.json"),
                        media_type="application/json")


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
    parser.add_argument("--port", type=int, default=6006)
    args = parser.parse_args()
    os.makedirs(JOBS_DIR, exist_ok=True)
    print(f"jobs kept in {JOBS_DIR}; docs at http://{args.host}:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
