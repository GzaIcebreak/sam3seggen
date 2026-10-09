"""The current pipeline, as one options object shared by the CLI and the HTTP API.

    paint -> guidance -> split -> units -> [merge] -> [complete -> bake]

Geometry decides every boundary; language only names. Stages 3 and 6 cost GPU minutes,
the rest is seconds once the renders are cached. `--merge off` stops after units;
`--complete off` stops after the open, textured parts.glb; the default is
`--complete hybrid` (X-Part, then HoloPart on a large solid that left its box).

This module is the contract. segment_parts.py / merge_parts.py / serve_api.py read their
defaults and switches from here so a knob cannot drift between the CLI and the API.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from hybrid_complete import SCORE_CANDIDATE, SCORE_CANDIDATE_SMALL, SCORE_FLOOR
from xpart_complete import DEFAULT_MERGE_MAX_SHARE

# --- stages -----------------------------------------------------------------

STAGES = (
    ("paint", True, "flat colour on a grey model so SAM3 has something to read"),
    ("guidance", True, "multi-view SAM3 masks + review overlays"),
    ("split", True, "intersect N prompt-free full_seg samples into atoms"),
    ("units", True, "connected components + fuse remesh inner shells"),
    ("merge", False, "vote names onto units; gated by --merge"),
    ("complete", False, "close each part (hybrid by default), then bake; gated by --complete"),
)

# --- enums -----------------------------------------------------------------

FLAT_PAINT_MODES = ("auto", "on", "off")
MERGE_MODES = ("name", "unit", "fragments")  # merge_parts; segment_parts also accepts "off"
MERGE_MODES_ALL = ("name", "unit", "off", "fragments")
MIRROR_MODES = ("auto", "none", "x", "y", "z")
COMPLETE_MODES = ("off", "boxes", "full", "hybrid")
CONDITION_MODES = ("surface", "box", "collar")   # see xpart_complete.CONDITION_MODES

GRANULARITY = {
    "fine": (150, 300),
    "medium": (300, 600),
    "coarse": (800, 1600),
}

# --- defaults (the current configuration) -----------------------------------

DEFAULT_SAMPLES = 7
DEFAULT_AZIMUTH = 0.0
DEFAULT_AZIMUTH_JITTER = 30.0
DEFAULT_GRANULARITY = "medium"
DEFAULT_COLOR_TOL = 20.0
DEFAULT_MIRROR = "auto"
# The same ring the smart-mode proposal measures (auto_prompts.AUTO_*): a word SAM3
# found from three of eight directions used to get no vote at all on a 4-view grid.
DEFAULT_VIEW_AZIMUTHS = "0,45,90,135,180,225,270,315"
DEFAULT_VIEW_ELEVATIONS = "15"
DEFAULT_RADIUS = 2.0
DEFAULT_RESOLUTION = 512
DEFAULT_SAM3_THRESHOLD = 0.4            # BANK_THRESHOLD; 0.3 is the no-bank painter
DEFAULT_FLAT_PAINT = "auto"
DEFAULT_MERGE = "name"
DEFAULT_PROMPTS = ("主体", "底座")
DEFAULT_UNASSIGNED_TO = "body"
DEFAULT_COMPLETE = "hybrid"
DEFAULT_CONDITION = "collar"
DEFAULT_MIN_AREA_SHARE = 0.005
DEFAULT_FRAGMENT_SHARE = 0.01
DEFAULT_REDRAWS = 2
# What each full_seg sample writes: labels = one colour per face (all the split reads;
# skips the UV unwrap and 4K bake), textured = the original UV-unwrapped glb.
SAMPLE_EXPORT_MODES = ("labels", "textured")
DEFAULT_SAMPLE_EXPORT = "labels"
DEFAULT_HOLOPART_LARGE = "score"   # escape | always | score; see hybrid_complete.py
HOLOPART_LARGE_MODES = ("escape", "always", "score")
DEFAULT_REFINE = "masks"           # cut a voted unit where its own masks name a patch differently; off keeps one name per unit
REFINE_MODES = ("masks", "off")
DEFAULT_REFINE_MIN_SHARE = 0.05
DEFAULT_EXPORT_FROM = "source"     # cut parts from the source model (full resolution, no bake)
DEFAULT_AUTO_PROMPTS = True        # no prompts: propose part names from the concept bank
DEFAULT_MODE = "auto"              # auto: rule-based pick; smart (智能分割): Kimi reviews the candidates
MODES = ("auto", "smart")
EXPORT_FROM_MODES = ("source", "remesh")
DEFAULT_OCTREE_RESOLUTION = 512
DEFAULT_SEED = 42
DEFAULT_TEXTURE_SIZE = 2048
DEFAULT_MIN_RECALL = 0.5

def _first_existing(env, *candidates):
    """Env override wins, else the first candidate on this box, else the first candidate."""
    return os.environ.get(env) or next((p for p in candidates if os.path.exists(p)), candidates[0])


_ROOT = os.path.dirname(os.path.abspath(__file__))
# Concept bank: v6 (5928 training objects, min_count 5; HF Zaun1996/sam3-concept-bank v6_all/bank.pt)
# when installed, else the v3 bank this box shipped with.
DEFAULT_CONCEPT_BANK = _first_existing(
    "SEGVIGEN_CONCEPT_BANK",
    os.path.join(_ROOT, "weights", "concept_bank_v6", "bank.pt"),
    "/root/autodl-tmp/datasets/concept_bank_v3/bank.pt")
# How the per-view SAM3 masks become one disjoint map (sam3_multiview.py): paint = the v3
# score-threshold overlay; rank = that overlay edited by the EASE Mask RankGNN (drop a mask
# it scores below rank_drop, add one it scores at least rank_add); auto = rank unless the
# ranker deletes a prompt outright. rank/auto fall back to paint when no ranker is installed.
ASSIGN_MODES = ("paint", "rank", "auto")
DEFAULT_ASSIGN = os.environ.get("SEGVIGEN_ASSIGN", "paint")
DEFAULT_RANK_MODEL = _first_existing(
    "SEGVIGEN_RANK_MODEL",
    os.path.join(_ROOT, "weights", "mask_rank_v6", "rank_ease_mc5_epoch6.pt"))
if not os.path.exists(DEFAULT_RANK_MODEL):
    DEFAULT_RANK_MODEL = ""   # no ranker installed: assign=rank/auto fall back to paint
DEFAULT_RANK_DROP = 0.2
DEFAULT_RANK_ADD = 0.9
DEFAULT_PY_XPART = os.environ.get(
    "SEGVIGEN_PY_XPART", "/root/autodl-tmp/envs/xpart/bin/python")
DEFAULT_XPART_ROOT = os.environ.get(
    "SEGVIGEN_XPART_ROOT", "/root/autodl-tmp/Hunyuan3D-Part/XPart")
DEFAULT_XPART_WEIGHTS = os.environ.get(
    "SEGVIGEN_XPART_WEIGHTS", "/root/autodl-tmp/Hunyuan3D-Part/weights")
DEFAULT_PY_HOLOPART = os.environ.get(
    "SEGVIGEN_PY_HOLOPART", "/root/autodl-tmp/envs/holopart/bin/python")
DEFAULT_HOLOPART_ROOT = os.environ.get(
    "SEGVIGEN_HOLOPART_ROOT", "/root/autodl-tmp/HoloPart")
DEFAULT_HOLOPART_WEIGHTS = os.environ.get(
    "SEGVIGEN_HOLOPART_WEIGHTS",
    "/root/autodl-tmp/HoloPart/pretrained_weights/HoloPart")


def floors(granularity=DEFAULT_GRANULARITY, min_atom_faces=None, min_unit_faces=None):
    """Resolve the two size floors. An explicit value wins over the named preset."""
    if granularity not in GRANULARITY:
        raise ValueError(f"granularity must be one of {tuple(GRANULARITY)}, "
                         f"got {granularity!r}")
    atom, unit = GRANULARITY[granularity]
    return (atom if min_atom_faces is None else min_atom_faces,
            unit if min_unit_faces is None else min_unit_faces)


@dataclass
class PipelineOptions:
    """Every user-facing switch, with the current default.

    Paths (`glb`, `out`, `work_dir`, `split`) stay on the call, not here: they change
    every run. Everything that configures *how* the pipeline runs lives on this object.
    """
    unassigned_to: str | None = DEFAULT_UNASSIGNED_TO
    samples: int = DEFAULT_SAMPLES
    sample_export: str = DEFAULT_SAMPLE_EXPORT
    azimuth: float = DEFAULT_AZIMUTH
    azimuth_jitter: float = DEFAULT_AZIMUTH_JITTER
    granularity: str = DEFAULT_GRANULARITY
    min_atom_faces: int | None = None
    min_unit_faces: int | None = None
    color_tol: float | None = None
    mirror: str = DEFAULT_MIRROR
    min_recall: float | None = None
    view_azimuths: str = DEFAULT_VIEW_AZIMUTHS
    view_elevations: str = DEFAULT_VIEW_ELEVATIONS
    radius: float = DEFAULT_RADIUS
    resolution: int = DEFAULT_RESOLUTION
    flat_paint: str = DEFAULT_FLAT_PAINT
    merge: str = DEFAULT_MERGE
    complete: str = DEFAULT_COMPLETE
    condition: str = DEFAULT_CONDITION
    min_area_share: float = DEFAULT_MIN_AREA_SHARE
    fragment_share: float = DEFAULT_FRAGMENT_SHARE
    redraws: int = DEFAULT_REDRAWS
    octree_resolution: int = DEFAULT_OCTREE_RESOLUTION
    seed: int = DEFAULT_SEED
    with_texture: bool = True
    texture_size: int = DEFAULT_TEXTURE_SIZE
    reuse: bool = True
    strict_parts: bool = False
    sam3_threshold: float = DEFAULT_SAM3_THRESHOLD
    concept_bank: str = DEFAULT_CONCEPT_BANK
    assign: str = DEFAULT_ASSIGN
    rank_model: str = DEFAULT_RANK_MODEL
    rank_drop: float = DEFAULT_RANK_DROP
    rank_add: float = DEFAULT_RANK_ADD
    py_xpart: str | None = None
    xpart_root: str = DEFAULT_XPART_ROOT
    xpart_weights: str = DEFAULT_XPART_WEIGHTS
    py_holopart: str | None = None
    holopart_root: str = DEFAULT_HOLOPART_ROOT
    holopart_weights: str = DEFAULT_HOLOPART_WEIGHTS
    # complete=hybrid: how X-Part and HoloPart are chosen per instance (hybrid_complete.py)
    holopart_large: str = DEFAULT_HOLOPART_LARGE
    score_candidate: float = SCORE_CANDIDATE
    score_candidate_small: float = SCORE_CANDIDATE_SMALL
    score_floor: float = SCORE_FLOOR
    # what counts as one prompt (xpart_complete.py)
    part_min_area_share: str | None = None
    fold_within_part: bool = False
    merge_gap: float = 0.0
    merge_max_share: float = DEFAULT_MERGE_MAX_SHARE
    # after the vote: split a unit where SAM3's per-face masks name a coherent patch of
    # it differently (refine_units.py); cut parts from the source model (source_export.py)
    refine: str = DEFAULT_REFINE
    refine_min_share: float = DEFAULT_REFINE_MIN_SHARE
    export_from: str = DEFAULT_EXPORT_FROM
    # export each big connected instance of these part names as its own part ("leg", "leg 2");
    # comma list or "all". The guided flow fills it from the reference's colours.
    separate: str = ""
    # empty prompts: ask SAM3 for every bank concept and pick the part names (auto_prompts.py)
    auto_prompts: bool = DEFAULT_AUTO_PROMPTS
    mode: str = DEFAULT_MODE

    def resolved_floors(self):
        return floors(self.granularity, self.min_atom_faces, self.min_unit_faces)

    def public(self):
        """JSON-safe snapshot of the configuration a client should see."""
        atom, unit = self.resolved_floors()
        return {
            "stages": [{"name": n, "always": always, "what": what}
                       for n, always, what in STAGES],
            "switches": {
                "flat_paint": list(FLAT_PAINT_MODES),
                "merge": list(MERGE_MODES_ALL),
                "complete": list(COMPLETE_MODES),
                "condition": list(CONDITION_MODES),
                "granularity": {k: {"min_atom_faces": a, "min_unit_faces": u}
                                for k, (a, u) in GRANULARITY.items()},
                "mirror": list(MIRROR_MODES),
                "holopart_large": list(HOLOPART_LARGE_MODES),
                "refine": list(REFINE_MODES),
                "export_from": list(EXPORT_FROM_MODES),
                "mode": list(MODES),
            },
            "defaults": {
                **{f.name: getattr(self, f.name) for f in fields(self)
                   if f.name not in ("py_xpart", "xpart_root", "xpart_weights",
                                     "py_holopart", "holopart_root",
                                     "holopart_weights", "concept_bank")},
                "prompts": list(DEFAULT_PROMPTS),
                "min_atom_faces": atom,
                "min_unit_faces": unit,
                "color_tol": DEFAULT_COLOR_TOL if self.color_tol is None else self.color_tol,
                "min_recall": DEFAULT_MIN_RECALL if self.min_recall is None else self.min_recall,
            },
        }

    def segment_kwargs(self):
        """kwargs for segment_parts.segment_parts, minus glb / prompts / out_glb."""
        return {
            "samples": self.samples,
            "sample_export": self.sample_export,
            "azimuth": self.azimuth,
            "azimuth_jitter": self.azimuth_jitter,
            "color_tol": self.color_tol,
            "granularity": self.granularity,
            "min_atom_faces": self.min_atom_faces,
            "mirror": self.mirror,
            "min_unit_faces": self.min_unit_faces,
            "min_recall": self.min_recall,
            "view_azimuths": self.view_azimuths,
            "view_elevations": self.view_elevations,
            "radius": self.radius,
            "resolution": self.resolution,
            "sam3_threshold": self.sam3_threshold,
            "concept_bank": self.concept_bank,
            "assign": self.assign,
            "rank_model": self.rank_model,
            "rank_drop": self.rank_drop,
            "rank_add": self.rank_add,
            "flat_paint": self.flat_paint,
            "unassigned_to": self.unassigned_to,
            "merge": self.merge,
            "complete": self.complete,
            "py_xpart": self.py_xpart,
            "xpart_root": self.xpart_root,
            "xpart_weights": self.xpart_weights,
            "py_holopart": self.py_holopart,
            "holopart_root": self.holopart_root,
            "holopart_weights": self.holopart_weights,
            "octree_resolution": self.octree_resolution,
            "seed": self.seed,
            "condition": self.condition,
            "min_area_share": self.min_area_share,
            "fragment_share": self.fragment_share,
            "redraws": self.redraws,
            "reuse": self.reuse,
            "strict_parts": self.strict_parts,
            "with_texture": self.with_texture,
            "texture_size": self.texture_size,
            "holopart_large": self.holopart_large,
            "score_candidate": self.score_candidate,
            "score_candidate_small": self.score_candidate_small,
            "score_floor": self.score_floor,
            "part_min_area_share": self.part_min_area_share,
            "fold_within_part": self.fold_within_part,
            "merge_gap": self.merge_gap,
            "merge_max_share": self.merge_max_share,
            "refine": self.refine,
            "refine_min_share": self.refine_min_share,
            "export_from": self.export_from,
            "separate": self.separate,
            "auto_prompts": self.auto_prompts,
            "mode": self.mode,
        }

    def merge_kwargs(self):
        """kwargs for merge_parts.merge_parts, minus glb / prompts / split_dir / out_glb."""
        skip = {"samples", "sample_export", "azimuth", "azimuth_jitter", "color_tol",
                "granularity", "min_atom_faces", "mirror", "auto_prompts", "mode"}
        return {k: v for k, v in self.segment_kwargs().items() if k not in skip}

    @classmethod
    def from_mapping(cls, data):
        """Build from a HTTP/JSON dict. Unknown keys are ignored; None keeps the default."""
        known = {item.name for item in fields(cls)}
        payload = dict(data)
        kwargs = {}
        if payload.get("sample_export") is not None and payload["sample_export"] not in SAMPLE_EXPORT_MODES:
            raise ValueError(f"sample_export must be one of {SAMPLE_EXPORT_MODES}, got {payload['sample_export']!r}")
        if payload.get("assign") is not None and payload["assign"] not in ASSIGN_MODES:
            raise ValueError(f"assign must be one of {ASSIGN_MODES}, got {payload['assign']!r}")
        if "rank_model" in payload and payload.get("rank_model") in ("", "string", None):
            payload.pop("rank_model")
        if payload.pop("no_concept_bank", False):
            kwargs["concept_bank"] = ""
        if payload.get("strict_parts") is not None:
            kwargs["strict_parts"] = bool(payload.pop("strict_parts"))
        elif "allow_partial" in payload:
            kwargs["strict_parts"] = not bool(payload.pop("allow_partial"))
        if payload.get("concept_bank") in ("", "string"):
            payload.pop("concept_bank")
        if "unassigned_to" in payload:
            raw = payload.pop("unassigned_to")
            if raw == "":
                kwargs["unassigned_to"] = None
            elif raw not in (None, "string"):
                kwargs["unassigned_to"] = raw
        for key, value in payload.items():
            if key in known and key not in kwargs and value is not None:
                kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def from_namespace(cls, args):
        """Build from an argparse namespace produced by add_cli_arguments."""
        concept_bank = "" if getattr(args, "no_concept_bank", False) else getattr(
            args, "concept_bank", DEFAULT_CONCEPT_BANK)
        return cls(
            unassigned_to=(
                None if getattr(args, "unassigned_to", DEFAULT_UNASSIGNED_TO) == ""
                else getattr(args, "unassigned_to", DEFAULT_UNASSIGNED_TO)
            ),
            samples=getattr(args, "samples", DEFAULT_SAMPLES),
            sample_export=getattr(args, "sample_export", DEFAULT_SAMPLE_EXPORT),
            azimuth=getattr(args, "azimuth", DEFAULT_AZIMUTH),
            azimuth_jitter=getattr(args, "azimuth_jitter", DEFAULT_AZIMUTH_JITTER),
            granularity=getattr(args, "granularity", DEFAULT_GRANULARITY),
            min_atom_faces=getattr(args, "min_atom_faces", None),
            min_unit_faces=getattr(args, "min_unit_faces", None),
            color_tol=getattr(args, "color_tol", None),
            mirror=getattr(args, "mirror", DEFAULT_MIRROR),
            min_recall=getattr(args, "min_recall", None),
            view_azimuths=getattr(args, "view_azimuths", DEFAULT_VIEW_AZIMUTHS),
            view_elevations=getattr(args, "view_elevations", DEFAULT_VIEW_ELEVATIONS),
            radius=getattr(args, "radius", DEFAULT_RADIUS),
            resolution=getattr(args, "resolution", DEFAULT_RESOLUTION),
            flat_paint=getattr(args, "flat_paint", DEFAULT_FLAT_PAINT),
            merge=getattr(args, "merge", DEFAULT_MERGE),
            complete=getattr(args, "complete", DEFAULT_COMPLETE),
            condition=getattr(args, "condition", DEFAULT_CONDITION),
            min_area_share=getattr(args, "min_area_share", DEFAULT_MIN_AREA_SHARE),
            fragment_share=getattr(args, "fragment_share", DEFAULT_FRAGMENT_SHARE),
            redraws=getattr(args, "redraws", DEFAULT_REDRAWS),
            octree_resolution=getattr(args, "octree_resolution", DEFAULT_OCTREE_RESOLUTION),
            seed=getattr(args, "seed", DEFAULT_SEED),
            with_texture=not getattr(args, "no_texture", False),
            texture_size=getattr(args, "texture_size", DEFAULT_TEXTURE_SIZE),
            reuse=not getattr(args, "no_reuse", False),
            strict_parts=bool(getattr(args, "strict_parts", False))
            and not getattr(args, "allow_partial", False),
            sam3_threshold=getattr(args, "sam3_threshold", DEFAULT_SAM3_THRESHOLD),
            concept_bank=concept_bank,
            assign=getattr(args, "assign", DEFAULT_ASSIGN),
            rank_model=getattr(args, "rank_model", DEFAULT_RANK_MODEL),
            rank_drop=getattr(args, "rank_drop", DEFAULT_RANK_DROP),
            rank_add=getattr(args, "rank_add", DEFAULT_RANK_ADD),
            py_xpart=getattr(args, "py_xpart", None),
            xpart_root=getattr(args, "xpart_root", DEFAULT_XPART_ROOT),
            xpart_weights=getattr(args, "xpart_weights", DEFAULT_XPART_WEIGHTS),
            py_holopart=getattr(args, "py_holopart", None),
            holopart_root=getattr(args, "holopart_root", DEFAULT_HOLOPART_ROOT),
            holopart_weights=getattr(args, "holopart_weights", DEFAULT_HOLOPART_WEIGHTS),
            holopart_large=getattr(args, "holopart_large", DEFAULT_HOLOPART_LARGE),
            score_candidate=getattr(args, "score_candidate", SCORE_CANDIDATE),
            score_candidate_small=getattr(args, "score_candidate_small", SCORE_CANDIDATE_SMALL),
            score_floor=getattr(args, "score_floor", SCORE_FLOOR),
            part_min_area_share=getattr(args, "part_min_area_share", None),
            fold_within_part=bool(getattr(args, "fold_within_part", False)),
            merge_gap=getattr(args, "merge_gap", 0.0),
            merge_max_share=getattr(args, "merge_max_share", DEFAULT_MERGE_MAX_SHARE),
            refine=getattr(args, "refine", DEFAULT_REFINE),
            refine_min_share=getattr(args, "refine_min_share", DEFAULT_REFINE_MIN_SHARE),
            export_from=getattr(args, "export_from", DEFAULT_EXPORT_FROM),
            separate=getattr(args, "separate", "") or "",
            auto_prompts=not getattr(args, "no_auto_prompts", False),
            mode=getattr(args, "mode", DEFAULT_MODE),
        )


def add_cli_arguments(parser, *, split=True, merge_off=True):
    """Attach the shared switches. `split=False` is the merge_parts half."""
    if split:
        parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                            help="full_seg samples to intersect; each costs one flow-model run")
        parser.add_argument("--azimuth", type=float, default=DEFAULT_AZIMUTH,
                            help="Degrees to orbit the conditioning camera for the base sample")
        parser.add_argument("--azimuth_jitter", type=float, default=DEFAULT_AZIMUTH_JITTER,
                            help="How far the extra samples orbit either side of --azimuth")
        parser.add_argument("--color_tol", type=float, default=None,
                            help="RGB distance separating two colours within one sample")
        parser.add_argument("--granularity", default=DEFAULT_GRANULARITY,
                            choices=tuple(GRANULARITY),
                            help="Both size floors at once (%s). An explicit floor wins."
                                 % ", ".join(f"{k} {a}/{u}" for k, (a, u) in GRANULARITY.items()))
        parser.add_argument("--min_atom_faces", type=int, default=None,
                            help="Slivers under this many faces join their majority neighbour")
        parser.add_argument("--mirror", default=DEFAULT_MIRROR, choices=MIRROR_MODES,
                            help="Also intersect each sample reflected across this plane")
    parser.add_argument("--min_unit_faces", type=int, default=None,
                        help="Connected components under this many faces are not voted on alone")
    parser.add_argument("--min_recall", type=float, default=None,
                        help="A mask claims a unit once it covers this share of the unit's pixels")
    parser.add_argument("--view_azimuths", default=DEFAULT_VIEW_AZIMUTHS)
    parser.add_argument("--view_elevations", default=DEFAULT_VIEW_ELEVATIONS)
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--unassigned_to", default=DEFAULT_UNASSIGNED_TO,
                        help="Part that absorbs units no concept claimed "
                             f"(default {DEFAULT_UNASSIGNED_TO}). Empty disables.")
    merge_choices = MERGE_MODES_ALL if merge_off else MERGE_MODES
    parser.add_argument("--merge", default=DEFAULT_MERGE, choices=merge_choices,
                        help="name = one node per prompt. unit = one node per voted unit. "
                             "fragments = keep the geometric split, fold only specks. "
                             + ("off = stop after the units." if merge_off else ""))
    parser.add_argument("--flat_paint", default=DEFAULT_FLAT_PAINT, choices=FLAT_PAINT_MODES,
                        help="Temporary flat colour for a model the renders show as grey")
    parser.add_argument("--complete", default=DEFAULT_COMPLETE, choices=COMPLETE_MODES,
                        help="off | boxes (prompts only) | full (X-Part only) | "
                             "hybrid (X-Part, HoloPart on large box-escapees; default)")
    parser.add_argument("--condition", default=DEFAULT_CONDITION, choices=CONDITION_MODES,
                        help="surface = faces the split assigned; box = whatever is in the box")
    parser.add_argument("--min_area_share", type=float, default=DEFAULT_MIN_AREA_SHARE,
                        help="Fold an X-Part component below this share of the surface "
                             "into its nearest neighbour instead of generating it alone")
    parser.add_argument("--fragment_share", type=float, default=DEFAULT_FRAGMENT_SHARE,
                        help="merge=fragments: fold a unit below this share of the surface "
                             "into a neighbour. Smaller keeps more pieces.")
    parser.add_argument("--redraws", type=int, default=DEFAULT_REDRAWS,
                        help="Times to redraw an X-Part solid that overruns its box")
    parser.add_argument("--holopart_large", default=DEFAULT_HOLOPART_LARGE,
                        choices=HOLOPART_LARGE_MODES,
                        help="complete=hybrid: escape = HoloPart only for a large solid that "
                             "left its box; always = every large one; score = measure each "
                             "X-Part solid against its open surface, draw HoloPart for the low "
                             "scorers and keep the better one (default)")
    parser.add_argument("--score_candidate", type=float, default=SCORE_CANDIDATE,
                        help="holopart_large=score: a large instance below this also gets "
                             "a HoloPart draw")
    parser.add_argument("--score_candidate_small", type=float, default=SCORE_CANDIDATE_SMALL,
                        help="holopart_large=score: the same for a small instance")
    parser.add_argument("--score_floor", type=float, default=SCORE_FLOOR,
                        help="holopart_large=score: both solids below this keep the open "
                             "surface instead")
    parser.add_argument("--part_min_area_share", default=None,
                        help="Per-part --min_area_share as name=share[,name=share] "
                             "(ornaments=0.001 keeps every bauble its own prompt)")
    parser.add_argument("--fold_within_part", action="store_true",
                        help="Fold a small component only into its own part (many small "
                             "things stuck on a big one: ornaments on a tree)")
    parser.add_argument("--merge_gap", type=float, default=0.0,
                        help="Rejoin same-part pieces whose surfaces come within this share "
                             "of the model diagonal (a hand the staff cut in two). 0 = off")
    parser.add_argument("--merge_max_share", type=float, default=DEFAULT_MERGE_MAX_SHARE,
                        help="merge_gap: only a piece under this share of the surface "
                             "joins a neighbour")
    parser.add_argument("--refine", default=DEFAULT_REFINE, choices=REFINE_MODES,
                        help="masks (default): after the vote, cut a unit where SAM3's "
                             "per-face masks name a coherent patch of it differently (the "
                             "back of a hand fused with the gauntlet); off keeps one name "
                             "per unit")
    parser.add_argument("--refine_min_share", type=float, default=DEFAULT_REFINE_MIN_SHARE,
                        help="refine=masks: a patch is cut off only above this share of "
                             "its unit's area")
    parser.add_argument("--mode", default=DEFAULT_MODE, choices=MODES,
                        help="auto: pick the part names from the bank candidates by rule; "
                             "smart (智能分割模式): Kimi reviews the candidates and the renders "
                             "and picks them (needs MOONSHOT_API_KEY)")
    parser.add_argument("--no_auto_prompts", action="store_true",
                        help="Without prompts, name the parts 主体 / 底座 instead of asking "
                             "SAM3 for every concept in the bank and picking the part names")
    parser.add_argument("--separate", default="",
                        help="Comma list of part names (or all) whose big connected instances are exported as separate parts: leg, leg 2 ...")
    parser.add_argument("--export_from", default=DEFAULT_EXPORT_FROM, choices=EXPORT_FROM_MODES,
                        help="source (default): cut the parts from the source model with "
                             "its own UVs and texture (full resolution, no bake); remesh: "
                             "cut the TRELLIS remesh and bake the texture back")
    parser.add_argument("--no_reuse", action="store_true",
                        help="Re-run every stage instead of reusing cached intermediates")
    parser.add_argument("--allow_partial", action="store_true",
                        help="Default: a prompt SAM3 never sees is skipped. Kept for "
                             "compatibility; the run already continues without it.")
    parser.add_argument("--strict_parts", action="store_true",
                        help="Fail if a requested name got no mask or no faces, instead "
                             "of skipping that prompt and finishing the rest")
    parser.add_argument("--no_texture", action="store_true",
                        help="Skip the Blender bake; parts get a flat placeholder colour")
    parser.add_argument("--texture_size", type=int, default=DEFAULT_TEXTURE_SIZE,
                        help="Small-part bake atlas; larger parts scale up to 8K")
    parser.add_argument("--sam3_threshold", type=float, default=DEFAULT_SAM3_THRESHOLD)
    parser.add_argument("--concept_bank", default=DEFAULT_CONCEPT_BANK,
                        help="SAM3 v3 bank.pt. Empty = raw SAM3.")
    parser.add_argument("--no_concept_bank", action="store_true",
                        help="Disable the v3 bank and fall back to raw SAM3 scores")
    parser.add_argument("--sample_export", default=DEFAULT_SAMPLE_EXPORT, choices=SAMPLE_EXPORT_MODES,
                        help="labels (default) = full_seg samples carry one colour per face, "
                             "no UV unwrap or bake; textured = the original textured glb")
    parser.add_argument("--assign", default=DEFAULT_ASSIGN, choices=ASSIGN_MODES,
                        help="paint = v3 score-threshold overlay; rank = overlay edited by the "
                             "EASE Mask RankGNN; auto = rank unless it deletes a prompt")
    parser.add_argument("--rank_model", default=DEFAULT_RANK_MODEL,
                        help="mask_rank.py checkpoint for --assign rank/auto "
                             "(default: $SEGVIGEN_RANK_MODEL; empty = fall back to paint)")
    parser.add_argument("--rank_drop", type=float, default=DEFAULT_RANK_DROP,
                        help="--assign rank: drop an overlaid mask the ranker scores below this")
    parser.add_argument("--rank_add", type=float, default=DEFAULT_RANK_ADD,
                        help="--assign rank: add a mask the overlay skipped that scores at least this")
    parser.add_argument("--py_xpart", default=None, help=f"default: {DEFAULT_PY_XPART}")
    parser.add_argument("--xpart_root", default=DEFAULT_XPART_ROOT)
    parser.add_argument("--xpart_weights", default=DEFAULT_XPART_WEIGHTS)
    parser.add_argument("--py_holopart", default=None, help=f"default: {DEFAULT_PY_HOLOPART}")
    parser.add_argument("--holopart_root", default=DEFAULT_HOLOPART_ROOT)
    parser.add_argument("--holopart_weights", default=DEFAULT_HOLOPART_WEIGHTS)
    parser.add_argument("--octree_resolution", type=int, default=DEFAULT_OCTREE_RESOLUTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def check_cli(parser, args):
    if getattr(args, "no_concept_bank", False) and getattr(
            args, "concept_bank", DEFAULT_CONCEPT_BANK) != DEFAULT_CONCEPT_BANK:
        parser.error("pass either --concept_bank or --no_concept_bank, not both")
