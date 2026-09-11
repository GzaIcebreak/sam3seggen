"""The current pipeline, as one options object shared by the CLI and the HTTP API.

    paint -> guidance -> split -> units -> [merge] -> [complete -> bake]

Geometry decides every boundary; language only names. Stages 3 and 6 cost GPU minutes,
the rest is seconds once the renders are cached. `--merge off` stops after units;
`--complete off` (the default) stops after the open, textured parts.glb.

This module is the contract. segment_parts.py / merge_parts.py / serve_api.py read their
defaults and switches from here so a knob cannot drift between the CLI and the API.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields

# --- stages -----------------------------------------------------------------

STAGES = (
    ("paint", True, "flat colour on a grey model so SAM3 has something to read"),
    ("guidance", True, "multi-view SAM3 masks + review overlays"),
    ("split", True, "intersect N prompt-free full_seg samples into atoms"),
    ("units", True, "connected components + fuse remesh inner shells"),
    ("merge", False, "vote names onto units; gated by --merge"),
    ("complete", False, "X-Part closes each part, then bake; gated by --complete"),
)

# --- enums -----------------------------------------------------------------

FLAT_PAINT_MODES = ("auto", "on", "off")
MERGE_MODES = ("name", "unit")          # merge_parts; segment_parts also accepts "off"
MERGE_MODES_ALL = ("name", "unit", "off")
MIRROR_MODES = ("auto", "none", "x", "y", "z")
COMPLETE_MODES = ("off", "boxes", "full")
CONDITION_MODES = ("surface", "box")

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
DEFAULT_VIEW_AZIMUTHS = "45,225"
DEFAULT_VIEW_ELEVATIONS = "10"
DEFAULT_RADIUS = 2.0
DEFAULT_RESOLUTION = 512
DEFAULT_SAM3_THRESHOLD = 0.4            # BANK_THRESHOLD; 0.3 is the no-bank painter
DEFAULT_FLAT_PAINT = "auto"
DEFAULT_MERGE = "name"
DEFAULT_COMPLETE = "off"
DEFAULT_CONDITION = "surface"
DEFAULT_MIN_AREA_SHARE = 0.005
DEFAULT_REDRAWS = 2
DEFAULT_OCTREE_RESOLUTION = 512
DEFAULT_SEED = 42
DEFAULT_TEXTURE_SIZE = 2048
DEFAULT_MIN_RECALL = 0.5

DEFAULT_CONCEPT_BANK = os.environ.get(
    "SEGVIGEN_CONCEPT_BANK", "/root/autodl-tmp/datasets/concept_bank_v3/bank.pt")
DEFAULT_PY_XPART = os.environ.get(
    "SEGVIGEN_PY_XPART", "/root/autodl-tmp/envs/xpart/bin/python")
DEFAULT_XPART_ROOT = os.environ.get(
    "SEGVIGEN_XPART_ROOT", "/root/autodl-tmp/Hunyuan3D-Part/XPart")
DEFAULT_XPART_WEIGHTS = os.environ.get(
    "SEGVIGEN_XPART_WEIGHTS", "/root/autodl-tmp/Hunyuan3D-Part/weights")


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
    unassigned_to: str | None = None
    samples: int = DEFAULT_SAMPLES
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
    redraws: int = DEFAULT_REDRAWS
    octree_resolution: int = DEFAULT_OCTREE_RESOLUTION
    seed: int = DEFAULT_SEED
    with_texture: bool = True
    texture_size: int = DEFAULT_TEXTURE_SIZE
    reuse: bool = True
    strict_parts: bool = False
    sam3_threshold: float = DEFAULT_SAM3_THRESHOLD
    concept_bank: str = DEFAULT_CONCEPT_BANK
    py_xpart: str | None = None
    xpart_root: str = DEFAULT_XPART_ROOT
    xpart_weights: str = DEFAULT_XPART_WEIGHTS

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
            },
            "defaults": {
                **{f.name: getattr(self, f.name) for f in fields(self)
                   if f.name not in ("py_xpart", "xpart_root", "xpart_weights",
                                     "concept_bank")},
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
            "flat_paint": self.flat_paint,
            "unassigned_to": self.unassigned_to,
            "merge": self.merge,
            "complete": self.complete,
            "py_xpart": self.py_xpart,
            "xpart_root": self.xpart_root,
            "xpart_weights": self.xpart_weights,
            "octree_resolution": self.octree_resolution,
            "seed": self.seed,
            "condition": self.condition,
            "min_area_share": self.min_area_share,
            "redraws": self.redraws,
            "reuse": self.reuse,
            "strict_parts": self.strict_parts,
            "with_texture": self.with_texture,
            "texture_size": self.texture_size,
        }

    def merge_kwargs(self):
        """kwargs for merge_parts.merge_parts, minus glb / prompts / split_dir / out_glb."""
        skip = {"samples", "azimuth", "azimuth_jitter", "color_tol",
                "granularity", "min_atom_faces", "mirror"}
        return {k: v for k, v in self.segment_kwargs().items() if k not in skip}

    @classmethod
    def from_mapping(cls, data):
        """Build from a HTTP/JSON dict. Unknown keys are ignored; None keeps the default."""
        known = {item.name for item in fields(cls)}
        payload = dict(data)
        kwargs = {}
        if payload.pop("no_concept_bank", False):
            kwargs["concept_bank"] = ""
        if payload.get("strict_parts") is not None:
            kwargs["strict_parts"] = bool(payload.pop("strict_parts"))
        elif "allow_partial" in payload:
            kwargs["strict_parts"] = not bool(payload.pop("allow_partial"))
        if payload.get("unassigned_to") == "":
            payload["unassigned_to"] = None
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
            unassigned_to=getattr(args, "unassigned_to", None),
            samples=getattr(args, "samples", DEFAULT_SAMPLES),
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
            py_xpart=getattr(args, "py_xpart", None),
            xpart_root=getattr(args, "xpart_root", DEFAULT_XPART_ROOT),
            xpart_weights=getattr(args, "xpart_weights", DEFAULT_XPART_WEIGHTS),
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
    parser.add_argument("--unassigned_to", default=None,
                        help="Part that absorbs units no concept claimed")
    merge_choices = MERGE_MODES_ALL if merge_off else MERGE_MODES
    parser.add_argument("--merge", default=DEFAULT_MERGE, choices=merge_choices,
                        help="name = one node per prompt. unit = one node per voted unit. "
                             + ("off = stop after the units." if merge_off else ""))
    parser.add_argument("--flat_paint", default=DEFAULT_FLAT_PAINT, choices=FLAT_PAINT_MODES,
                        help="Temporary flat colour for a model the renders show as grey")
    parser.add_argument("--complete", default=DEFAULT_COMPLETE, choices=COMPLETE_MODES,
                        help="off | boxes (prompts only) | full (regenerate + bake)")
    parser.add_argument("--condition", default=DEFAULT_CONDITION, choices=CONDITION_MODES,
                        help="surface = faces the split assigned; box = whatever is in the box")
    parser.add_argument("--min_area_share", type=float, default=DEFAULT_MIN_AREA_SHARE,
                        help="Fold an X-Part component below this share of the surface "
                             "into its nearest neighbour instead of generating it alone")
    parser.add_argument("--redraws", type=int, default=DEFAULT_REDRAWS,
                        help="Times to redraw an X-Part solid that overruns its box")
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
    parser.add_argument("--texture_size", type=int, default=DEFAULT_TEXTURE_SIZE)
    parser.add_argument("--sam3_threshold", type=float, default=DEFAULT_SAM3_THRESHOLD)
    parser.add_argument("--concept_bank", default=DEFAULT_CONCEPT_BANK,
                        help="SAM3 v3 bank.pt. Empty = raw SAM3.")
    parser.add_argument("--no_concept_bank", action="store_true",
                        help="Disable the v3 bank and fall back to raw SAM3 scores")
    parser.add_argument("--py_xpart", default=None, help=f"default: {DEFAULT_PY_XPART}")
    parser.add_argument("--xpart_root", default=DEFAULT_XPART_ROOT)
    parser.add_argument("--xpart_weights", default=DEFAULT_XPART_WEIGHTS)
    parser.add_argument("--octree_resolution", type=int, default=DEFAULT_OCTREE_RESOLUTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def check_cli(parser, args):
    if getattr(args, "no_concept_bank", False) and getattr(
            args, "concept_bank", DEFAULT_CONCEPT_BANK) != DEFAULT_CONCEPT_BANK:
        parser.error("pass either --concept_bank or --no_concept_bank, not both")
