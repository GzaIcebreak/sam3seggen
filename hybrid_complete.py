"""X-Part first; swap a large, box-escaping solid for that instance's HoloPart draw.

HoloPart is only loaded when at least one instance qualifies. Pairing is by the
``00_name`` prefix written into ``xpart_instances.glb`` / ``open_instances.glb``.
"""
from __future__ import annotations

import json
import os

import numpy as np
import trimesh

from xpart_complete import box_escape, group_solids, load_part_nodes

ESCAPE_LIMIT = 0.5
AREA_LARGE = 0.08
AXIS_LARGE = 0.55
# "escape": HoloPart only when a large X-Part solid left its box (the original rule).
# "always": every large instance goes to HoloPart. X-Part can stay inside the box and still
# be wrong: on a decorated tree the 63% trunk came back a smooth blob with 0% escape.
LARGE_POLICIES = ("escape", "always")


def source_extent(boxes):
    lows = np.min([np.array(row["box"])[0] for row in boxes], axis=0)
    highs = np.max([np.array(row["box"])[1] for row in boxes], axis=0)
    return highs - lows


def instance_index(name):
    prefix = str(name).split("_", 1)[0]
    return int(prefix) if prefix.isdigit() else None


def index_by_instance(nodes):
    mapping = {}
    for name, mesh in nodes:
        index = instance_index(name)
        if index is not None:
            mapping[index] = (name, mesh)
    if mapping:
        return mapping
    return {index: node for index, node in enumerate(nodes)}


def is_large(row, src_ext, area_large=AREA_LARGE, axis_large=AXIS_LARGE):
    box = np.array(row["box"])
    axis = float(np.max((box[1] - box[0]) / np.maximum(src_ext, 1e-9)))
    return row["area_share"] >= area_large or axis >= axis_large, axis


def decide_backend(row, generated_bounds, src_ext, escape_limit=ESCAPE_LIMIT,
                   area_large=AREA_LARGE, axis_large=AXIS_LARGE, large_policy="escape"):
    """One instance: HoloPart if it is large and (policy "escape") the X-Part solid left its box."""
    if large_policy not in LARGE_POLICIES:
        raise ValueError(f"large_policy must be one of {LARGE_POLICIES}, got {large_policy!r}")
    if generated_bounds is None:
        escape = float("inf")
    else:
        escape = box_escape(generated_bounds, np.array(row["box"]))
    large, axis = is_large(row, src_ext, area_large, axis_large)
    use_holo = large and (large_policy == "always" or escape > escape_limit)
    return {
        "name": row["name"],
        "instance": row.get("instance"),
        "escape": escape if np.isfinite(escape) else None,
        "area_share": row["area_share"],
        "max_axis_of_source": axis,
        "large": large,
        "backend": "holopart" if use_holo else "xpart",
    }


def decisions_from_instances(instances_glb, boxes, escape_limit=ESCAPE_LIMIT,
                             area_large=AREA_LARGE, axis_large=AXIS_LARGE,
                             large_policy="escape"):
    src_ext = source_extent(boxes)
    indexed = index_by_instance(load_part_nodes(instances_glb))
    decisions = []
    for row in boxes:
        inst = row.get("instance")
        node = indexed.get(inst) if inst is not None else None
        bounds = None if node is None else node[1].bounds
        decision = decide_backend(
            row, bounds, src_ext, escape_limit, area_large, axis_large, large_policy)
        decision["xpart_node"] = None if node is None else node[0]
        decisions.append(decision)
    return decisions


def instance_node_name(row):
    """The ``00_name`` node open_instances.glb / xpart_instances.glb use for this row."""
    return f"{int(row['instance']):02d}_{row['name']}"


def drop_duplicate_faces(mesh):
    """Remove faces that repeat another face's three vertices (in any order).

    HoloPart's pymeshlab simplification left 93 of them on a tree trunk. Blender drops
    them on import, the loop count no longer matches the UV array the bake wrote, and the
    glTF exporter then omits the whole part ("Array length mismatch").
    """
    faces = np.asarray(mesh.faces)
    if not len(faces):
        return mesh
    _, first = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    if len(first) == len(faces):
        return mesh
    mesh = mesh.copy()
    mesh.update_faces(np.sort(first))
    mesh.remove_unreferenced_vertices()
    return mesh


def pick_solids(xpart_nodes, holo_nodes, boxes, decisions):
    xpart = index_by_instance(xpart_nodes)
    holo = index_by_instance(holo_nodes) if holo_nodes else {}
    names, solids = [], []
    for row, decision in zip(boxes, decisions):
        inst = row.get("instance")
        xnode = xpart.get(inst)
        hnode = holo.get(inst)
        if decision["backend"] == "holopart" and hnode is not None:
            mesh = hnode[1]
        elif xnode is not None:
            mesh = xnode[1]
        elif hnode is not None:
            mesh = hnode[1]
        else:
            continue
        names.append(row["name"])
        solids.append(drop_duplicate_faces(mesh))
    return names, solids


def write_assembled(out_dir, names, solids, grouped_name="xpart_parts.glb"):
    instances = trimesh.Scene()
    for index, (name, mesh) in enumerate(zip(names, solids)):
        instances.add_geometry(mesh, geom_name=f"{index:02d}_{name}")
    instances_path = os.path.join(out_dir, "hybrid_instances.glb")
    instances.export(instances_path)
    grouped = trimesh.Scene()
    for name, mesh in group_solids(names, solids):
        grouped.add_geometry(mesh, geom_name=name)
    grouped_path = os.path.join(out_dir, grouped_name)
    grouped.export(grouped_path)
    return grouped_path, instances_path


def apply_hybrid(out_dir, holopart_glb=None, escape_limit=ESCAPE_LIMIT,
                 area_large=AREA_LARGE, axis_large=AXIS_LARGE, large_policy="escape"):
    """Read X-Part artifacts, optionally a HoloPart glb, write decisions + assembled parts.

    Returns (decisions, swapped). ``holopart_glb`` may be omitted when nothing needs it.
    """
    boxes_path = os.path.join(out_dir, "boxes.json")
    instances_glb = os.path.join(out_dir, "xpart_instances.glb")
    with open(boxes_path, "r", encoding="utf-8") as handle:
        boxes = json.load(handle)
    decisions = decisions_from_instances(
        instances_glb, boxes, escape_limit, area_large, axis_large, large_policy)
    holo_nodes = load_part_nodes(holopart_glb) if holopart_glb else []
    holo_index = index_by_instance(holo_nodes) if holo_nodes else {}
    missing = [row["name"] for row in decisions
               if row["backend"] == "holopart" and row.get("instance") not in holo_index]
    if missing:
        raise SystemExit(f"HoloPart missing instance(s): {missing}")
    names, solids = pick_solids(
        load_part_nodes(instances_glb), holo_nodes, boxes, decisions)
    write_assembled(out_dir, names, solids)
    with open(os.path.join(out_dir, "decisions.json"), "w", encoding="utf-8") as handle:
        json.dump(decisions, handle, indent=2)
    swapped = sum(1 for row in decisions if row["backend"] == "holopart")
    for row in decisions:
        mark = "HOLOPART" if row["backend"] == "holopart" else "xpart"
        escape = "   n/a" if row["escape"] is None else f"{row['escape']:6.1%}"
        print(f"  {row['name']:<18} escape={escape}  area={row['area_share']:.1%}  "
              f"axis={row['max_axis_of_source']:.2f}  -> {mark}")
    print(f"[hybrid] {swapped}/{len(decisions)} solids from HoloPart")
    return decisions, swapped


# --- score policy -----------------------------------------------------------------------
# A solid is scored against the open surface it was generated from, in units of its box
# diagonal. Box escape alone missed the decorated tree's trunk: X-Part returned a smooth
# blob that stayed inside the box (0% escape) yet covered only 80% of the branches within
# 2% of the diagonal, where HoloPart covered 97%.
SCORE_TAU = 0.02          # a surface point counts as covered within this share of the diagonal
SCORE_EXTRA_SCALE = 0.2   # invented geometry this far out (p90) zeroes the score
SCORE_CANDIDATE = 0.8     # a large instance below this also gets a HoloPart draw to compare
SCORE_CANDIDATE_SMALL = 0.6  # a small one; X-Part is at home on small parts, so ask less often
SCORE_FLOOR = 0.3         # both below this: keep the open surface instead of either solid
SCORE_SAMPLES = 20000


def solid_metrics(surface, solid, box, samples=SCORE_SAMPLES, tau=SCORE_TAU, seed=0):
    """Fit of a generated solid to the open part surface it should close. None if unscorable.

    cover     share of surface samples within tau of the solid (did it keep the input?)
    fit_p90   p90 surface -> solid distance
    extra_p90 p90 solid -> surface distance (did it invent shape that is not there?)
    escape    how far the solid overruns the prompt box
    """
    from scipy.spatial import cKDTree

    if surface is None or solid is None or not len(surface.faces) or not len(solid.faces):
        return None
    if surface.area <= 0 or solid.area <= 0:
        return None
    box = np.asarray(box, dtype=float)
    diag = max(float(np.linalg.norm(box[1] - box[0])), 1e-9)
    part_points = trimesh.sample.sample_surface(surface, samples, seed=seed)[0]
    dense = trimesh.sample.sample_surface(solid, samples * 10, seed=seed)[0]
    solid_points = trimesh.sample.sample_surface(solid, samples, seed=seed + 1)[0]
    fit = cKDTree(dense).query(part_points)[0] / diag
    extra = cKDTree(part_points).query(solid_points)[0] / diag
    return {
        "cover": float((fit < tau).mean()),
        "fit_p90": float(np.percentile(fit, 90)),
        "extra_p90": float(np.percentile(extra, 90)),
        "escape": float(box_escape(solid.bounds, box)),
        "largest_shell": largest_shell_share(solid),
    }


def largest_shell_share(solid):
    """Share of the solid's area in its biggest connected shell (1.0 = one piece).

    Distance alone cannot see that a solid is in tatters: HoloPart returned the monk's
    left hand as 3,012 slivers that all lie on the original surface and so scored
    better than X-Part's clean fist. A repaired part is one closed thing; a crumb pile is not.
    """
    welded = trimesh.Trimesh(np.asarray(solid.vertices), np.asarray(solid.faces), process=True)
    if not len(welded.faces):
        return 0.0
    shells = welded.split(only_watertight=False)
    if len(shells) <= 1:
        return 1.0
    return float(max(shell.area for shell in shells) / welded.area)


def quality_score(metrics, extra_scale=SCORE_EXTRA_SCALE):
    """0..1: coverage, discounted by invented geometry and by box escape."""
    if not metrics:
        return 0.0
    extra = min(1.0, metrics["extra_p90"] / extra_scale)
    escape = min(1.0, metrics["escape"])
    whole = metrics.get("largest_shell", 1.0)
    return float(metrics["cover"] * (1.0 - extra) * (1.0 - escape) * whole)


def _score_inputs(out_dir):
    with open(os.path.join(out_dir, "boxes.json"), "r", encoding="utf-8") as handle:
        boxes = json.load(handle)
    opened = index_by_instance(load_part_nodes(os.path.join(out_dir, "open_instances.glb")))
    xpart = index_by_instance(load_part_nodes(os.path.join(out_dir, "xpart_instances.glb")))
    return boxes, opened, xpart


def score_candidates(out_dir, candidate=SCORE_CANDIDATE, candidate_small=SCORE_CANDIDATE_SMALL):
    """Score every X-Part solid; flag the ones that should also get a HoloPart draw.

    A large instance (is_large: area or extent) is asked at `candidate`, a small one at
    `candidate_small`. On the decorated tree one flat 0.6 skipped the trunk (X-Part 0.70,
    a blob) while 0.8 everywhere sent half the ornaments to HoloPart for nothing.
    """
    boxes, opened, xpart = _score_inputs(out_dir)
    src_ext = source_extent(boxes)
    decisions = []
    for row in boxes:
        inst = row.get("instance")
        surface, solid = opened.get(inst), xpart.get(inst)
        metrics = solid_metrics(surface and surface[1], solid and solid[1], row["box"])
        score = quality_score(metrics)
        large, _ = is_large(row, src_ext)
        threshold = candidate if large else candidate_small
        decisions.append({
            "name": row["name"], "instance": inst, "area_share": row["area_share"],
            "node": instance_node_name(row), "large": large, "threshold": threshold,
            "xpart": metrics, "q_xpart": score, "candidate": score < threshold,
        })
    flagged = sum(d["candidate"] for d in decisions)
    print(f"[score] {flagged}/{len(decisions)} X-Part solids below their threshold "
          f"(large < {candidate}, small < {candidate_small}) -> HoloPart draw")
    return decisions


def apply_scored(out_dir, decisions, holopart_glb=None, floor=SCORE_FLOOR):
    """Keep the higher-scoring solid per instance; below `floor` fall back to the open surface."""
    boxes, opened, xpart = _score_inputs(out_dir)
    holo = index_by_instance(load_part_nodes(holopart_glb)) if holopart_glb else {}
    missing = [d["node"] for d in decisions if d["candidate"] and d["instance"] not in holo]
    if missing:
        raise SystemExit(f"HoloPart missing instance(s): {missing}")
    names, solids = [], []
    for row, decision in zip(boxes, decisions):
        inst = row.get("instance")
        options = []
        if inst in xpart:
            options.append(("xpart", decision["q_xpart"], xpart[inst][1]))
        if decision["candidate"]:
            metrics = solid_metrics(opened[inst][1], holo[inst][1], row["box"])
            decision["holopart"] = metrics
            decision["q_holopart"] = quality_score(metrics)
            options.append(("holopart", decision["q_holopart"], holo[inst][1]))
        backend, score, mesh = max(options, key=lambda option: option[1]) if options \
            else ("open", 0.0, None)
        if score < floor:
            if inst not in opened:
                continue
            backend, mesh = "open", opened[inst][1]
        decision["backend"], decision["score"] = backend, score
        names.append(row["name"])
        solids.append(drop_duplicate_faces(mesh))
    write_assembled(out_dir, names, solids)
    with open(os.path.join(out_dir, "decisions.json"), "w", encoding="utf-8") as handle:
        json.dump(decisions, handle, indent=2, ensure_ascii=False)
    for d in decisions:
        holo_text = f"{d['q_holopart']:.2f}" if "q_holopart" in d else "  - "
        print(f"  {d['node']:<18} area={d['area_share']:.1%}  q_xpart={d['q_xpart']:.2f}  "
              f"q_holo={holo_text}  -> {d.get('backend', 'skipped').upper()}")
    counts = {key: sum(d.get("backend") == key for d in decisions)
              for key in ("xpart", "holopart", "open")}
    print(f"[score] {counts['xpart']} X-Part, {counts['holopart']} HoloPart, "
          f"{counts['open']} open surface (floor {floor})")
    return decisions
