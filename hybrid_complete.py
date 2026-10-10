"""X-Part first; swap a large, box-escaping solid for that instance's HoloPart draw.

HoloPart is only loaded when at least one instance qualifies. Pairing is by the
``00_name`` prefix written into ``xpart_instances.glb`` / ``open_instances.glb``.
"""
from __future__ import annotations

import json
import os

import numpy as np
import trimesh

from xpart_complete import (
    box_escape, group_solids, hollow_share, load_part_nodes, plug_metrics, thin_faces,
)

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
SCORE_EXTRA_SCALE = 0.35  # invented geometry this far out (p90) zeroes the score; a plug
                          # closing a thigh sits ~0.1 of the diagonal off the open surface
ESCAPE_FREE = 0.2         # a solid may reach this far past its box before it counts (the plug)
RIM_BAND = 0.16           # within this x diagonal of the part's own cut rim: the collar zone
HOLOPART_MARGIN = 0.2     # HoloPart replaces an X-Part solid only when ahead by this much ...
SOCKET_FREE = 0.10        # mean plug height (loop diameters) a solid may fall short of unpenalised
SOCKET_SCALE = 0.40       # ... and the shortfall at which the socket penalty reaches 1
HOLOPART_ONLY_BELOW = 0.35  # ... and only when the X-Part solid is this bad (its cuts are cleaner)
REGROWN_SHARE = 0.2       # an intruding region this big (x own open area) is a regrown neighbour, not a plug
NEIGHBOUR_COVER = 0.35    # ... or one that lies on this share of a single neighbour's open surface
SCORE_CANDIDATE = 0.8     # a large instance below this also gets a HoloPart draw to compare
SCORE_CANDIDATE_SMALL = 0.6  # a small one; X-Part is at home on small parts, so ask less often
SCORE_FLOOR = 0.3         # both below this: keep the open surface instead of either solid
SCORE_SAMPLES = 20000

# --- part exclusivity -------------------------------------------------------------------
# X-Part closes a part by regenerating it from the whole model, and it grows the
# neighbours back while it is at it: the dog's body came back with a second tail (4.2% of
# its surface lay on the tail part's own surface) and four paws. Nothing above noticed:
# the p90 of solid->surface distance is blind to anything under 10% of the surface, and
# the tail sits inside the body's box. "Intrusion" is the share of a solid's surface that
# lies on another instance's open surface while being off its own. Such a region is cut
# away and the hole capped when the cut leaves a short rim (a tail root, an ankle); a
# region with a long rim (a body's skin under a wrap-around armour) is left alone. What
# remains discounts the score.
INTRUSION_SCALE = 0.1      # this much residual intrusion zeroes the score
INTRUSION_MAX_RIM = 1.0    # cut a region only when its rim is shorter than this x diagonal
INTRUSION_MIN_FACES = 20
INTRUSION_CRUMB = 0.02     # a shell under this share of the area after the cut is debris (leg chips left on the body)
HOLLOW_FREE = 0.15       # this share of thin samples is normal (fingers, rims)
HOLLOW_SCALE = 0.5       # this much above HOLLOW_FREE zeroes the score


class OpenSurfaces:
    """Sampled points of every open instance surface, for own/other distance queries."""

    def __init__(self, opened, samples=SCORE_SAMPLES, seed=0):
        from scipy.spatial import cKDTree

        self.trees = {}
        self.rims = {}
        self.opened = {}
        points, labels = [], []
        for inst, node in opened.items():
            surface = node[1]
            if surface is None or not len(surface.faces) or surface.area <= 0:
                continue
            self.opened[inst] = surface
            pts = trimesh.sample.sample_surface(surface, samples, seed=seed)[0]
            self.trees[inst] = cKDTree(pts)
            rim = _boundary_edges(surface)
            if len(rim):
                self.rims[inst] = cKDTree(np.asarray(surface.vertices)[np.unique(rim)])
            points.append(pts)
            labels.append(np.full(len(pts), inst))
        self.all = cKDTree(np.concatenate(points)) if points else None
        self.all_labels = np.concatenate(labels) if labels else None

    def intruding(self, inst, points, tau):
        """Per point: off its own surface (> tau) yet within tau of any other instance.

        Within tau only other instances' points can be nearest once the own surface is
        farther than tau away, so one bounded nearest-neighbour query settles both."""
        points = np.asarray(points, dtype=float)
        if self.all is None or not len(points):
            return np.zeros(len(points), dtype=bool)
        own = (self.trees[inst].query(points)[0] if inst in self.trees
               else np.full(len(points), np.inf))
        near_any = self.all.query(points, distance_upper_bound=tau)[0] < tau
        return (own > tau) & near_any

    def nearest_instance(self, points):
        """Per point: the instance whose sampled surface is nearest (-1 when none)."""
        points = np.asarray(points, dtype=float)
        if self.all is None or not len(points):
            return np.full(len(points), -1)
        return self.all_labels[self.all.query(points)[1]]

    def near_rim(self, inst, points, reach):
        """Per point: within `reach` of the instance's own cut rim (the collar zone)."""
        points = np.asarray(points, dtype=float)
        if inst not in self.rims or not len(points):
            return np.zeros(len(points), dtype=bool)
        return self.rims[inst].query(points, distance_upper_bound=reach)[0] < reach


def _boundary_edges(mesh):
    """Directed edges (face winding) that belong to exactly one face."""
    edges = mesh.edges
    once = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
    return edges[once]


def _edge_loops(edges):
    """Chain directed edges a->b into closed loops of vertex indices."""
    nxt = {}
    for a, b in edges:
        nxt.setdefault(int(a), []).append(int(b))
    loops, seen = [], set()
    for a, b in edges:
        a = int(a)
        if (a, int(b)) in seen:
            continue
        loop, current = [a], a
        while True:
            candidates = [c for c in nxt.get(current, []) if (current, c) not in seen]
            if not candidates:
                break
            c = candidates[0]
            seen.add((current, c))
            if c == loop[0]:
                break
            loop.append(c)
            current = c
        if len(loop) >= 3 and current in nxt and loop[0] in nxt.get(current, []):
            loops.append(loop)
    return loops


CAP_RINGS = 4            # concentric rings a cap is built from (a fan when the loop is tiny)
CAP_SMOOTH = 25          # Laplacian passes over the cap's interior, rim fixed


def cap_loops(mesh, loops, rings=CAP_RINGS, smooth=CAP_SMOOTH):
    """Close each boundary loop with a smooth cap; returns a new mesh.

    The cap is built from concentric rings shrinking towards the loop's centroid, so the
    boundary edges stay exactly those of the hole (watertight), and its interior vertices
    are then Laplacian-smoothed with the rim fixed: a soap film over the hole instead of a
    fan, which on a non-planar rim showed every triangle as a facet."""
    if not loops:
        return mesh
    base = np.asarray(mesh.vertices)
    vertices = [base]
    faces = [np.asarray(mesh.faces)]
    offset = len(base)
    first_new = offset
    for loop in loops:
        ring = np.asarray(loop)
        n = len(ring)
        points = base[ring]
        centre = points.mean(axis=0)
        layers = [ring]
        if n >= 6 and rings >= 2:
            for k in range(1, rings):
                t = 1.0 - k / rings
                vertices.append(centre + (points - centre) * t)
                layers.append(offset + np.arange(n))
                offset += n
        vertices.append(centre[None])
        centre_index = offset
        offset += 1
        # a boundary edge a->b is traversed a->b by its face; the cap traverses it b->a
        for outer, inner in zip(layers[:-1], layers[1:]):
            outer_next, inner_next = np.roll(outer, -1), np.roll(inner, -1)
            faces.append(np.stack([outer_next, outer, inner], axis=1))
            faces.append(np.stack([outer_next, inner, inner_next], axis=1))
        last = layers[-1]
        faces.append(np.stack([np.roll(last, -1), last, np.full(n, centre_index)], axis=1))
    capped = trimesh.Trimesh(np.concatenate(vertices), np.concatenate(faces), process=False)
    if smooth > 0 and offset > first_new:
        positions = np.array(capped.vertices)
        movable = np.arange(first_new, offset)
        neighbours = capped.vertex_neighbors
        for _ in range(smooth):
            for v in movable:
                nb = neighbours[v]
                if nb:
                    positions[v] = positions[nb].mean(axis=0)
        capped = trimesh.Trimesh(positions, np.asarray(capped.faces), process=False)
    return capped


def cull_intrusions(solid, inst, surfaces, tau, diag, max_rim=INTRUSION_MAX_RIM,
                    min_faces=INTRUSION_MIN_FACES, band=RIM_BAND, box=None):
    """Cut the solid's regions that duplicate a neighbour; cap the holes. (mesh, cut share).

    Two kinds of region go. Outside the band along the part's own cut rim: an intruding
    region with a short rim (a regrown tail, a paw). Inside the band: an intruding region
    that is THIN (thin_faces) -- the plate or ring the generator copied from the
    neighbour's surface along the cut; the plug of a part closed through its cut also
    touches the neighbour there but is thick, so it stays (slicing a ring out of it left
    the plug's dome as a loose shell)."""
    if solid is None or not len(solid.faces) or surfaces is None:
        return solid, 0.0
    centres = solid.triangles_center
    intruding = surfaces.intruding(inst, centres, tau)
    in_band = surfaces.near_rim(inst, centres, band * diag)
    thin = thin_faces(solid, box) if box is not None and intruding.sum() >= min_faces         else np.zeros(len(centres), dtype=bool)
    # inside the band a thick intruding region is a plug and is spared -- unless it is big
    # enough to be a regrown neighbour (X-Part handed the dog's body the whole dog back)
    bad = intruding
    if bad.sum() < min_faces:
        return solid, 0.0
    own_area = float(surfaces.opened[inst].area) if inst in getattr(surfaces, "opened", {}) else float(solid.area)
    adjacency = np.asarray(solid.face_adjacency)
    both = bad[adjacency].all(axis=1)
    components = trimesh.graph.connected_components(
        adjacency[both], nodes=np.flatnonzero(bad), min_len=min_faces)
    remove = np.zeros(len(bad), dtype=bool)
    edge_vertices = np.asarray(solid.face_adjacency_edges)
    face_area = np.asarray(solid.area_faces)
    for component in components:
        in_comp = np.zeros(len(bad), dtype=bool)
        in_comp[np.asarray(component)] = True
        comp_area = float(face_area[in_comp].sum())
        thin_area = float(face_area[in_comp & thin].sum() / max(comp_area, 1e-12))
        if thin_area > 0.5:                       # a flange (by area) goes whatever its rim length
            remove |= in_comp
            continue
        if comp_area >= REGROWN_SHARE * own_area:  # a regrown neighbour, band or not
            remove |= in_comp
            continue
        # ... or a copy of most of one neighbour: the body carrying all four legs
        owners = surfaces.nearest_instance(centres[in_comp])
        covered = False
        for other in np.unique(owners):
            if other < 0 or other == inst or other not in getattr(surfaces, "opened", {}):
                continue
            on_other = float(face_area[in_comp][owners == other].sum())
            if on_other >= NEIGHBOUR_COVER * float(surfaces.opened[other].area):
                covered = True
                break
        if covered:
            remove |= in_comp
            continue
        if in_band[in_comp].mean() > 0.5:         # a thick region along the cut: the plug
            continue
        rim = in_comp[adjacency].sum(axis=1) == 1
        rim_length = float(np.linalg.norm(
            solid.vertices[edge_vertices[rim, 0]] - solid.vertices[edge_vertices[rim, 1]],
            axis=1).sum())
        if rim_length <= max_rim * diag:
            remove |= in_comp
    if not remove.any():
        return solid, 0.0
    before = {tuple(sorted(e)) for e in _boundary_edges(solid)}
    culled = trimesh.Trimesh(np.asarray(solid.vertices), np.asarray(solid.faces)[~remove],
                             process=False)
    culled.remove_unreferenced_vertices()
    new_edges = np.array([e for e in _boundary_edges(culled)
                          if tuple(sorted(e)) not in before], dtype=int).reshape(-1, 2)
    culled = drop_crumbs(cap_loops(culled, _edge_loops(new_edges)))
    share = float(solid.area_faces[remove].sum() / max(solid.area, 1e-12))
    return culled, share


def drop_crumbs(mesh, crumb=INTRUSION_CRUMB):
    """Drop shells under `crumb` of the area: slivers the cut left on the neighbour's side."""
    shells = mesh.split(only_watertight=False)
    if len(shells) <= 1:
        return mesh
    kept = [shell for shell in shells if shell.area >= crumb * mesh.area]
    if not kept or len(kept) == len(shells):
        return mesh
    return trimesh.util.concatenate(kept)


def solid_metrics(surface, solid, box, samples=SCORE_SAMPLES, tau=SCORE_TAU, seed=0,
                  surfaces=None, instance=None):
    """Fit of a generated solid to the open part surface it should close. None if unscorable.

    cover     share of surface samples within tau of the solid (did it keep the input?)
    fit_p90   p90 surface -> solid distance
    extra_p90 p90 solid -> surface distance (did it invent shape that is not there?)
    escape    how far the solid overruns the prompt box
    intrusion share of the solid's surface lying on another instance's surface, outside
              the band along its own cut rim where the collar conditioning puts some
              (needs `surfaces`, an OpenSurfaces, and the solid's `instance`)
    hollow    share of the solid's surface that is a thin double wall (see hollow_share)
    hollow_ref the same measured on the open surface: a shell-shaped part is allowed to be one
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
    intrusion = 0.0
    plug = None
    if surfaces is not None and instance is not None:
        bad = surfaces.intruding(instance, solid_points, tau * diag)
        outside = ~surfaces.near_rim(instance, solid_points, RIM_BAND * diag)
        intrusion = float(bad[outside].mean()) if outside.any() else 0.0
        if surfaces.all is not None:
            others = surfaces.all.data[surfaces.all_labels != instance]
            model_diag = float(np.linalg.norm(surfaces.all.data.max(axis=0) - surfaces.all.data.min(axis=0)))
            plug = plug_metrics(surface, solid, others, model_diag)
    return {
        **(plug or {}),
        "cover": float((fit < tau).mean()),
        "fit_p90": float(np.percentile(fit, 90)),
        "extra_p90": float(np.percentile(extra, 90)),
        "escape": float(box_escape(solid.bounds, box)),
        "largest_shell": largest_shell_share(solid),
        "intrusion": intrusion,
        "hollow": hollow_share(solid, box, dense=dense, samples=samples, seed=seed),
        # the open surface's own hollowness: an ear cup or a bowl IS a thin double wall,
        # and a solid that reproduces it is right, not hollow
        "hollow_ref": hollow_share(surface, box, samples=samples, seed=seed),
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


def quality_score(metrics, extra_scale=SCORE_EXTRA_SCALE, intrusion_scale=INTRUSION_SCALE,
                  hollow_free=HOLLOW_FREE, hollow_scale=HOLLOW_SCALE, escape_free=ESCAPE_FREE):
    """0..1: coverage, discounted by invented geometry, box escape beyond a free margin,
    tatters, intrusion and hollowness (a thin double wall where a filled part should be)."""
    if not metrics:
        return 0.0
    extra = min(1.0, metrics["extra_p90"] / extra_scale)
    escape = min(1.0, max(0.0, metrics["escape"] - escape_free) / max(1.0 - escape_free, 1e-9))
    whole = metrics.get("largest_shell", 1.0)
    intrusion = min(1.0, metrics.get("intrusion", 0.0) / intrusion_scale)
    hollow = min(1.0, max(0.0, metrics.get("hollow", 0.0)
                          - max(hollow_free, metrics.get("hollow_ref", 0.0))) / hollow_scale)
    # a socket where a plug should be: a watertight dish sunk below the cut scores well on
    # every distance metric and looks like a hole in the part
    plug = metrics.get("plug_mean")
    socket = 0.0 if plug is None else min(1.0, max(0.0, SOCKET_FREE - plug) / SOCKET_SCALE)
    return float(metrics["cover"] * (1.0 - extra) * (1.0 - escape) * whole
                 * (1.0 - intrusion) * (1.0 - hollow) * (1.0 - socket))


def _score_inputs(out_dir):
    with open(os.path.join(out_dir, "boxes.json"), "r", encoding="utf-8") as handle:
        boxes = json.load(handle)
    opened = index_by_instance(load_part_nodes(os.path.join(out_dir, "open_instances.glb")))
    xpart = index_by_instance(load_part_nodes(os.path.join(out_dir, "xpart_instances.glb")))
    return boxes, opened, xpart, OpenSurfaces(opened)


def prepare_solid(row, solid, surfaces, tau=SCORE_TAU):
    """Cut what duplicates a neighbour, then measure. (mesh, metrics, cut share)."""
    box = np.asarray(row["box"], dtype=float)
    diag = max(float(np.linalg.norm(box[1] - box[0])), 1e-9)
    culled, share = cull_intrusions(solid, row.get("instance"), surfaces, tau * diag, diag, box=box)
    return culled, share


def score_candidates(out_dir, candidate=SCORE_CANDIDATE, candidate_small=SCORE_CANDIDATE_SMALL):
    """Score every X-Part solid; flag the ones that should also get a HoloPart draw.

    A large instance (is_large: area or extent) is asked at `candidate`, a small one at
    `candidate_small`. On the decorated tree one flat 0.6 skipped the trunk (X-Part 0.70,
    a blob) while 0.8 everywhere sent half the ornaments to HoloPart for nothing.
    """
    boxes, opened, xpart, surfaces = _score_inputs(out_dir)
    src_ext = source_extent(boxes)
    decisions = []
    for row in boxes:
        inst = row.get("instance")
        surface, solid = opened.get(inst), xpart.get(inst)
        culled, cut = prepare_solid(row, solid and solid[1], surfaces)
        metrics = solid_metrics(surface and surface[1], culled, row["box"],
                                surfaces=surfaces, instance=inst)
        score = quality_score(metrics)
        large, _ = is_large(row, src_ext)
        threshold = candidate if large else candidate_small
        decisions.append({
            "name": row["name"], "instance": inst, "area_share": row["area_share"],
            "node": instance_node_name(row), "large": large, "threshold": threshold,
            "xpart": metrics, "xpart_cut": cut, "q_xpart": score,
            "candidate": score < threshold,
        })
    flagged = sum(d["candidate"] for d in decisions)
    cut = [d for d in decisions if d["xpart_cut"] > 0]
    if cut:
        print(f"[score] cut neighbour duplicates off {len(cut)} X-Part solid(s): "
              + ", ".join(f"{d['node']} {d['xpart_cut']:.1%}" for d in cut))
    print(f"[score] {flagged}/{len(decisions)} X-Part solids below their threshold "
          f"(large < {candidate}, small < {candidate_small}) -> HoloPart draw")
    return decisions


def apply_scored(out_dir, decisions, holopart_glb=None, floor=SCORE_FLOOR, margin=HOLOPART_MARGIN):
    """Keep the better solid per instance (HoloPart only when ahead of X-Part by `margin`:
    it hugs the open surface and always looks a little better by distance); below `floor`
    fall back to the open surface."""
    boxes, opened, xpart, surfaces = _score_inputs(out_dir)
    holo = index_by_instance(load_part_nodes(holopart_glb)) if holopart_glb else {}
    missing = [d["node"] for d in decisions if d["candidate"] and d["instance"] not in holo]
    if missing:
        # HoloPart skipped them (its draw failed); they are judged on X-Part alone
        print(f"[score] no HoloPart draw for {missing}; X-Part or the open surface it is")
        for d in decisions:
            if d["candidate"] and d["instance"] not in holo:
                d["candidate"] = False
                d["holopart_skipped"] = True
    names, solids = [], []
    for row, decision in zip(boxes, decisions):
        inst = row.get("instance")
        options = []
        if inst in xpart:
            culled, _ = prepare_solid(row, xpart[inst][1], surfaces)
            options.append(("xpart", decision["q_xpart"], culled))
        if decision["candidate"]:
            culled, cut = prepare_solid(row, holo[inst][1], surfaces)
            metrics = solid_metrics(opened[inst][1], culled, row["box"],
                                    surfaces=surfaces, instance=inst)
            decision["holopart"] = metrics
            decision["holopart_cut"] = cut
            decision["q_holopart"] = quality_score(metrics)
            options.append(("holopart", decision["q_holopart"], culled))
        if len(options) == 2 and (options[0][1] >= HOLOPART_ONLY_BELOW
                                  or options[1][1] < options[0][1] + margin):
            options = options[:1]          # X-Part unless it is near the floor AND HoloPart is clearly better
        backend, score, mesh = max(options, key=lambda option: option[1]) if options \
            else ("open", 0.0, None)
        if score < floor:
            if inst not in opened:
                continue
            backend, mesh = "open", opened[inst][1]
        decision["backend"], decision["score"] = backend, score
        names.append(row["name"])
        if backend != "open":
            mesh = drop_crumbs(mesh)      # HoloPart shards (a mouth in 1,649 pieces) go too
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
