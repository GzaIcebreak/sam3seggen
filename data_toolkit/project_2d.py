"""Project a 2D part-colour map back onto visible 3D faces.

SegviGen treats the map as a condition, not a hard constraint, so touching
parts (a hand on a staff) often share one predicted colour. Faces that are
visible in the same camera that produced the map can be labelled directly
from those pixels instead.
"""
from __future__ import annotations

import json
import math

import numpy as np
from PIL import Image


def look_at_axes(cam_pos, target, world_up):
    cam_z = np.asarray(cam_pos, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    cam_z = cam_z / np.linalg.norm(cam_z)
    cam_x = np.cross(world_up, cam_z)
    norm = np.linalg.norm(cam_x)
    if norm < 1e-8:
        fallback = np.array([1.0, 0.0, 0.0]) if abs(cam_z[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
        cam_x = np.cross(fallback, cam_z)
        norm = np.linalg.norm(cam_x)
    cam_x = cam_x / norm
    cam_y = np.cross(cam_z, cam_x)
    return cam_x, cam_y, cam_z


def camera_from_transforms(entry, azimuth=0.0, resolution=512):
    """Match data_toolkit/bpy_render.py: orbit the calibrated camera, track origin."""
    matrix = np.array(entry["transform_matrix"], dtype=np.float64)
    if azimuth:
        radians = math.radians(azimuth)
        cosine, sine = math.cos(radians), math.sin(radians)
        orbit = np.array(
            [[cosine, -sine, 0, 0], [sine, cosine, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float64,
        )
        matrix = orbit @ matrix
    cam_pos = matrix[:3, 3]
    # transforms.json is a Blender camera: Z-up world, TRACK_TO the origin with UP_Y, so
    # the camera's up is the world Z projected into its image plane.
    axes = look_at_axes(cam_pos, np.zeros(3), np.array([0.0, 0.0, 1.0]))
    lens = 16.0 / math.tan(float(entry["camera_angle_x"]) / 2.0)
    focal = (resolution / 32.0) * lens
    principal = resolution / 2.0
    return cam_pos, axes, focal, principal


def load_camera(transforms_path, azimuth=0.0, resolution=512):
    with open(transforms_path, "r", encoding="utf-8") as file:
        entry = json.load(file)[0]
    return camera_from_transforms(entry, azimuth=azimuth, resolution=resolution)


def segvigen_to_render_frame(points):
    """SegviGen export frame -> the Blender world the conditioning view was rendered in.

    Both are the source model fitted to a unit cube about the origin; they differ by how
    glTF's Y-up was turned into Z-up. Blender's importer maps (x, y, z) to (x, -z, y),
    to_glb bakes (x, z, -y), so one is the other turned half a turn about X.
    """
    return np.asarray(points, dtype=np.float64) * np.array([1.0, -1.0, -1.0])


def project_points(points, cam_pos, axes, focal, principal):
    """Pinhole projection: (u, v) pixel coordinates (v down) and depth in front of the camera.

    Points behind the camera get a negative depth; callers must drop them.
    """
    cam_x, cam_y, cam_z = axes
    rel = np.asarray(points, dtype=np.float64) - cam_pos
    depth = -(rel @ cam_z)
    safe = np.where(np.abs(depth) > 1e-9, depth, 1e-9)
    u = principal + focal * (rel @ cam_x) / safe
    v = principal - focal * (rel @ cam_y) / safe
    return u, v, depth


def map_labels_of_faces(centroids, canvas, palette, cam_pos, axes, focal, principal):
    """Palette index the map paints on each face, or -1 where the face is hidden or on background.

    A face is visible when its centroid wins a z-buffer pixel; each face therefore reads at
    most one pixel, so faces smaller than a pixel are sampled, never averaged.
    """
    n = len(centroids)
    seen = np.full(n, -1, dtype=np.int32)
    u, v, depth = project_points(centroids, cam_pos, axes, focal, principal)
    height, width = canvas.shape[:2]
    inside = (
        (depth > 1e-6)
        & (u >= 0) & (u < width)
        & (v >= 0) & (v < height)
    )
    if not inside.any():
        return seen

    ui = np.floor(u[inside]).astype(np.int32)
    vi = np.floor(v[inside]).astype(np.int32)
    di = depth[inside]
    fi = np.nonzero(inside)[0]
    pixel = vi * width + ui
    order = np.argsort(-di)
    winner = np.full(height * width, -1, dtype=np.int32)
    winner[pixel[order]] = fi[order]

    hit = winner >= 0
    if not hit.any():
        return seen
    face = winner[hit]
    vv, uu = np.divmod(np.nonzero(hit)[0], width)
    rgb = canvas[vv, uu].astype(np.float64)
    background = np.all(rgb >= 250, axis=1)
    face = face[~background]
    rgb = rgb[~background]
    if len(face) == 0:
        return seen
    seen[face] = np.argmin(np.linalg.norm(rgb[:, None, :] - palette[None], axis=2), axis=1)
    return seen


def label_visible_faces(centroids, labels, canvas, palette, cam_pos, axes, focal, principal):
    """Overwrite labels for faces that win a z-buffer pixel on `canvas`."""
    labels = np.asarray(labels).copy()
    seen = map_labels_of_faces(centroids, canvas, palette, cam_pos, axes, focal, principal)
    visible = seen >= 0
    labels[visible] = seen[visible]
    return labels


def face_map_labels(mesh, map_path, palette, transforms_path, azimuth,
                    to_render_frame=segvigen_to_render_frame):
    """map_labels_of_faces for a mesh in SegviGen's export frame and a conditioning-view map."""
    canvas = np.asarray(Image.open(map_path).convert("RGB"))
    vertices = to_render_frame(np.asarray(mesh.vertices))
    centroids = vertices[np.asarray(mesh.faces)].mean(axis=1)
    cam_pos, axes, focal, principal = load_camera(
        transforms_path, azimuth=azimuth, resolution=canvas.shape[0],
    )
    return map_labels_of_faces(centroids, canvas, palette, cam_pos, axes, focal, principal)


def label_mesh_from_map(mesh, labels, map_path, palette, transforms_path, azimuth,
                        to_render_frame=segvigen_to_render_frame):
    labels = np.asarray(labels).copy()
    seen = face_map_labels(mesh, map_path, palette, transforms_path, azimuth, to_render_frame)
    visible = seen >= 0
    labels[visible] = seen[visible]
    return labels
