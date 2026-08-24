"""Deterministic multi-view cameras and face-ID rasterisation.

SAM3 labels 2D pixels; to keep those labels we need to know exactly which
triangle each pixel belongs to. That requires the nvdiffrast camera to match
the Blender camera that produced the RGB view SAM3 was run on, so both are
built from the same primitives here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Blender's glTF importer converts the Y-up glTF frame to its own Z-up frame.
GLTF_TO_BLENDER = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64
)


def gltf_to_blender(vertices):
    return np.asarray(vertices, dtype=np.float64) @ GLTF_TO_BLENDER.T


def normalize_to_unit_cube(vertices):
    """Match BpyRenderer.normalize_scene: longest bbox edge becomes 1, bbox centred."""
    vertices = np.asarray(vertices, dtype=np.float64)
    low = vertices.min(axis=0)
    high = vertices.max(axis=0)
    scale = 1.0 / float((high - low).max())
    return (vertices - (low + high) / 2.0) * scale, scale


@dataclass(frozen=True)
class Camera:
    azimuth: float
    elevation: float
    position: np.ndarray

    @property
    def name(self) -> str:
        return f"az{self.azimuth:g}_el{self.elevation:g}"


def camera_ring(azimuths, elevations, radius=2.0):
    """Cameras on a lat/long grid in Blender space (Z up), all aimed at the origin.

    Deterministic and model-agnostic: no per-model "front" angle is needed because
    every part is visible from several of these views.
    """
    cameras = []
    for elevation in elevations:
        polar = math.radians(elevation)
        for azimuth in azimuths:
            yaw = math.radians(azimuth)
            position = radius * np.array(
                [
                    math.cos(polar) * math.cos(yaw),
                    math.cos(polar) * math.sin(yaw),
                    math.sin(polar),
                ],
                dtype=np.float64,
            )
            cameras.append(Camera(float(azimuth), float(elevation), position))
    return cameras


def look_at(eye, target=(0.0, 0.0, 0.0), world_up=(0.0, 0.0, 1.0)):
    """OpenGL-style view matrix. Blender's TRACK_TO keeps world Z up, so does this."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    world_up = np.asarray(world_up, dtype=np.float64)

    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    norm = np.linalg.norm(right)
    if norm < 1e-8:
        fallback = np.array([1.0, 0.0, 0.0]) if abs(forward[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, fallback)
        norm = np.linalg.norm(right)
    right /= norm
    up = np.cross(right, forward)

    view = np.eye(4, dtype=np.float64)
    view[0, :3] = right
    view[1, :3] = up
    view[2, :3] = -forward
    view[:3, 3] = -view[:3, :3] @ eye
    return view


def perspective(fov_x, near=0.05, far=20.0):
    """Square-sensor perspective matrix; Blender uses a 32mm square sensor here."""
    focal = 1.0 / math.tan(float(fov_x) / 2.0)
    proj = np.zeros((4, 4), dtype=np.float64)
    proj[0, 0] = focal
    proj[1, 1] = focal
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -2.0 * far * near / (far - near)
    proj[3, 2] = -1.0
    return proj


def rasterize_face_ids(
    vertices,
    faces,
    cameras,
    fov_x,
    resolution=512,
    chunk=1,
    world_up=(0.0, 0.0, 1.0),
    flip_rows=True,
    flip_cols=False,
):
    """Return int32 [views, resolution, resolution]: 0 background, else face index + 1.

    nvdiffrast writes rows bottom-up like OpenGL, so rows are flipped by default to
    match a saved PNG. `world_up` and `flip_cols` exist so the convention can be
    pinned down against a Blender render instead of assumed.
    """
    import torch
    import nvdiffrast.torch as dr

    device = torch.device("cuda")
    context = dr.RasterizeCudaContext(device=device)
    homogeneous = np.concatenate(
        [np.asarray(vertices, dtype=np.float32), np.ones((len(vertices), 1), dtype=np.float32)],
        axis=1,
    )
    vertices_t = torch.as_tensor(homogeneous, device=device)
    faces_t = torch.as_tensor(np.ascontiguousarray(faces, dtype=np.int32), device=device)
    projection = perspective(fov_x)

    buffers = []
    for start in range(0, len(cameras), chunk):
        batch = cameras[start : start + chunk]
        matrices = np.stack(
            [projection @ look_at(camera.position, world_up=world_up) for camera in batch]
        )
        clip = torch.einsum(
            "bij,vj->bvi", torch.as_tensor(matrices, dtype=torch.float32, device=device), vertices_t
        )
        raster, _ = dr.rasterize(
            context, clip.contiguous(), faces_t, resolution=[resolution, resolution]
        )
        ids = raster[..., 3].to(torch.int32)
        if flip_rows:
            ids = ids.flip(1)
        if flip_cols:
            ids = ids.flip(2)
        buffers.append(ids.cpu().numpy())
    return np.concatenate(buffers, axis=0)
