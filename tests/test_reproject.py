"""UV reprojection of completed solids onto the textured source (no Blender)."""
import os
import sys
import unittest

import numpy as np
import torch
import trimesh

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_toolkit.parts_rebake import reproject_uv  # noqa: E402


class _TrimeshBVH:
    """cumesh.cuBVH.unsigned_distance stand-in (CPU, trimesh closest point)."""

    def __init__(self, mesh):
        self.mesh = mesh

    def unsigned_distance(self, points, return_uvw=False):
        pts = points.cpu().numpy().astype(np.float64)
        closest, dist, fid = trimesh.proximity.closest_point(self.mesh, pts)
        tri = self.mesh.triangles[fid]
        uvw = trimesh.triangles.points_to_barycentric(tri, closest)
        out = (torch.tensor(dist), torch.tensor(fid), torch.tensor(uvw))
        return out if return_uvw else out[:2]


def _square():
    # z = 0 unit square, uv = (x, y)
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float64)
    f = np.array([[0, 1, 2], [0, 2, 3]], np.int64)
    return v, f, v[:, :2].copy()


class ReprojectUV(unittest.TestCase):
    def test_on_surface_face_gets_affine_uv(self):
        sv, sf, suv = _square()
        bvh = _TrimeshBVH(trimesh.Trimesh(sv, sf, process=False))
        pv = np.array([[0.2, 0.1, 0.0], [0.6, 0.2, 0.0], [0.5, 0.4, 0.0]])
        uv, stats = reproject_uv(pv, np.array([[0, 1, 2]]), sv, sf, suv, bvh, diag=np.sqrt(2))
        np.testing.assert_allclose(uv[0], pv[:, :2], atol=1e-5)
        self.assertEqual(stats["far_faces"], 0)

    def test_far_face_takes_one_nearest_colour(self):
        sv, sf, suv = _square()
        bvh = _TrimeshBVH(trimesh.Trimesh(sv, sf, process=False))
        # a cap floating 0.3 above the square: all corners share the nearest point's uv
        pv = np.array([[0.2, 0.2, 0.3], [0.8, 0.2, 0.3], [0.5, 0.8, 0.3]])
        uv, stats = reproject_uv(pv, np.array([[0, 1, 2]]), sv, sf, suv, bvh, diag=np.sqrt(2))
        self.assertEqual(stats["far_faces"], 1)
        np.testing.assert_allclose(uv[0], np.tile(pv[:, :2].mean(axis=0), (3, 1)), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
