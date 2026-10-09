import unittest

import numpy as np
import trimesh

from hybrid_complete import OpenSurfaces, cull_intrusions
from xpart_complete import thin_faces


def plate_on_box():
    """A unit cube (the plug-like solid) with a thin plate glued flat on its +Y face that
    sticks out sideways: the plate is thin, the cube is not."""
    cube = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    plate = trimesh.creation.box(extents=(2.0, 0.02, 1.0))
    plate.apply_translation((0.0, 0.51, 0.0))
    for _ in range(3):                      # a box is 12 triangles; the tests need a few hundred
        cube = cube.subdivide()
        plate = plate.subdivide()
    return cube, plate, trimesh.util.concatenate([cube, plate])


class ThinFacesTest(unittest.TestCase):
    def test_a_plate_is_thin_and_a_cube_is_not(self):
        cube, plate, solid = plate_on_box()
        thin = thin_faces(solid, solid.bounds)
        n_cube = len(cube.faces)
        self.assertLess(thin[:n_cube].mean(), 0.2)      # the cube's faces: an inward step stays inside
        big = np.abs(solid.face_normals[n_cube:, 1]) > 0.9  # the plate's top and bottom (its edges are many tiny faces)
        wings = np.abs(solid.triangles_center[n_cube:, 0]) > 0.5
        self.assertGreater(thin[n_cube:][big & wings].mean(), 0.9)   # out in the air: thin on both sides


class FlangeCullTest(unittest.TestCase):
    def test_a_thin_plate_on_the_neighbour_is_cut_but_the_block_stays(self):
        cube, plate, solid = plate_on_box()
        # own open surface: the cube minus its +Y face (that is the cut); the neighbour's
        # surface is a plane at y = 0.52 that the plate lies on
        own = cube.submesh([np.flatnonzero(cube.face_normals[:, 1] < 0.5)], append=True)
        plane = trimesh.creation.box(extents=(3.0, 0.001, 3.0)).apply_translation((0.0, 0.52, 0.0))
        surfaces = OpenSurfaces({0: ("own", own), 1: ("neighbour", plane)}, samples=4000)
        diag = float(np.linalg.norm(solid.bounds[1] - solid.bounds[0]))
        culled, share = cull_intrusions(solid, 0, surfaces, tau=0.02 * diag, diag=diag, box=solid.bounds)
        self.assertGreater(share, 0.25)                      # the plate (its wings and underside) went
        self.assertLess(len(culled.faces), len(solid.faces))   # faces were cut, not only capped
        self.assertGreater(culled.convex_hull.volume, 0.8)   # the block itself stayed


if __name__ == "__main__":
    unittest.main()
